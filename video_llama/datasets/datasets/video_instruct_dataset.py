import os
from video_llama.datasets.datasets.base_dataset import BaseDataset
from video_llama.datasets.datasets.caption_datasets import CaptionDataset
import pandas as pd
import numpy as np
import decord
from decord import VideoReader
import random
import torch
from torch.utils.data.dataloader import default_collate
from PIL import Image
from typing import Dict, Optional, Sequence
import transformers
import pathlib
import json
import pickle
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
import copy
import math
from video_llama.processors import transforms_video,AlproVideoTrainProcessor
from torchvision import transforms
from video_llama.processors.video_processor import ToTHWC,ToUint8,load_video
from video_llama.conversation.conversation_video import Conversation,SeparatorStyle
import pdb

DEFAULT_IMAGE_PATCH_TOKEN = '<ImageHere>'
DEFAULT_VAD_CLS_TOKEN = '<ImageHere>'
video_conversation = Conversation(
    system="",
    roles=("Human", "Assistant"),
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)
IGNORE_INDEX = -100

class Video_Instruct_Dataset(BaseDataset):
    def __init__(self,vis_processor,text_processor,vis_root,ann_root,num_video_query_token=32,tokenizer_name = '/mnt/workspace/ckpt/vicuna-13b/',data_type = 'video'):
        """
        vis_root (string): Root directory of Llava images (e.g. webvid_eval/video/)
        ann_root (string): Root directory of video (e.g. webvid_eval/annotations/)
        split (string): val or test
        """
        super().__init__(vis_processor=vis_processor, text_processor=text_processor)

        data_path = pathlib.Path(ann_root)
        with data_path.open(encoding='utf-8') as f:
            self.annotation = json.load(f)
        
        self.num_video_query_token = num_video_query_token * 1
        self.vis_root = vis_root
        self.resize_size = 224
        self.num_frm = 16
        self.tokenizer = LlamaTokenizer.from_pretrained(tokenizer_name, use_fast=False)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        self.IMAGE_PATCH_TOKEN_ID = self.tokenizer.get_vocab()[DEFAULT_IMAGE_PATCH_TOKEN]

        self.transform = AlproVideoTrainProcessor(
            image_size=self.resize_size, n_frms = self.num_frm
        ).transform
        self.data_type = data_type
        self.video_info = self.load_video_info()
          
    def _get_video_path(self, sample):
        rel_video_fp = sample['video']
        full_video_fp = os.path.join(self.vis_root,  rel_video_fp)
        return full_video_fp

    def load_video_info(self):
        video_info = {}
        video_infofile = 'data/AnnotationTrainVideoList.txt'
        if video_infofile.endswith('txt'):
            for line in open(video_infofile):
                if line[-2] == ' ':
                    line = line[:-2]
                values = line.strip('\n').split(' ')

                if len(values) == 1:
                    continue
                # print(line)
                vid = values[0].split('/')[-1]
                video_info[vid] = []

                for i in range(1, len(values), 2):
                    # print(len(values),values[i],values[i+1])
                    video_info[vid].append([int(values[i]), int(values[i + 1])])
                    # pdb.set_trace()
            return video_info
        else:
            with open(video_infofile, 'rb') as f:
                video_dict = pickle.load(f)
            return video_dict
            
    def __getitem__(self, index):
        num_retries = 10  # skip error videos
        for _ in range(num_retries):
            try:
                sample = self.annotation[index]
                video_path = self._get_video_path(sample)
                video_path = video_path.replace('data/lvhui', 'storage/lvhui')
                if 'Normal' in video_path:
                    video_path = video_path.replace('frames/Training_Normal_Videos_Anomaly', 'normal')
                    label_vad = 0
                elif "frames" in video_path:
                    video_path = video_path.replace('frames', 'abnormal')
                    label_vad = 1
                else:
                    label_vad = 2
                conversation_list = sample['QA']
                
                videoarray = None

                video, msg, tid, conversation_list, tids = load_video(
                    video_path=video_path,
                    n_frms=self.num_frm,
                    height=self.resize_size,
                    width=self.resize_size,
                    ano_gt=videoarray,
                    conversation=conversation_list,
                    sampling="snippet_uniform", return_msg=True
                )
                video = self.transform(video)
                if 'cn' in self.data_type:
                    msg = ""
                # 添加视频<DEFAULT_IMAGE_PATCH_TOKEN>,以及msg到convsation list 0

                sources = preprocess_multimodal(copy.deepcopy(conversation_list), None,
                                                cur_token_len=self.num_video_query_token, msg=msg)
                new_sources = convert_source_vicuna_format(sources)

                data_dict = preprocess(
                    new_sources,
                    self.tokenizer)
                data_dict = dict(input_ids=data_dict["input_ids"][0],
                                 labels=data_dict["labels"][0])
                # image exist in the data

                data_dict['image'] = video
            except:
                print(f"Failed to load examples with video: {video_path}. "
                            f"Will randomly sample an example as a replacement.")
                index = random.randint(0, len(self) - 1)
                continue
            break
        else:  
            raise RuntimeError(f"Failed to fetch video after {num_retries} retries.")
        # "image_id" is kept to stay compatible with the COCO evaluation format
        return {
            "image": video,
            "text_input": data_dict["input_ids"],
            "labels": data_dict["labels"],
            "type":'video',
            "temporal_index": tid,
            "labels_vad": label_vad,
        }

    def __len__(self):
        return len(self.annotation)

    def collater(self, instances):
        input_ids, labels, tids, vads = tuple([instance[key] for instance in instances]
                                  for key in ("text_input", "labels", "temporal_index", "labels_vad"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            temporal_index=tids,
            labels_vad=vads,
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images
        batch['conv_type'] = 'multi'
        return batch


class Video_Instruct_Eval_Dataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_root, test_flag=False, num_video_query_token=32,
                 tokenizer_name='/mnt/workspace/ckpt/vicuna-13b/', data_type='video'):
        """
        vis_root (string): Root directory of Llava images (e.g. webvid_eval/video/)
        ann_root (string): Root directory of video (e.g. webvid_eval/annotations/)
        split (string): val or test
        """
        super().__init__(vis_processor=vis_processor, text_processor=text_processor)

        data_path = pathlib.Path(ann_root)
        with data_path.open(encoding='utf-8') as f:
             annos = json.load(f)

        self.num_video_query_token = num_video_query_token
        self.vis_root = vis_root
        self.resize_size = 224

        self.tokenizer = LlamaTokenizer.from_pretrained(tokenizer_name, use_fast=False)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        self.IMAGE_PATCH_TOKEN_ID = self.tokenizer.get_vocab()[DEFAULT_IMAGE_PATCH_TOKEN]

        self.test_flag = test_flag

        self.num_clip = 16
        self.clip_len = 8
        self.frame_interval = 8
        self.seg_interval = self.clip_len*self.frame_interval

        if self.test_flag:
            self.annotation = self.load_annotations(annos)
        else:
            self.annotation = annos

        self.transform = AlproVideoTrainProcessor(
            image_size=self.resize_size, n_frms=self.num_clip
        ).transform
        self.data_type = data_type

    def load_annotations(self, annos):
        segs = []
        vid = 0
        for sample in annos:
            video_path = self._get_video_path(sample)
            # pdb.set_trace()
            num_frames = sample['length']
            num_segs = math.ceil(num_frames / self.seg_interval)
            for i in range(num_segs):
                start = self.seg_interval * i
                end = min(self.seg_interval * (i + 1), num_frames)
                if end - start < self.clip_len:
                    continue
                if 'temporal_label' in sample:
                    seg = dict(video=video_path,temporal_label=sample['temporal_label'],QA=sample['QA'],
                           start=int(start),length=int(end)-int(start), vid=vid)
                else:
                    seg = dict(video=video_path, temporal_label=sample['video_label'], QA=sample['QA'],
                               start=int(start), length=int(end) - int(start), vid=vid)

                segs.append(seg)
            # pdb.set_trace()
            vid += 1
        return segs

    def _get_video_path(self, sample):
        rel_video_fp = sample['video']
        full_video_fp = os.path.join(self.vis_root, rel_video_fp)
        return full_video_fp

    def __getitem__(self, index):
        num_retries = 10  # skip error videos
        for _ in range(num_retries):
            try:
                sample = self.annotation[index]
                video_path = self._get_video_path(sample)
                conversation_list = sample['QA']
                if 'Normal' in video_path:
                    label_vad = 0
                else:
                    label_vad = 1
                video_path = video_path.replace('data/lvhui', 'storage/lvhui')
                if self.test_flag:
                    video, msg, tid = load_video(
                        video_path=video_path,
                        start=sample['start'],
                        vlen=sample['length'],
                        n_frms=self.clip_len,
                        height=self.resize_size,
                        width=self.resize_size,
                        sampling="uniform", return_msg=True
                    )
                else:
                    if 'Normal' in video_path:
                        video_path = video_path.replace('frames/Training_Normal_Videos_Anomaly', 'normal')
                        video_path = video_path.replace('frames/Normal', 'normal')
                    else:
                        video_path = video_path.replace('frames', 'abnormal')

                    video, msg, tid,_,_ = load_video(
                        video_path=video_path,
                        n_frms=self.num_clip,
                        height=self.resize_size,
                        width=self.resize_size,
                        sampling="snippet_uniform", return_msg=True
                    )

                video = self.transform(video)
                if 'cn' in self.data_type:
                    msg = ""
                # 添加视频<DEFAULT_IMAGE_PATCH_TOKEN>,以及msg到convsation list 0
                sources = preprocess_multimodal(copy.deepcopy(conversation_list), None,
                                                cur_token_len=self.num_video_query_token, msg=msg)
                new_sources = convert_source_vicuna_format(sources)

                data_dict = preprocess(
                    new_sources,
                    self.tokenizer)
                data_dict = dict(input_ids=data_dict["input_ids"][0],
                                 labels=data_dict["labels"][0])
                # image exist in the data
                data_dict['image'] = video
            except:
                print(f"Failed to load examples with video: {video_path}. "
                      f"Will randomly sample an example as a replacement.")
                index = random.randint(0, len(self) - 1)
                continue
            break
        else:
            raise RuntimeError(f"Failed to fetch video after {num_retries} retries.")
        # "image_id" is kept to stay compatible with the COCO evaluation format
        return {
            "image": video,
            "text_input": data_dict["input_ids"],
            "labels": data_dict["labels"],
            "type": 'video',
            "temporal_index": tid,
            "labels_vad": label_vad,
            "vid": video_path.split('/')[-1],
        }

    def __len__(self):
        return len(self.annotation)

    def collater(self, instances):
        input_ids, labels, tids, vads, vids = tuple([instance[key] for instance in instances]
                                           for key in ("text_input", "labels", "temporal_index", "labels_vad", "vid"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            temporal_index=tids,
            labels_vad=vads,
            vids=vids,
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images
        batch['conv_type'] = 'multi'
        return batch


def convert_source_vicuna_format(sources):
    new_sources = []
    for source in sources:
        new_source = []
        for i, sentence in enumerate(source):
            role_0_msg = sentence['q']
            role_1_msg = sentence['a']
            new_source.append({
                'from':'human',
                'value': role_0_msg,
            })
            new_source.append({
                'from':'gpt',
                'value': role_1_msg,
            })
        new_sources.append(new_source)
    return new_sources

def preprocess_multimodal(
    conversation_list: Sequence[str],
    multimodal_cfg: dict,
    cur_token_len: int,
    msg=''
) -> Dict:
    # 将conversational list中
    is_multimodal = True
    # image_token_len = multimodal_cfg['image_token_len']
    image_token_len = cur_token_len
    # conversation_list[0][
    #     "q"] = "<Video>" + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len + "</Video> " + msg + " " + \
    #            conversation_list[0]["q"]

    conversation_list[0]["q"] = "<Video>"+ DEFAULT_IMAGE_PATCH_TOKEN * image_token_len + DEFAULT_VAD_CLS_TOKEN * image_token_len +"</Video> " + msg +" "+ conversation_list[0]["q"]
    # pdb.set_trace()
    return [conversation_list]

def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "###"
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = video_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = video_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation
        
def _tokenize_fn(strings: Sequence[str],
                tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=512,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{video_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    conversations_tokenized = _tokenize_fn(conversations, tokenizer)
    input_ids = conversations_tokenized["input_ids"]
    targets = copy.deepcopy(input_ids)
    # print('0:', input_ids[0].shape, targets[0].shape)
    # print(conversations)
    for target, source in zip(targets, sources):
        tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source],
                                      tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)
    # pdb.set_trace()
    return dict(input_ids=input_ids, labels=targets)

def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len
