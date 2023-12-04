"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import torch
from video_llama.common.registry import registry
from decord import VideoReader
import decord
import numpy as np
from PIL import Image
from video_llama.processors import transforms_video
from video_llama.processors.base_processor import BaseProcessor
from video_llama.processors.randaugment import VideoRandomAugment
from video_llama.processors import functional_video as F
from omegaconf import OmegaConf
from torchvision import transforms
import random as rnd
import os
import pdb

MAX_INT = registry.get("MAX_INT")
decord.bridge.set_bridge("torch")

def load_video(video_path, start=0, vlen=1, n_frms=MAX_INT, height=-1, width=-1, sampling="uniform", ano_gt=None, conversation=None, return_msg=False):

    if os.path.isdir(video_path):
        end = start + vlen
        n_frms = min(n_frms, vlen)

        if sampling == "uniform":
            indices = np.arange(start, end, vlen / n_frms).astype(int).tolist()
        else:
            raise NotImplementedError

        transform = transforms.Compose([transforms.PILToTensor()])
        imgs = []
        for ind in indices:
            # pdb.set_trace()
            image_path = os.path.join(video_path, 'img_{:08}.jpg'.format(ind))
            image = Image.open(image_path)#.resize((224,224))
            img_tensor = transform(image)
            imgs.append(img_tensor)
        frms = torch.stack(imgs, 1).float()

        # pdb.set_trace()
        if not return_msg:
            return frms
        sec = ", ".join([str(round(f/30,1)) for f in indices])
        msg = f"The video is {round(vlen/30,1)}s long, which contains {len(indices)} frames sampled at {sec} seconds. "
        return frms, msg, indices
    else:
        decord.bridge.set_bridge("torch")
        vr = VideoReader(uri=video_path, height=height, width=width)

        vlen = len(vr)
        start, end = 0, vlen

        n_frms = min(n_frms, vlen)

        if sampling == "uniform":
            indices = np.arange(start, end, vlen / n_frms).astype(int).tolist()
        elif sampling == "headtail":
            indices_h = sorted(rnd.sample(range(vlen // 2), n_frms // 2))
            indices_t = sorted(rnd.sample(range(vlen // 2, vlen), n_frms // 2))
            indices = indices_h + indices_t
        elif sampling == "snippet":
            num_frames = vlen
            num_clips = n_frms
            clip_len =  8
            frame_interval = 8

            ori_clip_len = clip_len * frame_interval
            avg_interval = (vlen - ori_clip_len + 1) // num_clips

            if avg_interval > 0:
                base_offsets = np.arange(num_clips) * avg_interval
                clip_offsets = base_offsets + np.random.randint(
                    avg_interval, size=num_clips)
            elif num_frames > max(num_clips, ori_clip_len):
                clip_offsets = np.sort(
                    np.random.randint(
                        num_frames - ori_clip_len + 1, size=num_clips))
            elif avg_interval == 0:
                ratio = (num_frames - ori_clip_len + 1.0) / num_clips
                clip_offsets = np.around(np.arange(num_clips) * ratio)
            else:
                clip_offsets = np.zeros((num_clips,), dtype=np.int)
            tmp_ind = np.arange(clip_len)[None, :] * frame_interval
            indices = clip_offsets[:, None]  + tmp_ind
            indices = indices.reshape((-1,)).tolist()

        elif sampling == "snippet_random":
            num_frames = vlen
            num_clips = n_frms
            clip_len =  8
            frame_interval = 8

            ori_clip_len = clip_len * frame_interval
            avg_interval = (num_frames - ori_clip_len + 1) / float(num_clips)
            if num_frames > ori_clip_len - 1:
                base_offsets = np.arange(num_clips) * avg_interval
                clip_offsets = base_offsets.astype(np.int32)
            else:
                clip_offsets = np.zeros((num_clips,), dtype=np.int32)
            if avg_interval < ori_clip_len:
                indices = clip_offsets[:, None] + np.arange(clip_len)[None, :] * frame_interval
            else:
                indices = clip_offsets[:, None] + np.random.randint(avg_interval, size=frame_interval)
            indices = indices.reshape((-1,)).tolist()

        elif sampling == "snippet_uniform":
            num_frames = vlen
            num_clips = n_frms
            clip_len =  8
            frame_interval = 8

            ori_clip_len = clip_len * frame_interval
            avg_interval = (num_frames - ori_clip_len + 1) / float(num_clips)
            if num_frames > ori_clip_len - 1:
                base_offsets = np.arange(num_clips) * avg_interval
                clip_offsets = base_offsets.astype(np.int32)
            else:
                clip_offsets = np.zeros((num_clips,), dtype=np.int32)

            if avg_interval < ori_clip_len:
                tmp_ind = np.arange(clip_len)[None, :] * frame_interval
            else:
                tmp_ind = np.arange(0,avg_interval,avg_interval/clip_len)[None, :]
                tmp_ind = tmp_ind.astype(np.int32)

            # pdb.set_trace()
            indices = clip_offsets[:, None] + tmp_ind
            indices = indices.astype(np.int32)
            indices = indices.reshape((-1,)).tolist()

        else:
            raise NotImplementedError

        if conversation is None or "Is there any anomaly" in conversation[0]['q']:
            # pdb.set_trace()
            # get_batch -> T, H, W, C
            # pdb.set_trace()
            temp_frms = vr.get_batch(indices)
            # print(type(temp_frms))
            tensor_frms = torch.from_numpy(temp_frms) if type(temp_frms) is not torch.Tensor else temp_frms
            frms = tensor_frms.permute(3, 0, 1, 2).float()  # (C, T, H, W)

            if not return_msg:
                return frms

            fps = float(vr.get_avg_fps())
            vid = video_path.split('/')[-2]
            if ano_gt is None:
                random_id = np.random.randint(0, clip_offsets.shape[0], 1)[0]
                # pdb.set_trace()
                # random_id = 0
                tids = clip_offsets[random_id] + tmp_ind
                tids = tids.reshape((-1,)).tolist()
                # print(tids)
                sec = ", ".join([str(round(f / fps, 1)) for f in tids])
                # " " should be added in the start and end
                msg = f"The video contains {len(tids)} frames sampled at {sec} seconds."
                
                if conversation is not None:
                    if conversation[0]['a'].startswith('No'):
                        conversation[0]['a'] = f'No, there are no anomalies from {round(tids[0] / 30, 1)}s to {round(tids[-1] / 30, 1)}s.'
                    else:
                        # pdb.set_trace()
                        conversation[0]['a'] = f'Yes, there exists '+vid+f' from {round(tids[0] / 30, 1)}s to {round(tids[-1] / 30, 1)}s.'
                return frms, msg, random_id, conversation, tids
            else:
                back_up = []
                for ind in range(clip_offsets.shape[0]):
                    tid_1 = clip_offsets[ind] + tmp_ind[0,0]
                    tid_2 = clip_offsets[ind] + tmp_ind[0,-1]
                    # print(tid_1,tid_2,ano_gt)
                    if ano_gt[tid_1:tid_2].sum()>(tid_2-tid_1)/2:
                        back_up.append(ind)
                if len(back_up)>0:
                    random_id = np.random.randint(0,len(back_up),1)[0]
                    tids = clip_offsets[back_up[random_id]] + tmp_ind
                    tids = tids.reshape((-1,)).tolist()

                    sec = ", ".join([str(round(f / fps, 1)) for f in tids])
                    # " " should be added in the start and end
                    msg = f"The video contains {len(tids)} frames sampled at {sec} seconds."

                    conversation[0]['a'] = f'Yes, there exists '+vid+f' from {round(tids[0]/30,1)}s to {round(tids[-1]/30,1)}s.'
                    # pdb.set_trace()

                    return frms, msg, back_up[random_id], conversation, tids
                else:
                    random_id = np.random.randint(0, clip_offsets.shape[0], 1)[0]
                    tids = clip_offsets[random_id] + tmp_ind
                    tids = tids.reshape((-1,)).tolist()

                    sec = ", ".join([str(round(f / fps, 1)) for f in tids])
                    # " " should be added in the start and end
                    msg = f"The video contains {len(tids)} frames sampled at {sec} seconds."
                    conversation[0]['a'] = f'Yes, there exists '+vid+f' from {round(tids[0] / 30, 1)}s to {round(tids[-1] / 30, 1)}s.'
                    # pdb.set_trace()
                    return frms, msg, random_id, conversation, tids
        else:
            # pdb.set_trace()
            clip_len = 8
            indices = np.arange(start, end, vlen / clip_len).astype(int).tolist()
            temp_frms = vr.get_batch(indices)
            temp_frms = np.tile(temp_frms, (n_frms,1,1,1))
            # print(type(temp_frms))
            tensor_frms = torch.from_numpy(temp_frms) if type(temp_frms) is not torch.Tensor else temp_frms
            frms = tensor_frms.permute(3, 0, 1, 2).float()  # (C, T, H, W)

            if not return_msg:
                return frms
            # pdb.set_trace()
            fps = float(vr.get_avg_fps())
            random_id = np.random.randint(0, n_frms, 1)[0]
            tids = indices
            sec = ", ".join([str(round(f / fps, 1)) for f in tids])
            # " " should be added in the start and end
            msg = f"The video contains {len(tids)} frames sampled at {sec} seconds."
            return frms, msg, random_id, conversation, tids



class AlproVideoBaseProcessor(BaseProcessor):
    def __init__(self, mean=None, std=None, n_frms=MAX_INT):
        if mean is None:
            mean = (0.48145466, 0.4578275, 0.40821073)
        if std is None:
            std = (0.26862954, 0.26130258, 0.27577711)

        self.normalize = transforms_video.NormalizeVideo(mean, std)

        self.n_frms = n_frms


class ToUint8(object):
    def __init__(self):
        pass

    def __call__(self, tensor):
        return tensor.to(torch.uint8)

    def __repr__(self):
        return self.__class__.__name__


class ToTHWC(object):
    """
    Args:
        clip (torch.tensor, dtype=torch.uint8): Size is (C, T, H, W)
    Return:
        clip (torch.tensor, dtype=torch.float): Size is (T, H, W, C)
    """

    def __init__(self):
        pass

    def __call__(self, tensor):
        return tensor.permute(1, 2, 3, 0)

    def __repr__(self):
        return self.__class__.__name__


class ResizeVideo(object):
    def __init__(self, target_size, interpolation_mode="bilinear"):
        self.target_size = target_size
        self.interpolation_mode = interpolation_mode

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (C, T, H, W)
        Returns:
            torch.tensor: central cropping of video clip. Size is
            (C, T, crop_size, crop_size)
        """
        return F.resize(clip, self.target_size, self.interpolation_mode)

    def __repr__(self):
        return self.__class__.__name__ + "(resize_size={0})".format(self.target_size)


@registry.register_processor("alpro_video_train")
class AlproVideoTrainProcessor(AlproVideoBaseProcessor):
    def __init__(
        self,
        image_size=384,
        mean=None,
        std=None,
        min_scale=0.5,
        max_scale=1.0,
        n_frms=MAX_INT,
    ):
        super().__init__(mean=mean, std=std, n_frms=n_frms)

        self.image_size = image_size

        self.transform = transforms.Compose(
            [
                # Video size is (C, T, H, W)
                transforms_video.RandomResizedCropVideo(
                    image_size,
                    scale=(min_scale, max_scale),
                    interpolation_mode="bicubic",
                ),
                ToTHWC(),  # C, T, H, W -> T, H, W, C
                ToUint8(),
                transforms_video.ToTensorVideo(),  # T, H, W, C -> C, T, H, W
                self.normalize,
            ]
        )

    def __call__(self, vpath):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (C, T, H, W)
        Returns:
            torch.tensor: video clip after transforms. Size is (C, T, size, size).
        """
        clip = load_video(
            video_path=vpath,
            n_frms=self.n_frms,
            height=self.image_size,
            width=self.image_size,
            sampling="headtail",
        )

        return self.transform(clip)

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", 256)

        mean = cfg.get("mean", None)
        std = cfg.get("std", None)

        min_scale = cfg.get("min_scale", 0.5)
        max_scale = cfg.get("max_scale", 1.0)

        n_frms = cfg.get("n_frms", MAX_INT)

        return cls(
            image_size=image_size,
            mean=mean,
            std=std,
            min_scale=min_scale,
            max_scale=max_scale,
            n_frms=n_frms,
        )


@registry.register_processor("alpro_video_eval")
class AlproVideoEvalProcessor(AlproVideoBaseProcessor):
    def __init__(self, image_size=256, mean=None, std=None, n_frms=MAX_INT):
        super().__init__(mean=mean, std=std, n_frms=n_frms)

        self.image_size = image_size

        # Input video size is (C, T, H, W)
        self.transform = transforms.Compose(
            [
                # frames will be resized during decord loading.
                ToUint8(),  # C, T, H, W
                ToTHWC(),  # T, H, W, C
                transforms_video.ToTensorVideo(),  # C, T, H, W
                self.normalize,  # C, T, H, W
            ]
        )

    def __call__(self, vpath):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (C, T, H, W)
        Returns:
            torch.tensor: video clip after transforms. Size is (C, T, size, size).
        """
        clip = load_video(
            video_path=vpath,
            n_frms=self.n_frms,
            height=self.image_size,
            width=self.image_size,
        )

        return self.transform(clip)

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", 256)

        mean = cfg.get("mean", None)
        std = cfg.get("std", None)

        n_frms = cfg.get("n_frms", MAX_INT)

        return cls(image_size=image_size, mean=mean, std=std, n_frms=n_frms)
