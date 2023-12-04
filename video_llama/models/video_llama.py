import logging
import random

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from video_llama.common.registry import registry
from video_llama.models.blip2 import Blip2Base, disabled_train
from video_llama.models.modeling_llama import LlamaForCausalLM
# from video_llama.models.Qformer import BertEncoder
from transformers import LlamaTokenizer, BertConfig
from transformers import StoppingCriteria, StoppingCriteriaList
# from transformers.models.bert.modeling_bert import BertEncoder
import einops
import copy
from video_llama.models.Qformer import BertConfig, BertLMHeadModel
from video_llama.models.ImageBind.models.imagebind_model import ImageBindModel, ModalityType
from video_llama.models.ImageBind.models import imagebind_model
import pdb

# from flamingo_pytorch import PerceiverResampler
@registry.register_model("video_llama")
class VideoLLAMA(Blip2Base):
    """
    BLIP2 GPT-LLAMA model.
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_vicuna": "configs/models/video_llama.yaml",
    }

    @classmethod
    def init_video_Qformer(cls, num_query_token, vision_width, num_hidden_layers=2):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.num_hidden_layers = num_hidden_layers
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 1
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

    def __init__(
            self,
            vit_model="eva_clip_g",
            q_former_model="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth",
            img_size=224,
            drop_path_rate=0,
            use_grad_checkpoint=False,
            vit_precision="fp16",
            freeze_vit=True,
            freeze_qformer=True,
            num_query_token=32,

            llama_model="",
            prompt_path="",
            prompt_template="",
            max_txt_len=32,
            end_sym='\n',
            low_resource=False,  # use 8 bit and put vit in cpu
            device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.

            frozen_llama_proj=True,
            frozen_video_Qformer=True,
            frozen_audio_Qformer=True,

            clip_length=8,
            vad_decoder_model='',
            frozen_vad_decoder=False,
            frozen_LTC_fc=True,

            llama_proj_model='',
            fusion_header_type="seqTransf",
            max_frame_pos=32,
            fusion_head_layers=2,
            num_video_query_token=32,
            num_audio_query_token=8,
            imagebind_ckpt_path='/mnt/workspace/ckpt'
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()
        self.low_resource = low_resource

        print('Loading VIT')
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            for name, param in self.ln_vision.named_parameters():
                param.requires_grad = False
            self.ln_vision = self.ln_vision.eval()
            self.ln_vision.train = disabled_train
            logging.info("freeze vision encoder")
        print('Loading VIT Done')

        print('Loading Q-Former')
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        self.load_from_pretrained(url_or_filename=q_former_model)

        if freeze_qformer:
            for name, param in self.Qformer.named_parameters():
                param.requires_grad = False
            self.Qformer = self.Qformer.eval()
            self.Qformer.train = disabled_train
            self.query_tokens.requires_grad = False
            logging.info("freeze Qformer")
        logging.info('Loading Q-Former Done')

        logging.info('Loading LLAMA Tokenizer')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, use_fast=False)
        if self.llama_tokenizer.pad_token is None:
            self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        DEFAULT_IMAGE_PATCH_TOKEN = '<ImageHere>'
        self.llama_tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)

        self.IMAGE_PATCH_TOKEN_ID = self.llama_tokenizer.get_vocab()[DEFAULT_IMAGE_PATCH_TOKEN]

        logging.info('Loading LLAMA Model')
        if self.low_resource:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.bfloat16,
                load_in_8bit=True,
                device_map={'': device_8bit}
            )
        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.bfloat16,
            )

        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False
        logging.info('Loading LLAMA Done')

        logging.info('Loading Adaptor')
        self.llama_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.llama_model.config.hidden_size
        )
        self.vad_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.llama_model.config.hidden_size
        )
        if llama_proj_model:
            print("load llama proj weight: {}".format(llama_proj_model))
            llama_proj_weight = torch.load(llama_proj_model, map_location="cpu")
            msg = self.load_state_dict(llama_proj_weight['model'], strict=False)

        if frozen_llama_proj:
            #  todo frozen Adaptor
            for name, param in self.llama_proj.named_parameters():
                param.requires_grad = False
            for name, param in self.vad_proj.named_parameters():
                param.requires_grad = False
            logging.info('Adaptor is frozen')
        else:
            for name, param in self.llama_proj.named_parameters():
                param.requires_grad = True
            for name, param in self.vad_proj.named_parameters():
                param.requires_grad = True
            logging.info('Adaptor is not frozen')

        logging.info('Loading Adaptor Done')

        self.clip_length = clip_length
        logging.info('Loading VAD Decoder')
        self.LTC_fcs_L =  nn.Sequential(OrderedDict([
            ("fc_0", nn.Linear(self.Qformer.config.hidden_size, 64)),
            ("ln_0", nn.LayerNorm(64)),
            ("fc_1", nn.Linear(64, self.Qformer.config.hidden_size)),
        ]))
        self.LTC_fcs_S = nn.Sequential(OrderedDict([
            ("fc_0", nn.Linear(self.Qformer.config.hidden_size, 64)),
            ("ln_0", nn.LayerNorm(64)),
            ("fc_1", nn.Linear(64, self.Qformer.config.hidden_size)),
        ]))
        self.vad_decoder = nn.Sequential(OrderedDict([
            ("fc_0", nn.Linear(self.Qformer.config.hidden_size, 64)),
            ("ln_0", nn.LayerNorm(64)),
            ("fc_1", nn.Linear(64,2)),
        ]))
        if vad_decoder_model:
            print("load vad_decoder weight: {}".format(vad_decoder_model))
            vad_decoder_weight = torch.load(vad_decoder_model, map_location="cpu")
            msg = self.vad_decoder.load_state_dict(vad_decoder_weight['model'], strict=False)

        if frozen_vad_decoder:
            for name, param in self.vad_decoder.named_parameters():
                param.requires_grad = False
            logging.info('VAD_decoder is frozen')
        else:
           
            for name, param in self.vad_decoder.named_parameters():
                param.requires_grad = True
            logging.info('VAD_decoder is not frozen')

        if frozen_LTC_fc:
            for name, param in self.LTC_fcs_L.named_parameters():
                param.requires_grad = False
            for name, param in self.LTC_fcs_S.named_parameters():
                param.requires_grad = False
        else:
            for name, param in self.LTC_fcs_L.named_parameters():
                param.requires_grad = True
            for name, param in self.LTC_fcs_S.named_parameters():
                param.requires_grad = True

        logging.info('Loading VAD_decoder Done')

        self.max_txt_len = max_txt_len
        self.end_sym = end_sym

        if prompt_path:
            with open(prompt_path, 'r') as f:
                raw_prompts = f.read().splitlines()
            filted_prompts = [raw_prompt for raw_prompt in raw_prompts if "<ImageHere>" in raw_prompt]
            self.prompt_list = [prompt_template.format(p) for p in filted_prompts]
            print('Load {} training prompts'.format(len(self.prompt_list)))
            print('Prompt Example \n{}'.format(random.choice(self.prompt_list)))
        else:
            self.prompt_list = []

        self.video_frame_position_embedding = nn.Embedding(max_frame_pos, self.Qformer.config.hidden_size)

        self.num_video_query_token = num_video_query_token
        self.video_Qformer, self.video_query_tokens = self.init_video_Qformer(num_query_token=num_video_query_token, \
                                                                              vision_width=self.Qformer.config.hidden_size,
                                                                              num_hidden_layers=2)

        self.video_Qformer.cls = None
        self.video_Qformer.bert.embeddings.word_embeddings = None
        self.video_Qformer.bert.embeddings.position_embeddings = None
        for layer in self.video_Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        if frozen_video_Qformer:
            #  todo frozen  llama_proj
            for name, param in self.video_Qformer.named_parameters():
                param.requires_grad = False
            for name, param in self.video_frame_position_embedding.named_parameters():
                param.requires_grad = False
            self.video_query_tokens.requires_grad = False

            logging.info('video_Qformer is frozen')
        else:
            for name, param in self.video_Qformer.named_parameters():
                param.requires_grad = True
            for name, param in self.video_frame_position_embedding.named_parameters():
                param.requires_grad = True
            self.video_query_tokens.requires_grad = True
            logging.info('video_Qformer is not frozen')

        if not frozen_llama_proj:
            self.train_flag = 2  # adaptor
        elif frozen_LTC_fc:
            self.train_flag = 0  # only vador
        else:
            self.train_flag = 1  # co-train vador+LTC

    def vit_to_cpu(self):
        self.ln_vision.to("cpu")
        self.ln_vision.float()
        self.visual_encoder.to("cpu")
        self.visual_encoder.float()

    def encode_videoQformer_vad(self, image, his_hidden_state=None, next_video_ind=-1, tids=None, labels_vad=[0]):
        device = image.device

        # input shape b,c,t,h,w
        batch_size, _, time_length, _, _ = image.size()
        num_clip = time_length//self.clip_length
        image = einops.rearrange(image, 'b c t h w -> (b t) c h w')
        # pdb.set_trace()
        with self.maybe_autocast():
            # embed image features with blip2, out: (b t) q h
            image_embeds = self.ln_vision(self.visual_encoder(image)).to(device)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            # add frame_pos embedding
            position_ids = torch.arange(self.clip_length, dtype=torch.long, device=query_tokens.device)
            position_ids = position_ids.tile((num_clip,))
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            frame_position_embeddings = self.video_frame_position_embedding(position_ids)
            frame_position_embeddings = frame_position_embeddings.unsqueeze(-2)

            q_hidden_state = query_output.last_hidden_state
            frame_hidden_state = einops.rearrange(q_hidden_state, '(b t) q h -> b t q h', b=batch_size, t=time_length)

            logits, video_hidden = self.encode_videoQformer(frame_hidden_state, frame_position_embeddings, batch_size, num_clip, device)
            logits = einops.rearrange(logits, '(b c) q h -> b c q h', b=batch_size, c=num_clip)
            # pdb.set_trace()
            if self.train_flag==0:
                return logits

            if his_hidden_state is not None:
                frame_hidden_state_l = self.dual_memorybank(frame_hidden_state, logits, device, his_hidden_state,
                                                            next_video_ind, True)
                frame_hidden_state_s = self.his_memorybank(frame_hidden_state, logits, device, his_hidden_state,
                                                            next_video_ind, True)
            else:
                frame_hidden_state_l = self.dual_memorybank(frame_hidden_state, logits, device)
                frame_hidden_state_s = self.his_memorybank(frame_hidden_state, logits, device)

            frame_hidden_state_l = einops.rearrange(frame_hidden_state_l, 'b c (t q) h -> b (c t) q h', b=batch_size,
                                                     t=self.clip_length)
            frame_hidden_state_s = einops.rearrange(frame_hidden_state_s, 'b c (t q) h -> b (c t) q h', b=batch_size,
                                                    t=self.clip_length)

            frame_hidden_state_l = frame_hidden_state + self.LTC_fcs_L(frame_hidden_state_l)
            frame_hidden_state_s = frame_hidden_state + self.LTC_fcs_S(frame_hidden_state_s)

            logits_L, video_hidden_L = self.encode_videoQformer(frame_hidden_state_l, frame_position_embeddings, batch_size, num_clip,
                                                device)
            logits_L = einops.rearrange(logits_L, '(b c) q h -> b c q h', b=batch_size, c=num_clip)

            logits_S, video_hidden_S = self.encode_videoQformer(frame_hidden_state_s, frame_position_embeddings, batch_size, num_clip,
                                                device)
            logits_S = einops.rearrange(logits_S, '(b c) q h -> b c q h', b=batch_size, c=num_clip)

            if self.train_flag > 1:
                hidden = video_hidden
                ano_emb = video_hidden + video_hidden_L + video_hidden_S
                # pdb.set_trace()
                ano_emb = einops.rearrange(ano_emb, '(b c) q h -> b c q h', b=batch_size, c=num_clip)
                hidden = einops.rearrange(hidden, '(b c) q h -> b c q h', b=batch_size, c=num_clip)
                feas = []
                embs = []
                for id in range(len(tids)):
                    feas.append(hidden[id][tids[id]])
                    embs.append(ano_emb[id][tids[id]])
                hidden_feas = torch.stack(feas,0)
                ano_embs = torch.stack(embs,0)
                # pdb.set_trace()
                ano_embs = self.vad_proj(ano_embs)
                inputs_llama = self.llama_proj(hidden_feas)
                atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(device)

                return inputs_llama, atts_llama, ano_embs
            elif self.train_flag == 1:
                return logits, logits_L, logits_S, frame_hidden_state
            

    def his_memorybank(self, frame_hidden_state, logits, device, his_hidden_state=None, next_video_ind=-1, flag=False, LTC_len=3):

        scores_qry = F.softmax(logits.detach(), -1)
        scores_ano_qry = scores_qry[:, :, :, 1]
        max_prob_ano_qry, max_ind_qry = torch.max(scores_ano_qry, dim=-1)

        logits_seg = torch.gather(logits, 2, max_ind_qry[:, :, None, None].repeat((1, 1, 1, 2))).squeeze(2)

        batch_size, num_clip, _ = logits_seg.size()

        mask = torch.tril(torch.ones(num_clip, num_clip+LTC_len-1), diagonal=LTC_len-1).to(device) - \
               torch.tril(torch.ones(num_clip, num_clip+LTC_len-1), diagonal=-1).to(device)
        if next_video_ind > -1:
            mask[next_video_ind:,:(LTC_len+next_video_ind-1)] = 0
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1).unsqueeze(-1).unsqueeze(-1)

        frame_hidden_state = einops.rearrange(frame_hidden_state, 'b (c t) q h -> b c (t q) h', b=batch_size,
                                              c=num_clip)
        if flag:
            his_hidden_state = einops.rearrange(his_hidden_state, 'b (c t) q h -> b c (t q) h', b=batch_size, t=self.clip_length)
            frame_hidden_state_pad = his_hidden_state[:, (1-LTC_len):]
            # pdb.set_trace()
        else:
            frame_hidden_state_pad = torch.zeros_like(frame_hidden_state[:,:(LTC_len-1)])

        # if frame_hidden_state_pad.shape[2] != 256:
        #
        #     frame_hidden_state_pad = torch.zeros_like(frame_hidden_state[:,:(LTC_len-1)])

        his_hidden_state = torch.cat([frame_hidden_state_pad,frame_hidden_state],1)
        his_hidden_state = his_hidden_state.unsqueeze(1).expand(-1, num_clip, -1, -1, -1)

        his_hidden_state = his_hidden_state * mask

        his_hidden_state = einops.rearrange(his_hidden_state, 'b c t q h -> (b c) t (q h)', b=batch_size, c=num_clip)
        frame_hidden_state = einops.rearrange(frame_hidden_state, 'b (c 1) q h -> (b c) 1 (q h)', b=batch_size, c=num_clip)
        his_hidden_state = F.normalize(his_hidden_state,dim=-1)
        frame_hidden_state = F.normalize(frame_hidden_state, dim=-1)
        mat = torch.matmul(frame_hidden_state, his_hidden_state.permute((0, 2, 1)))
        # pdb.set_trace()
        frame_hidden_state_his = torch.matmul(mat, his_hidden_state)
        frame_hidden_state_his = einops.rearrange(frame_hidden_state_his, '(b c) 1 (q h) -> b (c 1) q h', b=batch_size, c=num_clip, h=768)
        # his_hidden_state = his_hidden_state.sum(2)/(mask.sum(2)+1e-6)
        return frame_hidden_state_his

    def dual_memorybank(self, frame_hidden_state, logits, device, his_hidden_state=None, next_video_ind=-1, flag=False, LTC_len=3):
        scores_qry = F.softmax(logits.detach(), -1)
        scores_ano_qry = scores_qry[:, :, :, 1]
        max_prob_ano_qry, max_ind_qry = torch.max(scores_ano_qry, dim=-1)

        logits_seg = torch.gather(logits, 2, max_ind_qry[:, :, None, None].repeat((1, 1, 1, 2))).squeeze(2)
        batch_size, num_clip, _ = logits_seg.size()

        scores_seg = F.softmax(logits_seg.detach(), -1)
        score_mask_ano = scores_seg[:,:,1]>=0.5
        score_mask_nor = ~score_mask_ano

        his_mask = torch.tril(torch.ones(num_clip, num_clip),diagonal=-1).to(device)
        his_mask = his_mask.unsqueeze(0).expand(batch_size, -1, -1)

        # ano segments mask
        score_mask_ano = score_mask_ano.unsqueeze(1).expand(-1, num_clip, -1)
        mask_ano = score_mask_ano * his_mask
        ano_score_his = scores_seg[:,:,1].unsqueeze(1).expand(-1, num_clip, -1,)
        ano_score_his = ano_score_his * mask_ano
        scores_k, ind_k = torch.topk(ano_score_his, LTC_len, dim=2)
        scores_thre = scores_k[:,:,-1:]
        ano_score_mask = ano_score_his > scores_thre
        ano_score_mask = ano_score_mask.unsqueeze(-1).unsqueeze(-1)

        # nor segments mask
        score_mask_nor = score_mask_nor.unsqueeze(1).expand(-1, num_clip, -1)
        mask_nor = score_mask_nor * his_mask
        nor_score_his = scores_seg[:, :, 0].unsqueeze(1).expand(-1, num_clip, -1)
        nor_score_his = nor_score_his * mask_nor
        scores_k, ind_k = torch.topk(nor_score_his, LTC_len, dim=2)
        scores_thre = scores_k[:, :, -1:]
        nor_score_mask = nor_score_his > scores_thre
        nor_score_mask = nor_score_mask.unsqueeze(-1).unsqueeze(-1)

        frame_hidden_state = einops.rearrange(frame_hidden_state, 'b (c t) q h -> b c (t q) h', b=batch_size,
                                              c=num_clip)
        his_hidden_state = frame_hidden_state.unsqueeze(1).expand(-1, num_clip, -1, -1, -1)
        dual_hidden_state = his_hidden_state * (ano_score_mask + nor_score_mask)

        his_hidden_state = einops.rearrange(dual_hidden_state, 'b c t q h -> (b c) t (q h)', b=batch_size, c=num_clip)
        frame_hidden_state = einops.rearrange(frame_hidden_state, 'b (c 1) q h -> (b c) 1 (q h)', b=batch_size, c=num_clip)

        his_hidden_state = F.normalize(his_hidden_state, dim=-1)
        frame_hidden_state = F.normalize(frame_hidden_state, dim=-1)

        mat = torch.matmul(frame_hidden_state, his_hidden_state.permute((0, 2, 1)))
        
        frame_hidden_state_his = torch.matmul(mat, his_hidden_state)
        frame_hidden_state_his = einops.rearrange(frame_hidden_state_his, '(b c) 1 (q h) -> b (c 1) q h', b=batch_size,
                                                  c=num_clip, h=768)
        return frame_hidden_state_his

    def encode_videoQformer(self, frame_hidden_state, frame_position_embeddings, batch_size, num_clip, device):
        # frame attention
        frame_hidden_state = frame_position_embeddings + frame_hidden_state
        frame_hidden_state = einops.rearrange(frame_hidden_state, 'b (c t) q h -> (b c) (t q) h', b=batch_size,
                                              c=num_clip)
        frame_atts = torch.ones(frame_hidden_state.size()[:-1], dtype=torch.long).to(device)
        video_query_tokens = self.video_query_tokens.expand(frame_hidden_state.shape[0], -1, -1)
        video_query_output = self.video_Qformer.bert(
            query_embeds=video_query_tokens,
            encoder_hidden_states=frame_hidden_state,
            encoder_attention_mask=frame_atts,
            return_dict=True,
        )

        video_hidden = video_query_output.last_hidden_state
        logits = self.vad_decoder(video_hidden)
        return logits, video_hidden

    def milandmargin_loss(self, logits, labels_binary, w_margin=1.0, w_smooth=0.01, w_sparse=0.01):
        with self.maybe_autocast():
            scores_qry = F.softmax(logits, -1)
            scores_ano_qry = scores_qry[:, :, :, 1]

            # scores_margin_seg = scores_ano_qry.max(-1)[0] - scores_ano_qry.min(-1)[0]
            max_prob_ano_qry, max_ind_qry = torch.max(scores_ano_qry, dim=-1)

            logits_seg = torch.gather(logits, 2, max_ind_qry[:, :, None, None].repeat((1, 1, 1, 2))).squeeze(2)

            scores_seg = F.softmax(logits_seg, -1)
            scores_ano_seg = scores_seg[:, :, 1]
            max_prob_ano_seg, max_ind_seg = torch.max(scores_ano_seg, dim=-1)

            logits_video = torch.gather(logits_seg, 1, max_ind_seg[:, None, None].repeat((1, 1, 2))).squeeze(1)

            max_prob_video, _ = torch.max(
                torch.gather(scores_seg, 1, max_ind_seg[:, None, None].repeat((1, 1, 2))).squeeze(1), dim=-1)

            loss_mil = F.cross_entropy(logits_video, labels_binary.long(), reduction='none')
            loss_mil = loss_mil * max_prob_video
            loss_mil = loss_mil.mean()

            margin_video = scores_ano_seg.max(-1)[0] - scores_ano_seg.min(-1)[0]
            loss_mar = margin_video * (1 - labels_binary) + labels_binary * (1 - margin_video)

            loss_mar = loss_mar * max_prob_video
            loss_mar = loss_mar.mean()

            scores_all = scores_ano_seg
            
            smooth_scores = (scores_all[:, 1:] - scores_all[:, :-1])
            smooth_loss = smooth_scores.pow(2).sum(dim=-1).mean()

            sparsity_loss = scores_all[:, :].sum(dim=-1).mean()
            
            return loss_mil + w_margin*loss_mar + w_smooth*smooth_loss + w_sparse*sparsity_loss

    def forward(self, samples):
        if 'conv_type' in samples.keys() and samples['conv_type'] == 'multi':

            im_patch_token_id = self.IMAGE_PATCH_TOKEN_ID
            image = samples["images"]
            input_ids = samples['input_ids']
            his_hidden_state = samples['his'] if not self.training else None
            next_video_ind = samples['next_video_ind'] if not self.training else -1
            if len(image.size()) == 4:
                time = 1
                image = einops.repeat(image, 'b c h w -> b c t h w', t=time)

            num_patch_tokens = self.num_video_query_token
            labels_binary =  torch.Tensor(samples["labels_vad"]).to(image.device)

            if self.train_flag < 1:
                logits = self.encode_videoQformer_vad(image, his_hidden_state, next_video_ind)

                if not self.training:
                    scores_qry = F.softmax(logits, -1)
                    scores_ano_qry = scores_qry[:, :, :, 1]
                    max_prob_ano_qry, max_ind_qry = torch.max(scores_ano_qry, dim=-1)
                    logits_seg = torch.gather(logits, 2, max_ind_qry[:, :, None, None].repeat((1, 1, 1, 2))).squeeze(2)
                
                    return {'y': logits_seg, 'his': None}
                
                vad_loss = self.milandmargin_loss(logits, labels_binary)

                return {"loss": vad_loss}

            elif self.train_flag == 1:
                
                logits, logits_l, logits_s, frame_hidden_state = self.encode_videoQformer_vad(image, his_hidden_state, next_video_ind)

                if not self.training:
                    scores_qry = F.softmax(logits, -1)
                    scores_ano_qry = scores_qry[:, :, :, 1]
                    max_prob_ano_qry, max_ind_qry = torch.max(scores_ano_qry, dim=-1)
                    logits_seg = torch.gather(logits, 2, max_ind_qry[:, :, None, None].repeat((1, 1, 1, 2))).squeeze(2)
                
                    scores_qry_l = F.softmax(logits_l, -1)
                    scores_ano_qry_l = scores_qry_l[:, :, :, 1]
                    max_prob_ano_qry_l, max_ind_qry_l = torch.max(scores_ano_qry_l, dim=-1)
                    logits_seg_l = torch.gather(logits_l, 2, max_ind_qry_l[:, :, None, None].repeat((1, 1, 1, 2))).squeeze(2)
                
                    scores_qry_s = F.softmax(logits_s, -1)
                    scores_ano_qry_s = scores_qry_s[:, :, :, 1]
                    max_prob_ano_qry_s, max_ind_qry_s = torch.max(scores_ano_qry_s, dim=-1)
                    logits_seg_s = torch.gather(logits_s, 2,
                                                max_ind_qry_s[:, :, None, None].repeat((1, 1, 1, 2))).squeeze(2)
                
                    return {'y': logits_seg, 'y_l': logits_seg_l, 'y_s': logits_seg_s, 'his': frame_hidden_state}
                    
                vad_loss = self.milandmargin_loss(logits, labels_binary) + \
                    self.milandmargin_loss(logits_l, labels_binary) + \
                    self.milandmargin_loss(logits_s, labels_binary)
                return {"loss": vad_loss}
                
            else:
                num_patch_tokens = self.num_video_query_token
                img_embeds, atts_img, ano_emb = self.encode_videoQformer_vad(image, his_hidden_state, next_video_ind, samples["temporal_index"], samples["labels_vad"])


            temp_input_ids = copy.deepcopy(input_ids)
            temp_input_ids[temp_input_ids == im_patch_token_id] = 0
            temp_input_embedding = self.llama_model.model.embed_tokens(temp_input_ids)

            new_input_embeds = []
            cur_image_idx = 0
            for cur_input_ids, cur_input_embeds in zip(input_ids, temp_input_embedding):
                cur_image_features = img_embeds[cur_image_idx]
                cur_ano_emb = ano_emb[cur_image_idx]

                if (cur_input_ids == im_patch_token_id).sum() != num_patch_tokens*2:
                    raise ValueError(
                        "The number of image patch tokens should be the same as the number of image patches.")
                masked_indices = torch.where(cur_input_ids == im_patch_token_id)[0]
                mask_index_start = masked_indices[0]
                if (masked_indices != torch.arange(mask_index_start, mask_index_start + num_patch_tokens*2,
                                                   device=masked_indices.device, dtype=masked_indices.dtype)).any():
                    raise ValueError("The image patch tokens should be consecutive.")

                cur_new_input_embeds = torch.cat((cur_input_embeds[:mask_index_start], cur_image_features, cur_ano_emb, cur_input_embeds[mask_index_start + num_patch_tokens*2:]), dim=0)
                
                new_input_embeds.append(cur_new_input_embeds)

                cur_image_idx += 1

            inputs_embeds = torch.stack(new_input_embeds, dim=0)
            
            targets = samples['labels']
            attention_mask = samples['attention_mask']
            with self.maybe_autocast():
                outputs = self.llama_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=targets,
                )
            loss = outputs.loss
            return {"loss": loss}

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        q_former_model = cfg.get("q_former_model",
                                 "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        llama_model = cfg.get("llama_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)
        freeze_qformer = cfg.get("freeze_qformer", True)
        low_resource = cfg.get("low_resource", False)
        device_8bit = cfg.get("device_8bit", 0)

        prompt_path = cfg.get("prompt_path", "")
        prompt_template = cfg.get("prompt_template", "")
        max_txt_len = cfg.get("max_txt_len", 32)
        end_sym = cfg.get("end_sym", '\n')

        frozen_llama_proj = cfg.get("frozen_llama_proj", True)
        frozen_vad_decoder = cfg.get("frozen_VAD_decoder", True)
        frozen_LTC_fc = cfg.get("frozen_LTC_fc", True)
        frozen_video_Qformer = cfg.get("frozen_video_Qformer", True)
        frozen_audio_Qformer = cfg.get("frozen_audio_Qformer", True)

        llama_proj_model = cfg.get("llama_proj_model", '')

        vad_decoder_model = cfg.get("VAD_decoder_model", '')

        fusion_header_type = cfg.get("fusion_header_type", 'seqTransf')
        max_frame_pos = cfg.get("max_frame_pos", 32)
        fusion_head_layers = cfg.get("fusion_head_layers", 2)
        num_video_query_token = cfg.get("num_video_query_token", 32)
        num_audio_query_token = cfg.get("num_audio_query_token", 8)
        imagebind_ckpt_path = cfg.get("imagebind_ckpt_path", '/mnt/workspace/ckpt')
        model = cls(
            vit_model=vit_model,
            q_former_model=q_former_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            freeze_qformer=freeze_qformer,
            num_query_token=num_query_token,
            llama_model=llama_model,
            prompt_path=prompt_path,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            low_resource=low_resource,
            device_8bit=device_8bit,
            fusion_header_type=fusion_header_type,
            max_frame_pos=max_frame_pos,
            fusion_head_layers=fusion_head_layers,
            frozen_llama_proj=frozen_llama_proj,
            frozen_video_Qformer=frozen_video_Qformer,
            frozen_audio_Qformer=frozen_audio_Qformer,
            num_video_query_token=num_video_query_token,
            num_audio_query_token=num_audio_query_token,
            imagebind_ckpt_path=imagebind_ckpt_path,
            vad_decoder_model=vad_decoder_model,
            frozen_vad_decoder=frozen_vad_decoder,
            frozen_LTC_fc=frozen_LTC_fc,
        )
        vad_ckpt_path = cfg.get("VAD_decoder_ckpt", "")
        if vad_ckpt_path:
            print("Load VAD Checkpoint: {}".format(vad_ckpt_path))
            ckpt = torch.load(vad_ckpt_path, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)
        ckpt_path = cfg.get("ckpt", "")  # load weights of MiniGPT-4
        if ckpt_path:
            print("Load Checkpoint: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)
        return model
