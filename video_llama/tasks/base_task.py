"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import os
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from video_llama.common.dist_utils import get_rank, get_world_size, is_main_process, is_dist_avail_and_initialized
from video_llama.common.logger import MetricLogger, SmoothedValue
from video_llama.common.registry import registry
from video_llama.datasets.data_utils import prepare_sample
import pdb
import pickle
from einops import rearrange
from sklearn.metrics import roc_auc_score, roc_curve

class BaseTask:
    def __init__(self, **kwargs):
        super().__init__()

        self.inst_id_key = "instance_id"

    @classmethod
    def setup_task(cls, **kwargs):
        return cls()

    def build_model(self, cfg):
        model_config = cfg.model_cfg
        model_cls = registry.get_model_class(model_config.arch)
        return model_cls.from_config(model_config)

    def build_datasets(self, cfg):
        """
        Build a dictionary of datasets, keyed by split 'train', 'valid', 'test'.
        Download dataset and annotations automatically if not exist.

        Args:
            cfg (common.config.Config): _description_

        Returns:
            dict: Dictionary of torch.utils.data.Dataset objects by split.
        """

        datasets = dict()

        datasets_config = cfg.datasets_cfg

        assert len(datasets_config) > 0, "At least one dataset has to be specified."
        
        for name in datasets_config:
            dataset_config = datasets_config[name]

            builder = registry.get_builder_class(name)(dataset_config)
            dataset = builder.build_datasets()
            dataset['train'].name = name
            if 'sample_ratio' in dataset_config:
                dataset['train'].sample_ratio = dataset_config.sample_ratio
            if 'eval_anno_dir' in dataset_config['build_info']:
                dataset['eval'].name = name+'_eval'
                if 'sample_ratio' in dataset_config:
                    dataset['eval'].sample_ratio = dataset_config.sample_ratio
            datasets[name] = dataset
        # import pdb;pdb.set_trace()

        return datasets

    def train_step(self, model, samples):
        loss = model(samples)["loss"]
        return loss

    def valid_step(self, model, samples):
        raise NotImplementedError

    def before_evaluation(self, model, dataset, **kwargs):
        model.before_evaluation(dataset=dataset, task_type=type(self))

    def after_evaluation(self, val_result, split_name, epoch, dir, log, **kwargs):

        out_path = os.path.join(dir, 'result')

        if not os.path.exists(out_path):
            os.makedirs(out_path)
        with open(os.path.join(out_path, "epoch_{}.pickle".format(epoch)), 'wb') as handle:
            pickle.dump(val_result, handle, protocol=pickle.HIGHEST_PROTOCOL)
        logging.info(
            f'writing results to {out_path}'
        )
        # pdb.set_trace()
        return log

    def inference_step(self):
        raise NotImplementedError

    def evaluation(self, model, data_loader, test_flag=False, cur_epoch='tmp', cuda_enabled=True):
        metric_logger = MetricLogger(delimiter="  ")
        header = "Evaluation"
        # TODO make it configurable
        print_freq = 50
        w_cls = [1,1,1]

        if not hasattr(data_loader, "__next__"):
            # convert to iterator if not already
            data_loader = iter(data_loader)

        vid_list = []
        anno_file = '/storage/lvhui/Anomaly_Detection_splits/Anomaly_Test_multiclass.txt'

        scores_dict = dict()
        log_dict = dict()
        scores_dict['prd'] = dict()
        scores_dict['lmb'] = dict()
        scores_dict['smb'] = dict()
        his_hidden_state = None
        # pdb.set_trace()
        for samples in metric_logger.log_every(range(len(data_loader)), print_freq, header):
            samples = next(data_loader)
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)

            samples['his'] = his_hidden_state
            ind = -1
            samples['next_video_ind'] = ind

            if test_flag:
                img = rearrange(samples['images'], 'b c t h w -> 1 c (b t) h w')
                samples['images'] = img

                if samples["vids"][0] != samples["vids"][-1]:
                    for ind in range(len(samples["vids"])):
                        if samples["vids"][ind] != samples["vids"][0]:
                            samples['next_video_ind'] = ind
                            break
                    # pdb.set_trace()

            output = model(samples)
            scores_prd = F.softmax(output['y'], dim=-1)
            if 'y_l' in output:
                scores_lmb = F.softmax(output['y_l'], dim=-1)
                scores_smb = F.softmax(output['y_s'], dim=-1)
            else:
                scores_lmb = scores_smb = scores_prd

            if test_flag:
                his_hidden_state = output['his']

            scores_np_prd = scores_prd.cpu().data.numpy().copy()
            scores_np_lmb = scores_lmb.cpu().data.numpy().copy()
            scores_np_smb = scores_smb.cpu().data.numpy().copy()

            if test_flag:
                scores_np_prd = scores_np_prd[0]
                scores_np_lmb = scores_np_lmb[0]
                scores_np_smb = scores_np_smb[0]

            for ind in range(scores_np_prd.shape[0]):
                # pdb.set_trace()
                v_name = samples["vids"][ind]

                if v_name not in scores_dict['prd']:
                    scores_dict['prd'][v_name] = []
                    scores_dict['lmb'][v_name] = []
                    scores_dict['smb'][v_name] = []

                scores_dict['prd'][v_name].append(scores_np_prd[ind])
                scores_dict['lmb'][v_name].append(scores_np_lmb[ind])
                scores_dict['smb'][v_name].append(scores_np_smb[ind])

        tmp_dict = {}
        p_dict = {}
        l_dict = {}
        s_dict = {}
        for v_name in scores_dict["prd"].keys():
            p_scores = np.array(scores_dict["prd"][v_name]).copy()
            l_scores = np.array(scores_dict["lmb"][v_name]).copy()
            s_scores = np.array(scores_dict["smb"][v_name]).copy()
            # pdb.set_trace()
            if p_scores.shape[0] == 1:
                # 1,T,2
                tmp_dict[v_name] = [w_cls[0] * p_scores[0, :, 1] + w_cls[1] * l_scores[0, :, 1] + w_cls[2] * s_scores[0, :, 1]]
                p_dict[v_name] = [p_scores[0, :, 1]]
                l_dict[v_name] = [l_scores[0, :, 1]]
                s_dict[v_name] = [s_scores[0, :, 1]]
            else:
                # T,2
                tmp_dict[v_name] = [w_cls[0] * p_scores[:, 1] + w_cls[1] * l_scores[:, 1]+ w_cls[2] * s_scores[:, 1]]
                p_dict[v_name] = [p_scores[:, 1]]
                l_dict[v_name] = [l_scores[:, 1]]
                s_dict[v_name] = [s_scores[:, 1]]


        auc_all, auc_ano = self.evaluate_result(tmp_dict, anno_file)
        auc_all_p, auc_ano_p = self.evaluate_result(p_dict, anno_file)
        auc_all_l, auc_ano_l = self.evaluate_result(l_dict, anno_file)
        auc_all_s, auc_ano_s = self.evaluate_result(s_dict, anno_file)

        logging.info(
            f'AUC: [{auc_all:.3f}/{auc_ano:.3f}], P: [{auc_all_p:.3f}/{auc_ano_p:.3f}], L: [{auc_all_l:.3f}/{auc_ano_l:.3f}], S: [{auc_all_s:.3f}/{auc_ano_s:.3f}]\t'
        )

        log_dict['AUC_all'] = auc_all
        log_dict['AUC_ano'] = auc_ano

        if is_dist_avail_and_initialized():
            dist.barrier()
        # pdb.set_trace()
        return scores_dict, log_dict

    def evaluate_result(self, vid2abnormality, anno_file, root=''):
        LABEL_PATH = anno_file
        gt = []
        ans = []
        GT = []
        ANS = []
        video_path_list = []
        videos = {}
        for video in open(LABEL_PATH):
            vid = video.strip().split(' ')[0].split('/')[-1]
            video_len = int(video.strip().split(' ')[1])
            sub_video_gt = np.zeros((video_len,), dtype=np.int8)
            anomaly_tuple = video.split(' ')[3:]
            for ind in range(len(anomaly_tuple) // 2):
                start = int(anomaly_tuple[2 * ind])
                end = int(anomaly_tuple[2 * ind + 1])
                if start > 0:
                    sub_video_gt[start:end] = 1
            videos[vid] = sub_video_gt

        for vid in videos:

            if vid not in vid2abnormality.keys():
                # print("The video %s is excluded on the result!" % vid)
                continue

            cur_ab = np.array(vid2abnormality[vid])
            if cur_ab.shape[0] == 1:
                cur_ab = cur_ab[0, :, ]
            else:
                cur_ab = cur_ab[:, 0, ]
            cur_gt = np.array(videos[vid])
            ratio = float(len(cur_gt)) / float(len(cur_ab))
            cur_ans = np.zeros_like(cur_gt, dtype='float32')
            for i in range(len(cur_ab)):
                b = int(i * ratio + 0.5)
                e = int((i + 1) * ratio + 0.5)
                cur_ans[b: e] = cur_ab[i]

            cur_ans = self.postpress(cur_ans, seg_size=32)

            if cur_gt.max() >= 1:
                gt.extend(cur_gt.tolist())
                ans.extend(cur_ans.tolist())

            GT.extend(cur_gt.tolist())
            ANS.extend(cur_ans.tolist())

        ret = roc_auc_score(gt, ans)
        Ret = roc_auc_score(GT, ANS)
        fpr, tpr, threshold = roc_curve(GT, ANS)
        
        if root != '':
            output_file = path + "AUC.npz"
            np.savez(output_file, fpr=fpr, tpr=tpr, thre=threshold)

        return Ret, ret

    def postpress(self, curve, seg_size=32):
        leng = curve.shape[0]
        window_size = leng // seg_size
        new_curve = np.zeros_like(curve)
        for i in range(seg_size):
            new_curve[window_size * i:window_size * (i + 1)] = np.mean(curve[window_size * i:window_size * (i + 1)])
        if leng > window_size * seg_size:
            new_curve[seg_size * window_size:] = np.mean(curve[seg_size * window_size:])
        return new_curve

    def train_epoch(
        self,
        epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        cuda_enabled=False,
        log_freq=50,
        accum_grad_iters=1,
    ):
        return self._train_inner_loop(
            epoch=epoch,
            iters_per_epoch=lr_scheduler.iters_per_epoch,
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
            log_freq=log_freq,
            cuda_enabled=cuda_enabled,
            accum_grad_iters=accum_grad_iters,
        )

    def train_iters(
        self,
        epoch,
        start_iters,
        iters_per_inner_epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        cuda_enabled=False,
        log_freq=50,
        accum_grad_iters=1,
    ):
        return self._train_inner_loop(
            epoch=epoch,
            start_iters=start_iters,
            iters_per_epoch=iters_per_inner_epoch,
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
            log_freq=log_freq,
            cuda_enabled=cuda_enabled,
            accum_grad_iters=accum_grad_iters,
        )

    def _train_inner_loop(
        self,
        epoch,
        iters_per_epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        start_iters=None,
        log_freq=50,
        cuda_enabled=False,
        accum_grad_iters=1,
    ):
        """
        An inner training loop compatible with both epoch-based and iter-based training.

        When using epoch-based, training stops after one epoch; when using iter-based,
        training stops after #iters_per_epoch iterations.
        """
        use_amp = scaler is not None

        if not hasattr(data_loader, "__next__"):
            # convert to iterator if not already
            data_loader = iter(data_loader)

        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("epoch", SmoothedValue(window_size=1, fmt="{value:.1f}"))
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))

        # if iter-based runner, schedule lr based on inner epoch.
        logging.info(
            "Start training epoch {}, {} iters per inner epoch.".format(
                epoch, iters_per_epoch
            )
        )
        header = "Train: data epoch: [{}]".format(epoch)
        if start_iters is None:
            # epoch-based runner
            inner_epoch = epoch
        else:
            # In iter-based runner, we schedule the learning rate based on iterations.
            inner_epoch = start_iters // iters_per_epoch
            header = header + "; inner epoch [{}]".format(inner_epoch)

        for i in metric_logger.log_every(range(iters_per_epoch), log_freq, header):
            # if using iter-based runner, we stop after iters_per_epoch iterations.
            if i >= iters_per_epoch:
                break

            samples = next(data_loader)
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            samples.update(
                {
                    "epoch": inner_epoch,
                    "num_iters_per_epoch": iters_per_epoch,
                    "iters": i,
                }
            )

            lr_scheduler.step(cur_epoch=inner_epoch, cur_step=i)

            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = self.train_step(model=model, samples=samples)

            # after_train_step()
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # update gradients every accum_grad_iters iterations
            if (i + 1) % accum_grad_iters == 0:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()                     
                else:    
                    optimizer.step()
                optimizer.zero_grad()

            metric_logger.update(epoch=epoch)
            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # after train_epoch()
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        logging.info("Averaged stats: " + str(metric_logger.global_avg()))
        return {
            k: "{:.8f}".format(meter.global_avg)
            for k, meter in metric_logger.meters.items()
        }

    @staticmethod
    def save_result(result, result_dir, filename, remove_duplicate=""):
        import json

        result_file = os.path.join(
            result_dir, "%s_rank%d.json" % (filename, get_rank())
        )
        final_result_file = os.path.join(result_dir, "%s.json" % filename)

        json.dump(result, open(result_file, "w"))

        if is_dist_avail_and_initialized():
            dist.barrier()

        if is_main_process():
            logging.warning("rank %d starts merging results." % get_rank())
            # combine results from all processes
            result = []

            for rank in range(get_world_size()):
                result_file = os.path.join(
                    result_dir, "%s_rank%d.json" % (filename, rank)
                )
                res = json.load(open(result_file, "r"))
                result += res

            if remove_duplicate:
                result_new = []
                id_list = []
                for res in result:
                    if res[remove_duplicate] not in id_list:
                        id_list.append(res[remove_duplicate])
                        result_new.append(res)
                result = result_new

            json.dump(result, open(final_result_file, "w"))
            print("result file saved to %s" % final_result_file)

        return final_result_file
