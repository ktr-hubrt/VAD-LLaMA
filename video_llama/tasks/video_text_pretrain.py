"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from video_llama.common.registry import registry
from video_llama.tasks.base_task import BaseTask


@registry.register_task("video_text_pretrain")
class VideoTextPretrainTask(BaseTask):
    def __init__(self):
        super().__init__()

    # def evaluation(self, model, data_loader, cuda_enabled=True):
    #     metric_logger = MetricLogger(delimiter="  ")
    #     header = "Evaluation"
    #     # TODO make it configurable
    #     print_freq = 10
    #
    #     results = []
    #
    #     for samples in metric_logger.log_every(data_loader, print_freq, header):
    #         samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
    #         pdb.set_trace()
    #         eval_output = self.valid_step(model=model, samples=samples)
    #         results.extend(eval_output)
    #
    #     if is_dist_avail_and_initialized():
    #         dist.barrier()
    #
    #     return results
