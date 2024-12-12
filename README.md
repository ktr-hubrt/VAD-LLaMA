This is an official implementation of VAD-LLaMA: Video Anomaly Detection and Explanation via Large Language Models. Pdf version is available at https://arxiv.org/abs/2401.05702.

    
# Environment Setup
To set up the environment, you can easily run the following command:
```
conda env create -f environment.yml
conda activate vadllama
```

# Data Preparation

Download the videos and labels for UCF-crime or TAD dataset and extract frames from videos in the link: https://github.com/ktr-hubrt/UMIL.

# Ckpt Preparation
Download the pre-trained weights of "llama_model, llama_proj_model, ckpt, imagebind_ckpt", following the instructions from link: https://github.com/DAMO-NLP-SG/Video-LLaMA.

If this does not work properly, generating you an error of the form 
```
huggingface_hub.utils._validators.HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '$CHECKPOINT$',
```

It is adviced to download the weights from HuggingFace (remembering to download all the files of the model, i.e. pth, json etc) or using 'https://github.com/lm-sys/FastChat', where you can download any weight using HuggingFace Hub or FastChat commands, such as running this command 'python3 -m fastchat.serve.cli --model-path lmsys/vicuna-7b-v1.5' and remembering to edit the 

```
video_llama_eval.yaml
```
Modifying accordingly the llama_model, llama_proj_model and imagebind_ckpt_path indicating the actual path of the model's weights. 

# Train
## Phase 1
Change the path in the 'TrainingPhase12_UCF.yaml' and set "frozen_LTC_fc: True, Then run:
```
bash run.sh
```

## Phase 2
Update the "VAD_decoder_ckpt" to the ckpt path of Phase 1 in the 'TrainingPhase12_UCF.yaml' and set "frozen_LTC_fc: False", Then run:
```
bash run.sh
```

## Testing in WSVAD
Change the "evaluate" to "True" and update the "VAD_decoder_ckpt" in 'TrainingPhase12_UCF.yaml', then run:
```
bash run.sh
```

## Phase 3
Update the "VAD_decoder_ckpt" to the ckpt path of Phase 2 in the 'TrainingPhase3.yaml' and run:
```
bash run.sh
```

# Test
Update the "ckpt" to the ckpt path of Phase 3, then run the demo of VAD-LLaM as:
```
bash demo.sh
```
