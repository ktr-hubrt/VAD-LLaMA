#Phase 1
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_addr="0.0.0.0" --master_port=9333 train.py --cfg-path ./train_configs/TrainingPhase12_UCF.yaml
#Phase 2
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_addr="0.0.0.0" --master_port=9333 train.py --cfg-path ./train_configs/TrainingPhase12_UCF.yaml
#Phase 3
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_addr="0.0.0.0" --master_port=3333 train.py --cfg-path ./train_configs/TrainingPhase3.yaml