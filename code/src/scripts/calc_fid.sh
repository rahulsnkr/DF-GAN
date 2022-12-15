# configs of different datasets
cfg=$1
batch_size_per_gpu=32
transformer='xlm'
# DDP settings
multi_gpus=True
master_port=11122
#check_point='state_epoch_1220.pth'
# You can set CUDA_VISIBLE_DEVICES=0,1,2..., node=number_of_GPUs to accelerate the evaluation process if you have multiple GPUs
nodes=4
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=$nodes --master_port $master_port src/test_FID.py \
                    --cfg $cfg \
                    --batch_size $batch_size_per_gpu \
                    --transformer $transformer \
                    --multi_gpus $multi_gpus \
