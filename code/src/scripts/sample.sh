# configs of different datasets
cfg=$1

# model settings
imgs_per_sent=16
cuda=True
gpu_id=0
transformer='xlm'
#check_point='state_epoch_1220.pth'

python src/sample.py \
        --cfg $cfg \
        --imgs_per_sent $imgs_per_sent \
        --cuda $cuda \
        --transformer $transformer \
        --gpu_id $gpu_id \

