# configs of different datasets
cfg=$1

# model settings
imgs_per_sent=16
cuda=True
gpu_id=0
use_transformer=True
transformer_type='roberta'

python src/sample.py \
        --cfg $cfg \
        --imgs_per_sent $imgs_per_sent \
        --cuda $cuda \
        --gpu_id $gpu_id \
        --use_transformer  $use_transformer \
        --transformer_type $transformer_type
