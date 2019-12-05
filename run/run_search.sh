python3 -W ignore rl_quantize.py \
 --arch resnet50                \
 --dataset imagenet             \
 --suffix ratio010              \
 --preserve_ratio 0.1           \
 --n_worker 2                   \
 --data_bsize 256               \
 --no-finetune                  \
 --optimizer RS                 \
 --val_size 49800
