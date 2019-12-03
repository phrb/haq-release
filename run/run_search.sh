python -W ignore rl_quantize.py \
 --arch resnet50                \
 --dataset imagenet             \
 --suffix ratio010              \
 --preserve_ratio 0.1           \
 --n_worker 4                  \
 --data_bsize 256               \
 --no-finetune \
 --val_size 49900               \
 --gpu_id 0,1,2,3               \
