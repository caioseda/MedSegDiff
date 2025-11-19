
#!/bin/bash
IMAGE="medsegdiff-vfss:latest"

docker run \
  --gpus all \
  -v /home/caioseda/projetos/vfss-data-split/data:/workspace/vfss-data-split/data \
  -v /home/caioseda/projetos/MedSegDiff/data/vfss/out:/workspace/medsegdiff/data/vfss/out \
  --name medsegdiff_train \
  -u $(id -u):$(id -g) \
  $IMAGE \
  python scripts/segmentation_train.py \
    --data_name VFSS \
    --data_dir /workspace/vfss-data-split/data \
    --out_dir data/vfss/out \
    --image_size 256 \
    --num_channels 128 \
    --class_cond False \
    --num_res_blocks 2 \
    --num_heads 1 \
    --learn_sigma True \
    --use_scale_shift_norm False \
    --attention_resolutions 16 \
    --diffusion_steps 1000 \
    --noise_schedule linear \
    --rescale_learned_sigmas False \
    --rescale_timesteps False \
    --lr 1e-4 \
    --batch_size 8