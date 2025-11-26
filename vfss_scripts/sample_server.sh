
#!/bin/bash
IMAGE="medsegdiff-vfss:latest"

docker run -d\
  --gpus all \
  -v /home/caioseda/projetos/vfss-data-split/data:/workspace/vfss-data-split/data \
  -v /home/caioseda/projetos/MedSegDiff/data/out_sample_again:/workspace/medsegdiff/data/out_sample \
  --name medsegdiff_sample \
  -u $(id -u):$(id -g) \
  $IMAGE \
  python scripts/segmentation_sample.py \
    --data_name VFSS \
    --data_dir /workspace/vfss-data-split/data \
    --out_dir data/out_sample \
    --dpm_solver True \
    --diffusion_steps 100 \
    --model_path data/runs/out/savedmodel100000.pt \
    --image_size 256 \
    --num_channels 128 \
    --class_cond False \
    --num_res_blocks 2 \
    --num_heads 1 \
    --learn_sigma True \
    --use_scale_shift_norm False \
    --attention_resolutions 16 \
    --noise_schedule linear \
    --rescale_learned_sigmas False \
    --rescale_timesteps False \
    --num_ensemble 5