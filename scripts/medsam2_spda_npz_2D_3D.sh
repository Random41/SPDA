#!/usr/bin/env bash

python main.py spda \
  --model_cfg configs/sam2/sam2.1_hiera_t512.yaml \
  --checkpoint "path to checkpoint" \
  --data_dir "path to dataset" \
  --output_dir "path to output" \
  --lr 3.5e-5 --prompt_type box \
  --subsampled_prompt_type box --box_expand_ratio 0.105 \
  --spda_K_low 16 --spda_gamma 1.7 --spda_eta 0.1 --spda_beta 1.0
