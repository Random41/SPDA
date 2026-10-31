#!/usr/bin/env bash

python main.py eval \
  --model_cfg configs/sam2/sam2.1_hiera_t512.yaml \
  --checkpoint "path to checkpoint" \
  --data_dir "path to dataset" \
  --output_dir "path to output" \
  --prompt_type box