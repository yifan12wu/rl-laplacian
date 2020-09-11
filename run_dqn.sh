#!/bin/bash
python -u -B train_dqn.py \
--env_id=OneRoom \
--log_sub_dir=test \
--args="device='cuda'"