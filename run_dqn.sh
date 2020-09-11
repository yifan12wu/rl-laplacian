#!/bin/bash
python -u -B train_dqn.py \
--env_id=TwoRoom \
--log_sub_dir=test \
--args="device='cuda'"