#!/bin/bash
python -u -B train_laprepr.py \
--env_id=HardMaze \
--log_sub_dir=test \
--args="device='cuda'" \
--args="d=20" \
--args="w_neg=5.0" \
--args="c_neg=1.0" \
--args="reg_neg=0.0" \
--args="opt_args.lr=0.001" \
--args="n_samples=30000" \
--args="total_train_steps=50000"