#!/bin/bash

# スクリプトを実行中にエラーが発生した場合、すぐに終了する設定
set -e

# Pythonスクリプトの1つ目のコマンド
python3 hsr_imitate_episodes11090222.py --task_name sim_hsr_pick --ckpt_dir ckpt_dir1123_p_c_ac5_kl5 --policy_class ACT --kl_weight 5 --chunk_size 5 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 3000 --lr 1e-5 --seed 0

# Pythonスクリプトの1つ目のコマンド
python3 hsr_imitate_episodes11090222.py --task_name sim_hsr_pick --ckpt_dir ckpt_dir1123_p_c_ac5_kl100 --policy_class ACT --kl_weight 100 --chunk_size 5 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 3000 --lr 1e-5 --seed 0

# Pythonスクリプトの1つ目のコマンド
python3 hsr_imitate_episodes11090222.py --task_name sim_hsr_pick --ckpt_dir ckpt_dir1123_p_c_ac5_kl1 --policy_class ACT --kl_weight 1 --chunk_size 5 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 3000 --lr 1e-5 --seed 0