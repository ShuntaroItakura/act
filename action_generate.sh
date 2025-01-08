#!/bin/bash

# 実行スクリプト名
SCRIPT="hsr_imitate_episodes1208generate_action.py"

# 固定の引数
TASK_NAME="sim_hsr_pick"
CKPT_DIR="ckpt_dir1208_p_c_ac50_epi10tmpout3000_gripper"
POLICY_CLASS="ACT"
KL_WEIGHT=10
CHUNK_SIZE=50
HIDDEN_DIM=512
BATCH_SIZE=10
DIM_FEEDFORWARD=3200
NUM_EPOCHS=3000
LR=1e-5
SEED=0
EVAL="--eval"

# --action_seed の値の範囲
ACTION_SEED_START=535
ACTION_SEED_END=555

# --action_save_dir_num の範囲
ACTION_SAVE_DIR_START=1
ACTION_SAVE_DIR_END=2

# ループで順番に実行
for ACTION_SEED in $(seq $ACTION_SEED_START $ACTION_SEED_END); do
    # for ACTION_SAVE_DIR_NUM in $(seq $ACTION_SAVE_DIR_START $ACTION_SAVE_DIR_END); do
        echo "Running with --action_seed $ACTION_SEED and --action_save_dir_num $ACTION_SEED"
        python3 $SCRIPT \
            --task_name $TASK_NAME \
            --ckpt_dir $CKPT_DIR \
            --policy_class $POLICY_CLASS \
            --kl_weight $KL_WEIGHT \
            --chunk_size $CHUNK_SIZE \
            --hidden_dim $HIDDEN_DIM \
            --batch_size $BATCH_SIZE \
            --dim_feedforward $DIM_FEEDFORWARD \
            --num_epochs $NUM_EPOCHS \
            --lr $LR \
            --seed $SEED \
            $EVAL \
            --action_seed $ACTION_SEED \
            --action_save_dir_num $ACTION_SEED
        python3 $SCRIPT \
            --task_name $TASK_NAME \
            --ckpt_dir $CKPT_DIR \
            --policy_class $POLICY_CLASS \
            --kl_weight $KL_WEIGHT \
            --chunk_size $CHUNK_SIZE \
            --hidden_dim $HIDDEN_DIM \
            --batch_size $BATCH_SIZE \
            --dim_feedforward $DIM_FEEDFORWARD \
            --num_epochs $NUM_EPOCHS \
            --lr $LR \
            --seed $SEED \
            $EVAL \
            --action_seed $ACTION_SEED \
            --action_save_dir_num $ACTION_SEED
    # done
done
