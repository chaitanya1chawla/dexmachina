python dexmachina/rl/train_rl_games.py -B 4096 -obf -obt --max_epochs 5000 \
    --actuate_object --retarget_name para --horizon 16  -imw 0.5 --gain_mode all --curr_schedule uniform --wait_epochs 100 --learning_rate 0.0003 \
    --contact_beta 10 --upper_ratios 0.9 0.9 1 --lower_ratios 0.8 0.8 1 --save_freq 5000 --group_collisions --fixed_mode uniform --uniform_mode slow \
    --action_penalty 0.01 --dialback_ep_len 80 --skip_grad --deque_len 30 --task_rew_betas 10 1 5 --use_retarget_contact \
    --aux_reset_thres 0 0 0 --curr_rew_thres 0.6 0.01 0.01 0.01  -am hybrid --hybrid_scales 0.1 1.0 --kp_init 80 --kv_init 5 \
    --clip box-30-230 -imi 0.3 -bc 0.3 -con 3 -ert 0.6 -exp example_slowdown_0.5 --slowdown_rate 0.5


# Eval - 
# use eval_rl_games.py instead here --
python dexmachina/rl/train_rl_games.py \  
    --num_envs 1 \
    -obf -obt \
    --actuate_object \
    --retarget_name para \
    --hand inspire_hand \
    --clip box-30-230 \
    --action_mode hybrid \
    --horizon 16 \
    --imi_rew_weight 0.3 \
    --contact_rew_weight 3.0 \
    --bc_rew_weight 0.3 \
    --early_reset_threshold 0.6 \
    --hybrid_scales 0.1 1.0 \
    --kp_init 80 \
    --kv_init 5 \
    --use_retarget_contact \
    --aux_reset_thres 0 0 0 \
    --curr_rew_thres 0.6 0.01 0.01 0.01  \
    --curr_schedule uniform \
    --checkpoint logs/rl_games/inspire_hand/inspire-example_box30-230-s01-u01_B4096_hybrid_thres0.6_ho16_imi0.3_con3.0_bc0.3/nn/inspire_hand.pth \
    --exp_name example_eval \
    --vis