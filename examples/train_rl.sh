python dexmachina/rl/train_rl_games.py -B 4096 -obf -obt --max_epochs 5000 \
    --actuate_object --retarget_name para --horizon 16  -imw 0.5 --gain_mode all --curr_schedule uniform --wait_epochs 100 --learning_rate 0.0003 \
    --contact_beta 10 --upper_ratios 0.9 0.9 1 --lower_ratios 0.8 0.8 1 --save_freq 5000 --group_collisions --fixed_mode uniform --uniform_mode slow \
    --action_penalty 0.01 --dialback_ep_len 80 --skip_grad --deque_len 30 --task_rew_betas 10 1 5 --use_retarget_contact \
    --aux_reset_thres 0 0 0 --curr_rew_thres 0.6 0.01 0.01 0.01  -am hybrid --hybrid_scales 0.1 1.0 --kp_init 80 --kv_init 5 \
    --clip box-30-230 -imi 0.3 -bc 0.3 -con 3 -ert 0.6 -exp example
