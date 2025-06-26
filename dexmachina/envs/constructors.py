import os  
import argparse
import numpy as np
from dexmachina.envs.base_env import BaseEnv, get_env_cfg
from dexmachina.envs.randomizations import get_randomization_cfg
from dexmachina.envs.curriculum import get_curriculum_cfg
from dexmachina.envs.maniptrans_curr import get_maniptrans_cfg
from dexmachina.envs.object import ArticulatedObject, get_arctic_object_cfg
from dexmachina.envs.robot import BaseRobot, get_default_robot_cfg
from dexmachina.envs.rewards import RewardModule, get_reward_cfg
from dexmachina.envs.demo_data import get_demo_data, load_genesis_retarget_data 

def parse_clip_string(clip):
    vals = clip.split('-')
    if len(vals) == 3:
        print("Default to using subject s01 and clip 01")
        vals += ['s01', 'u01']
    assert len(vals) == 5, "Clip should be in format: obj_name-start-end-subject-clip"
    obj_name = vals[0]
    start = int(vals[1])
    end = int(vals[2])
    subject = vals[3]
    use_clip = vals[4].replace('u', '') # just 01/02
    return obj_name, start, end, subject, use_clip

def get_all_env_cfg(args, device, load_retarget_data=True):
    num_envs = args.num_envs  
    # check if args has this attribute:
    last_n = args.last_n_frame if hasattr(args, 'last_n_frame') else -1
    reward_cfg = get_reward_cfg(last_n_frame=last_n)
    reward_cfg['imi_rew_weight'] = args.imi_rew_weight
    reward_cfg['imi_wrist_weight'] = args.imi_wrist_weight
    reward_cfg['contact_rew_weight'] = args.contact_rew_weight
    reward_cfg['contact_rew_function'] = args.contact_rew_function
    reward_cfg['contact_beta'] = args.contact_beta
    reward_cfg['wrist_frame_contact'] = not args.skip_wrist_frame_contact
    reward_cfg['mask_zero_contact'] = not args.nomask_zero_contact
    reward_cfg['use_retarget_contact'] = args.use_retarget_contact 
    reward_cfg['retarget_objframe'] = not args.retarget_worldfr
    reward_cfg['action_penalty'] = args.action_penalty

    if args.objdex_baseline:
        print("Setting action mode to hybrid for objdex baseline and task rew beta to lower")
        args.action_mode = 'hybrid' 
        args.task_rew_betas = [30, 2, 10]
    
    task_betas = args.task_rew_betas
    assert len(task_betas) == 3, "Task reward betas should be of length 3"
    reward_cfg['obj_pos_beta'] = task_betas[0]
    reward_cfg['obj_rot_beta'] = task_betas[1]
    reward_cfg['obj_arti_beta'] = task_betas[2]
    reward_cfg['bc_rew_weight'] = args.bc_rew_weight
    reward_cfg['bc_beta'] = args.bc_beta
    reward_cfg['objdex_baseline'] = args.objdex_baseline    
    reward_cfg['multiply_all_rew'] = args.multiply_all_rew
    
    assert args.clip is not None, "Please provide a clip name"
    obj_name, start, end, subject, use_clip = parse_clip_string(args.clip)
    args.arctic_object = obj_name
    args.frame_start = start
    args.frame_end = end
    
    retarget_data = dict()
    if args.use_teleop and load_retarget_data:
        raise NotImplementedError("Teleoperation data loading is not implemented yet.") 
    else:
        if load_retarget_data: 
            print(f"Loading retarget data for {args.arctic_object}")
            _, retarget_data = load_genesis_retarget_data(
                obj_name=args.arctic_object,
                hand_name=args.hand,
                frame_start=start, 
                frame_end=end,
                save_name=args.retarget_name,
                use_clip=use_clip,
                subject_name=subject,
            ) 
        demo_data = get_demo_data(
            obj_name=args.arctic_object,
            hand_name=args.hand,
            frame_start=start, 
            frame_end=end,
            use_clip=use_clip,
            subject_name=subject,
            load_retarget_contact=args.use_retarget_contact,
        ) 
    
    ep_len = int(int(end) - int(start))
    if args.interp > 1:
        raise NotImplementedError("Interpolation is not implemented yet for retarget data.") 

    env_cfg = get_env_cfg(use_visualizer=args.vis, show_viewer=args.vis, show_fps=args.show_fps)
    rand_cfg = get_randomization_cfg(
        randomize=args.use_rand,
        on_friction=args.rand_friction,
        on_com=args.rand_com,
        on_mass=args.rand_mass,
        external_force=args.external_force,
        force_scale=args.force_scale,
        torque_scale=args.torque_scale
    )
    env_cfg['num_envs'] = num_envs
    env_cfg['early_reset_threshold'] = args.early_reset_threshold
    env_cfg['early_reset_aux_thres'] = dict(con=args.aux_reset_thres[0], imi=args.aux_reset_thres[1], bc=args.aux_reset_thres[2])
    env_cfg['episode_length'] = ep_len
    env_cfg['observe_tip_dist'] = args.observe_tip_dist
    env_cfg['observe_contact_force'] = True #ve_contact_force
    print(f"Setting observe_contact_force to True")
    env_cfg['use_contact_reward'] = args.contact_rew_weight > 0
    env_cfg['use_rl_games'] = args.use_rl_games
    env_cfg['rand_init_ratio'] = args.rand_init_ratio  
    env_cfg['chunk_ep_length'] = args.chunk_ep_length

    if args.record_interval > 0 or args.record_video:
        env_cfg["record_video"] = True
        # disable viewer 
        env_cfg['scene_kwargs']["use_visualizer"] = True
        env_cfg['scene_kwargs']["show_viewer"] = False
        env_cfg['scene_kwargs']["raytrace"] = args.raytrace
        if args.raytrace:
            print(f"Using white plane for raytracing")
            env_cfg['plane_urdf_path'] = 'assets/plane/plane_custom.urdf'
        env_cfg["max_video_frames"] = int(2 * ep_len)
    robot_cfgs = {
        'left': get_default_robot_cfg(name=args.hand, side='left'),
        'right': get_default_robot_cfg(name=args.hand, side='right')
    }
    for side in ['left', 'right']: 
        robot_cfgs[side]['action_mode'] = args.action_mode
        robot_cfgs[side]['hybrid_scales'] = tuple(args.hybrid_scales)
        robot_cfgs[side]['res_cap'] = args.res_cap
        robot_cfgs[side]['show_keypoints'] = args.show_kpts
        if args.hide_hand:
            robot_cfgs[side]['visualization'] = False

    obj_name = args.arctic_object
    object_cfgs = {
        obj_name: get_arctic_object_cfg(name=obj_name, convexify=args.convexify_object, texture_mesh=args.texture_object)
    } 
    if args.actuate_object:
        object_cfgs[obj_name]['actuated'] = True
        object_cfgs[obj_name]['kp'] = args.kp_init 
        object_cfgs[obj_name]['kv'] = args.kv_init
        object_cfgs[obj_name]['force_range'] = args.force_range_init
        print('Setting batch_dofs_info=True for actuated object')
        env_cfg['scene_kwargs']['batch_dofs_info'] = True
    if args.color_object is not None:
        if args.color_object == 'gray':
            object_cfgs[obj_name]['color'] = (0.5, 0.5, 0.5, 1)
    if args.no_object:
        # print('WARNING: No object in the environment. ONLY using imitation reward.')
        reward_cfg['task_rew_weight'] = 0.0 
        object_cfgs = dict()

    assert len(args.upper_ratios) == len(args.lower_ratios) == 3, "Upper and lower ratios should have length 3"
    ups = list(args.upper_ratios)
    lows = list(args.lower_ratios)
    upper_ratios = dict(kp=ups[0], kv=ups[1], fr=ups[2])
    lower_ratios = dict(kp=lows[0], kv=lows[1], fr=lows[2])
    curr_rew_thres = list(args.curr_rew_thres)
    assert len(curr_rew_thres) == 4, "Curriculum reward thresholds should have length 4"
    zero_epoch = max(int(args.max_epochs - args.num_zero_epoch), 0)

    if args.maniptrans:
        assert not args.actuate_object, "Maniptrans curriculum only works with non-actuated objects"
        print("Using maniptrans curriculum")
        maniptrans_kwargs = dict(
            wait_epochs=args.wait_epochs,
            eps_finger_range=args.eps_finger_range,
            eps_object_Prange=args.eps_object_Prange,
            eps_object_Rrange=args.eps_object_Rrange,
            friction_range=args.friction_range,
            mode=args.maniptrans_mode,
            interval=args.interval,
            schedule=args.maniptrans_schedule,
            zero_epoch=zero_epoch,
        )
        curr_cfg = get_maniptrans_cfg(maniptrans_kwargs)
    else:
        curr_kwargs = dict(
            kp_init=args.kp_init,
            kv_init=args.kv_init,
            force_range_init=args.force_range_init, 
            rew_thresholds=dict(
                task=curr_rew_thres[0],
                con=curr_rew_thres[1],
                imi=curr_rew_thres[2],
                bc=curr_rew_thres[3],
            ),
            wait_epochs=args.wait_epochs,
            decay_rew=args.decay_rew,
            schedule=args.curr_schedule,
            fixed_mode=args.fixed_mode,
            interval=args.interval,
            first_ratio=args.first_ratio,
            first_stop_iter=args.first_stop_iter,
            second_stop_iter=args.second_stop_iter,
            
            deque_freq=args.deque_freq,
            grad_threshold=args.grad_threshold,
            gain_mode=args.gain_mode,

            uniform_mode=args.uniform_mode,
            upper_ratios=upper_ratios,
            lower_ratios=lower_ratios,
            deque_len=args.deque_len,
            decay_solimp=args.decay_solimp,
            solip_multiplier=args.solip_multiplier,
            resample_every_epoch=args.resample_every_epoch,
            skip_grad=args.skip_grad,
            zero_epoch=zero_epoch, # set this to last 
            dialback_ep_len=args.dialback_ep_len,
            dialback_min_epochs=args.dialback_min_epochs,
            dialback_ratios=dict(kp=args.dialback_ratios[0], kv=args.dialback_ratios[1], fr=args.dialback_ratios[2]),
        )
        curr_cfg = get_curriculum_cfg(curr_kwargs)
    
    assert args.slowdown_rate <= 1.0, "Slowdown rate should be less than 1"
    if args.slowdown_rate < 1.0:
        if 1 % args.slowdown_rate != 0:
            print(f'WARNING: 1 / {args.slowdown_rate} is not an integer, rounding rate to {1 / (1 // args.slowdown_rate)}')
            rate = int(1 // args.slowdown_rate)
        else:
            rate = int(1 / args.slowdown_rate)
            
        env_cfg['episode_length'] = env_cfg['episode_length'] * rate
        env_cfg["max_video_frames"] = env_cfg["max_video_frames"] * rate
        env_cfg['chunk_ep_length'] = env_cfg['chunk_ep_length'] * rate
        curr_cfg['dialback_ep_len'] = curr_cfg['dialback_ep_len'] * rate
            
        demo_data['obj_pos'] = np.repeat(demo_data['obj_pos'], rate, axis=0) # 175, 3
        demo_data['obj_quat'] = np.repeat(demo_data['obj_quat'], rate, axis=0) # 175, 4
        demo_data['obj_arti'] = np.repeat(demo_data['obj_arti'], rate, axis=0) # 175,
        
        demo_data['contact_links_left'] = np.repeat(demo_data['contact_links_left'], rate, axis=0) # 175, 2, 13, 4
        demo_data['contact_links_right'] = np.repeat(demo_data['contact_links_right'], rate, axis=0) # 175, 2, 13, 4
        demo_data['contact_links_valid_left'] = np.repeat(demo_data['contact_links_valid_left'], rate, axis=0) # 175, 2, 13
        demo_data['contact_links_valid_right'] = np.repeat(demo_data['contact_links_valid_right'], rate, axis=0) # 175, 2, 13
        
        for hand in ['left', 'right']:
            import torch
            retarget_data[hand]['num_frames'] = int(retarget_data[hand]['num_frames'] // args.slowdown_rate)

            retarget_data[hand]['wrist_pose'] = torch.repeat_interleave(retarget_data[hand]['wrist_pose'], rate, dim=0) # 175, 7
            retarget_data[hand]['kpts_data']['kpt_pos'] = torch.repeat_interleave(retarget_data[hand]['kpts_data']['kpt_pos'], rate, dim=0) # 175, 18, 3
            
            for k, v in retarget_data[hand]['qpos_targets'].items():
                retarget_data[hand]['qpos_targets'][k] = torch.repeat_interleave(v, rate, dim=0) # [175] -- all are 175
            
            for k, v in retarget_data[hand]['residual_qpos'].items():
                retarget_data[hand]['residual_qpos'][k] = torch.repeat_interleave(v, rate, dim=0) # [175] -- all are 175

    env_kwargs = {
        'env_cfg': env_cfg,
        'robot_cfgs': robot_cfgs,
        'object_cfgs': object_cfgs,
        'reward_cfg': reward_cfg,
        'demo_data': demo_data,
        'retarget_data': retarget_data,
        'device': device,
        'visualize_contact': args.vis_contact,
        'rand_cfg': rand_cfg,
        'curriculum_cfg': curr_cfg,
        'group_collisions': args.group_collisions,
    }
    if args.overlay:
        env_kwargs['env_cfg']['env_spacing'] = (0.0, 0.0)  
    if args.n_envs_per_row is not None:
        env_kwargs['env_cfg']['n_envs_per_row'] = int(args.n_envs_per_row)
    return env_kwargs

def get_common_argparser():
    """ add the commonly used command line arguments for env_cfg so that they can be used in multiple scripts """
    parser = argparse.ArgumentParser()
    parser.add_argument('-B', '--num_envs', type=int, default=1)
    parser.add_argument('--vis', '-v', action='store_true')
    parser.add_argument('--overlay', '-ov', action='store_true')
    parser.add_argument('--n_envs_per_row', '-nrow', type=int, default=None)
    parser.add_argument('--texture_object', '-to', action='store_true', help='Show texture for the object and hide the actual URDF') 

    parser.add_argument('--arctic_object', '-ao', type=str, default='box')
    parser.add_argument('--hand', type=str, default='inspire_hand')
    parser.add_argument('--frame_start', '-fs', type=int, default=40)
    parser.add_argument('--frame_end', '-fe', type=int, default=200)
    parser.add_argument('--slowdown_rate', '-sd', type=float, default=1.0)
    parser.add_argument('--clip', '-cl', type=str, default="box-40-200-s01-u01")
    parser.add_argument('--show_markers', action='store_true', help='Whether to show contact markers')
    parser.add_argument('--interp', type=int, default=1, help='Interpolation multiplier for the demo data')
    parser.add_argument('--seed', type=int, default=42) 
    parser.add_argument('--action_mode', '-am', default='residual', choices=['residual', 'absolute', 'relative', 'hybrid','kinematic'])
    parser.add_argument('--show_kpts', action='store_true', help='Whether to show keypoints')
    parser.add_argument('--res_cap', action='store_true')
    parser.add_argument('--hybrid_scales', type=float, nargs='+', default=[0.04, 0.5])
    parser.add_argument('--hide_hand', action='store_true')

    parser.add_argument('--last_n_frame', type=int, default=-1)
    parser.add_argument('--early_reset_threshold', '-ert', type=float, default=0.5)
    parser.add_argument('--aux_reset_thres', type=float, nargs='+', default=[0, 0, 0])
    parser.add_argument('--record_video', '-rv', action='store_true')
    parser.add_argument('--render_camera', '-rcam', type=str, default='front')
    parser.add_argument('--raytrace', '-ray', action='store_true')
    parser.add_argument('--record_interval', '-ri', type=int, default=-1)
    parser.add_argument('--use_teleop', action='store_true') 
    parser.add_argument('--retarget_name', '-rn', type=str, default='genesis')
    parser.add_argument('--teleop_fname', type=str, default='data/scripted/tmp_box/tmp_box.npz')
    parser.add_argument('--observe_tip_dist', '-obt', action='store_true')
    parser.add_argument('--observe_contact_force', '-obf', action='store_true')
    parser.add_argument('--task_rew_betas', '-trb', type=float, nargs='+', default=[10, 1, 5]) 

    parser.add_argument('--no_object', '-no_obj', action='store_true')
    parser.add_argument('--show_fps', '-fps', action='store_true')
    parser.add_argument('--color_object', '-co', default=None, type=str, help='Color of the object to be rendered')
    parser.add_argument('--action_penalty', '-ap', type=float, default=0.0)
    parser.add_argument('--imi_rew_weight', '-imi', type=float, default=0.0)
    parser.add_argument('--imi_wrist_weight', '-imw', type=float, default=0.5) 
    parser.add_argument('--contact_rew_weight', '-con', type=float, default=0.0)
    parser.add_argument('--bc_rew_weight', '-bc', type=float, default=0.0)
    parser.add_argument('--bc_beta', '-beta', type=float, default=500.0)
    parser.add_argument('--contact_beta', '-cbeta', type=float, default=10.0)
    parser.add_argument('--skip_wrist_frame_contact', '-swfc', action='store_true')
    parser.add_argument('--nomask_zero_contact',action='store_true')
    parser.add_argument('--contact_rew_function', '-crf', type=str, default='exp', choices=['exp', 'sigmoid'])
    parser.add_argument('--use_retarget_contact', '-urc', action='store_true')
    parser.add_argument('--retarget_worldfr', '-rwf', action='store_true')

    parser.add_argument('--kp', '-kp', type=float, default=200.0)
    parser.add_argument('--kv', '-kv', type=float, default=20.0)
    parser.add_argument('--force_range', '-fr', type=float, default=100.0)
    parser.add_argument('--obj_stiffness', '-ostiff', type=float, default=0.0)
    parser.add_argument('--obj_damping', '-odamp', type=float, default=0.0)

    parser.add_argument('--vis_contact', '-vc', action='store_true')
    parser.add_argument('--use_rl_games', '-rlg', action='store_true')
    parser.add_argument('--is_eval', '-eval', action='store_true')
    parser.add_argument('--rand_init_ratio', '-randr', type=float, default=0.0)   
    parser.add_argument('--chunk_ep_length', '-chunk', type=int, default=-1)
    
    parser.add_argument('--use_rand', '-rand', action='store_true')
    parser.add_argument('--rand_friction', '-rf', action='store_true')
    parser.add_argument('--rand_com', '-rc', action='store_true')
    parser.add_argument('--rand_mass', '-rm', action='store_true')
    parser.add_argument('--external_force', '-ef', action='store_true')
    parser.add_argument('--force_scale', '-fscale', type=float, default=5.0)
    parser.add_argument('--torque_scale', '-tscale', type=float, default=0.0)
    parser.add_argument('--convexify_object', '-cvf', action='store_true')
    parser.add_argument('--actuate_object', '-act', action='store_true')
    parser.add_argument('--objdex_baseline', '-objdex', action='store_true')
    parser.add_argument('--multiply_all_rew', '-mar', action='store_true')
    parser.add_argument('--group_collisions', '-gcol', action='store_true')

    parser.add_argument('--interval', type=int, default=500) 
    parser.add_argument('--first_ratio', type=float, default=0.3)
    parser.add_argument('--first_stop_iter', type=int, default=7000)
    parser.add_argument('--second_stop_iter', type=int, default=16000)
    parser.add_argument('--kp_init', type=float, default=100.0)
    parser.add_argument('--kv_init', type=float, default=10.0)
    parser.add_argument('--force_range_init', type=float, default=50.0)
    parser.add_argument("--max_epochs", "-me", type=int, default=5000, help="Maximum number of epochs to train the agent.")     
    parser.add_argument('--num_zero_epoch', '-nze', type=int, default=1000)
    parser.add_argument('--curr_schedule', type=str, default='uniform', choices=['fixed', 'exp', 'uniform'])
    parser.add_argument('--fixed_mode', type=str, default='uniform', choices=['exp', 'lin', 'uniform'])
    parser.add_argument('--decay_rew', '-drw', action='store_true')
    parser.add_argument('--wait_epochs', type=int, default=500)
    parser.add_argument('--decrease_rew', '-dr', action='store_true')
    parser.add_argument('--curr_rew_thres', type=float, nargs='+', default=[0.5, 0, 0, 0])
    parser.add_argument('--deque_freq', type=int, default=1)
    parser.add_argument('--grad_threshold', type=float, default=0.001)
    parser.add_argument('--gain_mode', '-gm', type=str, default='all', choices=['all', 'kpkv', 'fr'])
    
    # uniform-sampling schedule 
    parser.add_argument('--uniform_mode', '-um', type=str, default='slow', choices=['fast', 'slow'])
    parser.add_argument('--upper_ratios', type=float, nargs='+', default=[0.8, 0.95, 0.95])
    parser.add_argument('--lower_ratios', type=float, nargs='+', default=[0.7, 0.9, 0.9])
    parser.add_argument('--deque_len', type=int, default=30)
    parser.add_argument('--decay_solimp', '-ds', action='store_true')
    parser.add_argument('--solip_multiplier', '-solip', type=float, default=0.95)
    parser.add_argument('--resample_every_epoch', '-resample', type=int, default=-1)
    parser.add_argument('--skip_grad', action='store_true')

    parser.add_argument('--dialback_ep_len', type=int, default=30)
    parser.add_argument('--dialback_min_epochs', type=int, default=500)
    parser.add_argument('--dialback_ratios', type=float, nargs='+', default=[0.98, 1.0, 1.0])

    # additional arguments for maniptrans
    parser.add_argument('--maniptrans', '-mpt', action='store_true')
    parser.add_argument('--friction_range', type=float, nargs='+', default=[4, 1])
    parser.add_argument('--eps_finger_range', '-efr', type=float, nargs='+', default=[0.1, 0.05])
    parser.add_argument('--eps_object_Prange', '-eopr', type=float, nargs='+', default=[0.1, 0.03])
    parser.add_argument('--eps_object_Rrange', '-eorr', type=float, nargs='+', default=[1.57, 0.52])
    parser.add_argument('--maniptrans_mode', '-mpmode', type=str, default='fixed', choices=['fixed', 'auto'])
    parser.add_argument('--maniptrans_schedule', '-mpsch', type=str, default='exp', choices=['exp', 'linear'])
    return parser
