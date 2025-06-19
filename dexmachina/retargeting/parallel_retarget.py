import os
import sys
import time
import yaml 
import torch
import argparse
import numpy as np
import genesis as gs
from os.path import join
from pathlib import Path
from copy import deepcopy

from dexmachina.asset_utils import get_asset_path
from dexmachina.envs.demo_data import get_demo_data 
from dexmachina.envs.base_env import BaseEnv, get_env_cfg
from dexmachina.envs.robot import BaseRobot, get_default_robot_cfg 
from dexmachina.envs.object import ArticulatedObject, get_arctic_object_cfg
from dexmachina.envs.constructors import get_common_argparser, parse_clip_string  
from dexmachina.retargeting.retarget_utils import compose_retarget_config, retarget_all_steps

# need `pip install dex-retargeting`
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.kinematics_adaptor import KinematicAdaptor, MimicJointKinematicAdaptor

PROCESSED_DATADIR=get_asset_path("arctic/processed")
RETARGET_DIR=get_asset_path("retargeted")
RETARGETER_RESULTS_DIR=get_asset_path("retargeter_results")

def create_scene(
    num_envs, 
    robot_cfgs, 
    object_cfgs,
    demo_data,
    vis=False, 
    record_video=False,
    render_image=False,
    dt=1/60,
    visualize_contact=False,
    device=torch.device("cuda"),
    n_rendered_envs=None,
    group_collisions=False,
    enable_self_collision=False,
):
    scene_cfg = dict(
        sim_options=gs.options.SimOptions(
            dt=dt,
            substeps=2,
            gravity=(0, 0, -9.81),
        ), 
        vis_options=gs.options.VisOptions(
            n_rendered_envs=n_rendered_envs,
            show_world_frame=False,
            visualize_contact=False, # NOTE set to false to speed up rendering!!
            ),
        rigid_options=gs.options.RigidOptions(
            dt=dt,
            constraint_solver=gs.constraint_solver.Newton,
            enable_collision=True,
            enable_joint_limit=True,
            max_collision_pairs=100, # default was 100
            batch_dofs_info=False, 
        ), 
        use_visualizer=(vis or record_video or render_image),
        show_viewer=vis,
        show_FPS=False,
    )
    if group_collisions: 
        scene_cfg['rigid_options'].enable_self_collision = True
        scene_cfg['rigid_options'].self_collision_group_filter = True
        collision_groups = robot_cfgs['left'].get('collision_groups', dict())
        print('Setting the SAME collision grouping to both hands')
        print(collision_groups)
        scene_cfg['rigid_options'].link_group_mapping = collision_groups
    if enable_self_collision:
        print("Enabling self collision, will slow down simulation")
        scene_cfg['rigid_options'].enable_self_collision
    if args.raytrace:
        scene_cfg['renderer'] = gs.renderers.RayTracer(
            env_surface=gs.surfaces.Emission(
                emissive_texture=gs.textures.ImageTexture(
                    image_path="textures/indoor_bright.png",
                ),
            ),
            env_radius=10.0,
            env_euler=(0, 0, 180),
            lights=[
                {"pos": (0.0, 0.0, 10.0), "radius": 1.0, "color": (15.0, 15.0, 15.0)},
            ],
        )
    scene = gs.Scene(**scene_cfg)
    plane_urdf_path = 'urdf/plane/plane.urdf'
    if args.raytrace:
        plane_urdf_path = "/home/mandi/chiral/assets/plane/plane_custom.urdf" # white plane
    ground = scene.add_entity(gs.morphs.URDF(file=plane_urdf_path,fixed=True))
    cardbox_size = (0.2,0.2,0.1)
    if 'notebook' in object_cfgs:
        print("Adding a wider cardboard box for notebook")
        cardbox_size = (0.25, 0.2, 0.1)

    cardboard_box = scene.add_entity(
        gs.morphs.Box(
            pos=(0, -0.08, 0.90),
            size=cardbox_size,
            fixed=True,
        ),
        surface=gs.surfaces.Smooth(
            roughness=0.1, 
        ),
    ) 
    robots = dict()
    objects = dict()
    for k, cfg in robot_cfgs.items():
        robots[k] = BaseRobot(
            robot_cfg=cfg, 
            scene=scene,
            num_envs=num_envs,
            device=device,
            retarget_data=dict(),
            visualize_contact=visualize_contact,
            is_eval=False,
        ) 
    for k, cfg in object_cfgs.items():
        objects[k] = ArticulatedObject(
            cfg, 
            device=device,
            scene=scene,
            num_envs=num_envs,
            demo_data=demo_data,
            visualize_contact=visualize_contact,
        ) 
    camera = scene.add_camera(
        pos=(12, -11.5, 2),
        lookat=(-12, 11.5, 0),
        res=(1024, 1024), 
        fov=65, 
        GUI=False
        )

    scene.build(n_envs=num_envs, env_spacing=(1.0, 1.0), n_envs_per_row=24)
    for k, robot in robots.items():
        robot.post_scene_build_setup()
    assert len(objects) == 1, "Only one object is supported"
    obj = list(objects.values())[0]
    for k, obj in objects.items():
        obj.post_scene_build_setup()

    return scene, robots, obj, camera

def prepare_cfgs(args, hand_name, obj_name, start, end, subject_name, use_clip):
    start = int(start)
    end = int(end)
    demo_data = get_demo_data(
        obj_name=obj_name,
        hand_name=hand_name,
        frame_start=start, 
        frame_end=end,
        use_clip=use_clip,
        subject_name=subject_name,
    )
    obj_cfg = get_arctic_object_cfg(name=obj_name, convexify=False)
    print("Setting object base to fixed and collect data to True")
    obj_cfg['fixed'] = True
    obj_cfg['collect_data'] = True
    object_cfgs = {
        obj_name: obj_cfg,
    } 
    
    robot_cfgs = dict()
    print("Setting action mode to absolute and setting collect_data to True")
    for side in ['left', 'right']:
        _cfg = get_default_robot_cfg(name=hand_name, side=side)
        _cfg['action_mode'] = 'absolute'
        _cfg['collect_data'] = True 
        robot_cfgs[side] = _cfg

    num_envs = end - start
    kwargs = {
        'num_envs': num_envs,
        'robot_cfgs': robot_cfgs,
        'object_cfgs': object_cfgs,
        'demo_data': demo_data,
        'vis': args.vis,
        'record_video': args.record_video,
        'render_image': args.render_image,
        "n_rendered_envs": args.n_render,
        "group_collisions": args.group_collisions,
        "enable_self_collision": args.enable_self_collision,
    }
    return kwargs 

def prepare_retarget_cfgs(args, hand_name, obj_name, robot_cfgs, subject_name="s01", use_clip="01"):
    if 'mano' in hand_name:
        config_path = "assets/mano_hand/retarget_config.yaml"
        robot_dir = "mano-urdf"
    else:
        config_path = f"assets/{hand_name}/retarget_config.yaml"
        robot_dir = f"assets/{hand_name}"
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    with Path(config_path).open('r') as f:
        input_cfg = yaml.safe_load(f)
    print(f"Start retargeting with config {config_path}")
    retarget_type = args.retarget_type
    if 'shadow' in hand_name:
        retarget_type = 'position'
        print(f"Setting retarget type to position for shadow")
    low_pass_alpha = input_cfg.get("low_pass_alpha", 1.0)
    scaling_factor = input_cfg.get("scaling_factor", 1.0)
    ignore_mimic_joint = input_cfg.get("ignore_mimic_joint", False)
    add_dummy_free_joint = input_cfg.get("ignore_mimic_joint", False)
    print(f'NOTE: getting kpt link names directly from collison links in Genesis entity! ')
    retargeters = dict()
    for side in ['left', 'right']:
        config_dict = compose_retarget_config(
            input_cfg[side],
            retarget_type,
            low_pass_alpha,
            scaling_factor,
            ignore_mimic_joint,
            add_dummy_free_joint,
        ) 
        retarget_cfg = RetargetingConfig.from_dict(
            deepcopy(config_dict)
        )
        retargeters[side] = retarget_cfg.build() 
        target_origin_link = input_cfg[side]['target_origin_link'] 
        urdf_path = config_dict['urdf_path']

        robot_cfg = robot_cfgs[side]
        if 'inspire' not in hand_name and 'ability' not in hand_name and 'schunk' not in hand_name: # exceptions for hands with mimic joints
            assert str(robot_cfg['urdf_path']).split('/')[-1]  == urdf_path.split('/')[-1], f"Retargeting and robot should use the same URDF path, got {robot_cfg['urdf_path']} and {urdf_path}"
        # also check wrist link name is the same! 
        assert robot_cfg['wrist_link_name'] == target_origin_link,   f"Retargeting and robot should use the same wrist link name"
    
  
    input_fname = f"{subject_name}/{obj_name}_use_{use_clip}.npy"
    input_fname = join(PROCESSED_DATADIR, input_fname)
    assert os.path.exists(input_fname), f"Input file {input_fname} does not exist" 
    loaded = np.load(input_fname, allow_pickle=True).item() 
    print(f"Loaded data from {input_fname}")
    world_data = loaded["world_coord"]  

    return retargeters, world_data, input_fname

def set_init_object_states(obj, obj_pos, obj_quat, obj_arti, joint_only=False):
    """ demo_data: dict of shape (num_demo_step, k) """   
    num_demo_steps = obj_pos.shape[0]
    assert num_demo_steps == obj.num_envs, f"Number of demo steps {num_demo_steps} should match number of envs {obj.num_envs}"
    env_idxs = [i for i in range(obj.num_envs)]   
    if joint_only:
        obj.entity.set_dofs_position(position=obj_arti, dofs_idx_local=obj.dof_idxs, zero_velocity=True, envs_idx=env_idxs)
        return 
    obj.set_object_state(
        root_pos=obj_pos,
        root_quat=obj_quat,
        joint_qpos=obj_arti,
        env_idxs=env_idxs
    )

def get_obj_demo_tensors(demo_data, device=torch.device("cuda")):
    obj_pos = torch.tensor(demo_data['obj_pos'], device=device)
    obj_quat = torch.tensor(demo_data['obj_quat'], device=device)
    obj_arti = torch.tensor(demo_data['obj_arti'], device=device)[:, None] # shape (num_demo_steps, 1)
    return obj_pos, obj_quat, obj_arti

def set_hand_to_step(hands, retar_data, step, env_idxs=None): 
    for side in ['left', 'right']:
        hand = hands[side] 
        if env_idxs is None:
            env_idxs = [i for i in range(hand.num_envs)]
        num_envs = len(env_idxs)
        hand_qpos, wrist_qpos, wrist_idxs = [retar_data[side][key][step] for key in ['hand_qpos', 'wrist_qpos', 'wrist_idxs']] 
        hand_qpos = torch.tensor(hand_qpos).to(hand.init_qpos.device).unsqueeze(0).repeat(num_envs, 1)
        hand.set_joint_position(
            hand_qpos,
            env_idxs=env_idxs,
        )
    return

def prepare_robot_actions(hand, side_retar_data):
    """hand needs actions shape (num_envs, action_dim), each env_id should be a different demo-t step"""
    hand_qpos = torch.tensor(side_retar_data['hand_qpos']).to(hand.device) 
    hand_action = hand.map_joint_targets_to_actions(hand_qpos) 
    return hand_action

def gather_parallel_save_data(hands, retar_data):
    """ need to concatenate all the env id threads into one long (num_demo_step, num_joints) tensor for saving"""
    retarget_data = dict()
    for side, hand in hands.items():
        retarget_data[side] = hand.flush_episode_data()
        for k, v in retarget_data[side].items():
            if isinstance(v, torch.Tensor):
                print(f"Reshaping {k} from {v.shape} to {v[0].shape}")
                retarget_data[side][k] = v[0] # get rid of the first dim!
    return dict( 
        retarget_data=retarget_data, # this is from the robots
        retargeter_results=retar_data
        ) 

def resample_hand_qpos(hand, hand_qpos, c_steps, min_controlled_steps=100, resample_range=[10, 50], error_threshold=0.1):
    """ Resample the hand joint positions to further reduce the control error """
    control_err = hand.get_control_errors() 
    resample_mask = torch.zeros_like(control_err, dtype=torch.bool)

    resample_mask = torch.where(
        (control_err > error_threshold) & (c_steps > min_controlled_steps),
        torch.ones_like(resample_mask, dtype=torch.bool),
        resample_mask
    )

    resample_env_ids = [i for i in range(hand.num_envs) if resample_mask[i]]
    if len(resample_env_ids) == 0:
        return c_steps
    print('Resampling hand joint positions for envs:', resample_env_ids)
    rand_init_ts = torch.randint(resample_range[0], resample_range[1], (len(resample_env_ids),))
    hand_qpos_init = hand_qpos.clone()[resample_env_ids]
    hand.set_joint_position(hand_qpos_init, env_idxs=resample_env_ids)
    c_steps[resample_env_ids] = 0
    return c_steps

def get_save_fname(args, hand_name, input_fname, retarget_type, subject_name="s01"):
    save_dir = f"{RETARGET_DIR}/{hand_name}/{subject_name}"
    os.makedirs(save_dir, exist_ok=True)
    traj_name = input_fname.split("/")[-1] # e.g. box_use_01 
    save_fname = os.path.join(save_dir, traj_name.replace(".npy", f"_{retarget_type}_{args.save_name}.pt"))
    return save_fname

def get_retargeter_save_fname(args, hand_name, input_fname, retarget_type, subject_name="s01"):
    # save only retargeter results
    save_dir = f"{RETARGETER_RESULTS_DIR}/{hand_name}/{subject_name}"
    os.makedirs(save_dir, exist_ok=True)
    traj_name = input_fname.split("/")[-1] # e.g. box_use_01
    save_fname = os.path.join(save_dir, traj_name.replace(".npy", f"_{retarget_type}.npy"))
    return save_fname
 
def main(args): 
    obj_name, start, end, subject_name, use_clip = parse_clip_string(args.clip)
    hand_name = args.hand if "hand" in args.hand else f"{args.hand}_hand" 
    kwargs = prepare_cfgs(
        args, hand_name, obj_name, start, end, subject_name, use_clip
        )
    retargeters, world_data, input_fname  = prepare_retarget_cfgs(args, hand_name, obj_name, kwargs['robot_cfgs'], subject_name, use_clip)
    retarget_type = args.retarget_type if 'shadow' not in hand_name else 'position'

    assert not (args.replay_only and args.save), "Cannot save and replay at the same time" 
    save_fname = get_save_fname(args, hand_name, input_fname, retarget_type, subject_name)
    retargeter_save_fname = get_retargeter_save_fname(args, hand_name, input_fname, retarget_type, subject_name)
    if not args.overwrite and os.path.exists(save_fname) and not args.save_retargeter_only and not args.replay_only:
        print(f"File {save_fname} already exists, use --overwrite to overwrite")
        breakpoint()
    if args.replay_only:
        # load back the saved data
        if not os.path.exists(save_fname):
            # try the .pt file
            save_fname = save_fname.replace(".npy", ".pt")
            assert os.path.exists(save_fname), f"File {save_fname} does not exist"
            loaded_data = torch.load(save_fname)
        else:
            if save_fname.endswith(".pt"):
                loaded_data = torch.load(save_fname)
            else:
                loaded_data = np.load(save_fname, allow_pickle=True).item()

    gs.init(backend=gs.gpu)  
    scene, hands, obj, cam = create_scene(**kwargs) 
    num_envs = kwargs['num_envs'] 

    if args.replay_only:
        retar_data = loaded_data['retargeter_results']
    else:
        retar_data = dict()
        for side, hand in hands.items():
            dof_limits = hand.entity.get_dofs_limit(hand.actuated_dof_idxs) 
            retargeted_vals = retarget_all_steps(
                dof_limits,
                hand_init_qpos=hand.init_qpos.clone(),
                actuated_dof_names=hand.actuated_dof_names,
                actuated_dof_idxs=hand.actuated_dof_idxs,
                retargeter=retargeters[side],
                num_steps=num_envs,
                joint_pos_demo=world_data[f"joints.{side}"],
                retarget_type=retarget_type,
                frame_start=start,
            ) 
            retar_data[side] = retargeted_vals  
            # also save actuated_dof_names and actuated_dof_idxs
            retar_data[side]['actuated_dof_names'] = hand.actuated_dof_names
            retar_data[side]['actuated_dof_idxs'] = hand.actuated_dof_idxs
        # save only retar_data into npy file
        np.save(retargeter_save_fname, retar_data) 
        print(f"Saved retargeter data to {retargeter_save_fname}")
        # loaded = np.load(retargeter_save_fname, allow_pickle=True).item()
        # breakpoint()
        if args.save_retargeter_only:    
            return
    
    device = torch.device("cuda")
    hand_actions = {side: prepare_robot_actions(hands[side], retar_data[side]) for side in ['left', 'right']}
    hand_qposes = {side: torch.tensor(retar_data[side]['hand_qpos'], device=device) for side in ['left', 'right']}
    demo_data = kwargs['demo_data']
    obj_pos, obj_quat, obj_arti = get_obj_demo_tensors(demo_data, device=device)
    set_init_object_states(obj, obj_pos, obj_quat, obj_arti, joint_only=False)
    set_hand_to_step(hands, retar_data, 30)
    scene.step()

    render_frames = []

    if args.render_image:
        # cam.set_pose(pos=(-0.5, -0.5, 0.5), lookat=(0, 0, 0))
        # cv2.imwrite("retargeting/rendered_image.png", cam.render()[0])
        import cv2
        img, _, _, _ = cam.render()
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) 
        # cv2.imwrite("retargeting/rendered_image.png", img)
        render_frames.append(img)
        
    
    # for each env idx, set the initial object pose to the demo step 
    iters = 0
    controlled_steps = {side: torch.zeros(num_envs, device=device) for side in ['left', 'right']}
    for i in range(args.control_steps):
        set_init_object_states(obj, obj_pos, obj_quat, obj_arti, joint_only=True)
        for side in ['left', 'right']:
            hand = hands[side] 
            actions = hand_actions[side]
            hand.step(actions) 
        scene.step()
        for side, hand in hands.items():
            hand.update_value_buffers()
            # print(f"Side: {side} | control_errs: {hand.get_control_errors().sum() }")
            c_steps = controlled_steps[side] 
            c_steps += 1 
            hand_qpos = hand_qposes[side] 
            new_c_steps = resample_hand_qpos(
                hand, 
                hand_qpos, 
                c_steps, 
                min_controlled_steps=(20 if args.render_image else 90),
                resample_range=([10,80] if args.render_image  else [20, 60]), 
                error_threshold=(0.001 if args.render_image else 0.05),
                )
            controlled_steps[side] = new_c_steps
        if args.render_image:
            img, _, _, _ = cam.render()
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frame_name = f"retargeting/frames/rendered_{i}.png"
            if args.raytrace:
                print(f"Rendering raytrace image step {i}\n")
                cv2.imwrite(frame_name, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            render_frames.append(img)
        if i == args.control_steps - 1:
            for side, hand in hands.items():
                hand.collect_data_step(collect_all_envs=True)
    if args.render_image:
        from moviepy.editor import ImageSequenceClip
        clip = ImageSequenceClip(render_frames, fps=15)
        
        vfname = "retargeting/rendered_video.mp4"
        if args.raytrace:
            vfname = "retargeting/rendered_video_raytrace_whiteplane.mp4"
        clip.write_videofile(vfname) #, codec="libx264")
        print(f"Saved video to retargeting/{vfname}")
        breakpoint()
    print("Final control errors: ")
    for side, hand in hands.items():
        errs = hand.get_control_errors()
        print(f"Side: {side} | control_err mean: {errs.mean()} | max: {errs.max()} | min: {errs.min()}")
    
    data = gather_parallel_save_data(hands, retar_data)
    data['demo_data'] = demo_data
    if args.save and (not args.replay_only):
        torch.save(data, save_fname)
        print(f"Saved data to {save_fname}")
    return 

if __name__ == '__main__':
    parser = get_common_argparser()   
    parser.add_argument("--save_name", "-sn", type=str, default="para", help="Name of the saved file") 
    parser.add_argument("--subject_name", type=str, default="s01") 
    parser.add_argument("--set_target", "-st", action="store_true", default=False, help="Directly set target joint positions")
    parser.add_argument("--control_steps", "-cs", type=int, default=100, help="Control joint position steps")
    parser.add_argument("--save", action="store_true", default=False, help="Save the keypoints")
    parser.add_argument("--overwrite", "-ow", action="store_true", default=False, help="Overwrite existing files")
    parser.add_argument("--skip_object", "-so", action="store_true", default=False, help="Skip object")
    parser.add_argument("--remove_object", "-ro", action="store_true", default=False, help="Remove object")  
    parser.add_argument("--anchor_object", "-anc", action="store_true", default=False, help="Anchor object")
    parser.add_argument("--retarget_type", '-r', type=str, default="vector", help="either position or vector")  
    parser.add_argument("--replay_only", "-rep", action="store_true", default=False, help="Replay only")
    parser.add_argument("--catchup_freq", "-cf", type=int, default=10, help="Catch up frequency")
    parser.add_argument("--decay_force", action="store_true", default=False)
    parser.add_argument("--weak_control", action="store_true", default=False)
    parser.add_argument("--n_render", "-nr", type=int, default=100, help="Number of rendered envs")  
    parser.add_argument("--render_image", action="store_true", default=False, help="Render image for visualization") 
    parser.add_argument("--enable_self_collision", "-sc", action="store_true", default=False, help="Enable self collision")
    parser.add_argument("--save_retargeter_only", "-sro", action="store_true", default=False, help="Save only retargeter results")
    args = parser.parse_args() 
    main(args)
