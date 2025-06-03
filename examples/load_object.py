import os
import sys 
import torch   
import copy
import genesis as gs
import argparse
import numpy as np

from dexmachina.envs.object import ArticulatedObject, get_arctic_object_cfg
from dexmachina.envs.demo_data import get_demo_data


OBJ_DEFAULT_POS = (0.0597, -0.2476,  1.0354)
OBJ_DEFAULT_ROT = (-0.6413,  0.2875,  0.6467, -0.2964) 

def main(args):
    num_envs = args.num_envs

    gs.init(backend=gs.gpu)
    scene_cfg = dict(
        sim_options=gs.options.SimOptions(
            dt=1/60,
            substeps=2,
            gravity=(0, 0, -9.81) if not args.zero_gravity else (0, 0, 0), 
        ), 
        show_viewer=args.vis,
        show_FPS=False,
        use_visualizer=True, 
    )

    scene = gs.Scene(**scene_cfg)
    device = torch.device('cuda:0')

    # add support box
    cardboard_box = scene.add_entity(
        gs.morphs.Box(
            pos=(0, -0.08, 0.90),
            size=(0.2, 0.2, 0.1),
            fixed=True,
        ),
    )  
 
    obj_cfg = get_arctic_object_cfg(name=args.obj_name, texture_mesh=True)
    obj_cfg['base_init_pos'] = OBJ_DEFAULT_POS
    obj_cfg['base_init_quat'] = OBJ_DEFAULT_ROT
    if args.actuate_object:
        obj_cfg['actuated'] = True 
    demo_data = None
    if args.load_demo:
        demo_data = get_demo_data(
            obj_name=args.obj_name,
            frame_start=10,
            frame_end=510,
            hand_name='inspire_hand',
            load_retarget_contact=False,
        ) 
        obj_pos = torch.tensor(demo_data["obj_pos"], dtype=torch.float32, device=device)
        obj_quat = torch.tensor(demo_data["obj_quat"], dtype=torch.float32, device=device)
        obj_arti = torch.tensor(demo_data["obj_arti"], dtype=torch.float32, device=device)
        
    obj = ArticulatedObject(obj_cfg, device=device, scene=scene, num_envs=num_envs, demo_data=demo_data)

    scene.build(
        n_envs=num_envs, 
        env_spacing=(1.0, 1.0)
        )
    scene.reset()
    obj.post_scene_build_setup()
    step = 0
    max_step = 300
    while True:
        if args.load_demo:
            _pos, _quat, _arti = obj_pos[step][None], obj_quat[step][None], obj_arti[step][None]
        else:
            _pos = np.random.uniform(0, 0.1, size=3) + np.array(OBJ_DEFAULT_POS)
            _pos = torch.tensor(_pos, dtype=torch.float32, device=device)[None]

            _quat = np.random.uniform(-0.1, 0.1, size=4) + np.array(OBJ_DEFAULT_ROT)
            _quat = torch.tensor(_quat, dtype=torch.float32, device=device)[None]

            _arti = torch.randn(1, device=device)[None]
        obj.set_object_state(root_pos=_pos, root_quat=_quat, joint_qpos=_arti, env_idxs=[0])
        obj.step() 
        scene.step()

        obs_dict = obj.get_observations() 
        step += 1 
        if step > max_step: 
            step = 0
            scene.reset()
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_envs', type=int, default=1)
    parser.add_argument('--vis', '-v', action='store_true')
    parser.add_argument('--zero_gravity', action='store_true')
    parser.add_argument('--obj_name', '-ao', type=str, default='box')
    parser.add_argument('--actuate_object', action='store_true')
    parser.add_argument('--load_demo', action='store_true')
    args = parser.parse_args()
    main(args)
