import os
import argparse
import copy
import numpy as np
import torch 
from dexmachina.envs import BaseRobot, get_default_robot_cfg 
import genesis as gs
import argparse

"""
Creates a basic scene with a robot hand and a plane, and steps through random actions.
""" 

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
        use_visualizer=True,
        show_FPS=False, 
    )
    scene = gs.Scene(**scene_cfg)
    device = torch.device('cuda:0')
    robot_cfg = get_default_robot_cfg(
        name=args.hand,
        side='left'
        )
        
    robot_cfg['action_mode'] = "absolute"
    robot = BaseRobot(
        robot_cfg, device=device, scene=scene, num_envs=num_envs
        )
    plane = scene.add_entity(
        gs.morphs.URDF(file='urdf/plane/plane.urdf', fixed=True))

    scene.build(
        n_envs=num_envs, 
        env_spacing=(2.0, 2.0)
        )
    robot.post_scene_build_setup()
    robot.reset_idx() 
    step = 0 
    max_step = 200
    iters = 0
    rand_actions = np.random.randn(max_step, robot.action_dim)
    rand_actions = torch.tensor(rand_actions, dtype=torch.float32, device=device)
    while True:
        robot.step(rand_actions[step].repeat(num_envs, 1))
        _ = robot.get_observations()
        step += 1
        scene.step()
        if step >= max_step:
            robot.reset_idx()
            # scene.reset() 
            step = 0
            iters += 1 
    return 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_envs', type=int, default=1)
    parser.add_argument('--hand', type=str, default='allegro_hand')
    parser.add_argument('--vis', '-v', action='store_true')
    parser.add_argument('--zero_gravity', action='store_true')
    args = parser.parse_args()
    main(args)

