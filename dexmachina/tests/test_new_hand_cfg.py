import os 
import argparse
from dexmachina.envs.robot import BaseRobot, get_default_robot_cfg


def main(args):
    print(f"Looking up config for hand: {args.hand}")
    robot_cfgs = {
        'left': get_default_robot_cfg(name=args.hand, side='left'),
        'right': get_default_robot_cfg(name=args.hand, side='right')
    }
    print(f"Scuess! Found robot configurations:")
    for side, cfg in robot_cfgs.items():
        print(f"{side.capitalize()} Hand:")
        for key, value in cfg.items():
            print(f"  {key}: {value}")
    exit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test new hand configurations.")
    parser.add_argument("--hand", type=str, required=True, help="Name of the robot configuration to test.")
    args = parser.parse_args()
    main(args)
