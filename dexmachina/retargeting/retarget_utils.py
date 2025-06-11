import os
import torch
import numpy as np  
from lxml import etree
from copy import deepcopy 
from collections import defaultdict 

from dex_retargeting.constants import RobotName, RetargetingType, HandType, get_default_config_path
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.kinematics_adaptor import KinematicAdaptor, MimicJointKinematicAdaptor

def get_link_names(urdf_path):
    assert os.path.exists(urdf_path), f"Does not exist: {urdf_path}"
    lxml_parser = etree.XMLParser(remove_comments=True, remove_blank_text=True)
    tree = etree.parse(urdf_path, parser=lxml_parser)
    robot_elem = tree.getroot()
    assert robot_elem.tag == "robot", "The first element should be a robot element."
    links = []
    for link_elem in robot_elem.findall("link"):
        if link_elem.find("collision") is not None and link_elem.attrib.get("name", None) is not None:
            links.append(link_elem.attrib["name"])
    return sorted(links)

def compose_retarget_config(
        input,
        retarget_type="vector",
        low_pass_alpha=1.0,
        scaling_factor=1.0,
        add_dummy_free_joint=False,
        ignore_mimic_joint=False,
    ):
    """ write retargeting config given input type and offline cfgs"""
    assert retarget_type in ["position", "vector"], f"Unsupported: {retarget_type}"
    urdf_path = input.get("urdf_path", None)
    assert urdf_path is not None, f"Need input urdf path"

    finger_links = input.get('target_finger_links', [])
    human_finger_idxs = input.get('human_finger_idxs', [])
    assert len(finger_links) == len(human_finger_idxs) and len(finger_links) > 0
    origin_link = input.get('target_origin_link', None)
    human_origin_idx = input.get('human_origin_idx', None)
    assert origin_link is not None and human_origin_idx is not None
    origin_link = str(origin_link)
    human_origin_idx = int(human_origin_idx)
    cfg = dict(
        type=str(retarget_type),
        urdf_path=urdf_path,
        low_pass_alpha=low_pass_alpha,
        scaling_factor=scaling_factor,
        add_dummy_free_joint=add_dummy_free_joint,
        ignore_mimic_joint=ignore_mimic_joint,
    )

    if retarget_type == "position":
        cfg['target_link_names'] = finger_links #+ [origin_link]
        cfg['target_link_human_indices'] = [int(idx) for idx in human_finger_idxs] #+ [int(human_origin_idx)]
    elif retarget_type == "vector":
        num_fingers = len(finger_links)
        cfg['target_origin_link_names'] = [origin_link] * num_fingers
        cfg['target_task_link_names'] = finger_links
        cfg['target_link_human_indices'] = [
            [int(human_origin_idx) for _ in range(num_fingers)],
            [int(idx) for idx in human_finger_idxs]
        ]
    target_joints = input.get("target_joint_names", [])
    if len(target_joints) > 0:
        cfg['target_joint_names'] = target_joints
    return cfg

def get_ref_val(joint_pos, indices):
    if len(indices.shape) > 1:
        origin_ind = indices[0, :]
        task_ind = indices[1, :]
        return joint_pos[task_ind, :] - joint_pos[origin_ind, :]
    else: # shape is (num_joints,)
        return joint_pos[indices]
 
def get_demo_obj_tensors(loaded_data, step, device):
    obj_trans = loaded_data['params']['obj_trans'][step]
    obj_rot = loaded_data['params']['obj_quat'][step]
    obj_arti = loaded_data['params']['obj_arti'][step]
    obj_trans = torch.tensor(obj_trans, dtype=torch.float32, device=device)
    obj_rot = torch.tensor(obj_rot, dtype=torch.float32, device=device)
    obj_arti = torch.tensor(obj_arti, dtype=torch.float32, device=device)
    return obj_trans, obj_rot, obj_arti


def retarget_one_hand( 
    wrist_pos, 
    retarget_type, 
    ref_value, 
    retargeter,
    hand_init_qpos,
    actuated_dof_names,
    actuated_dof_idxs
):
    
    if retarget_type == "position":
        wrist_pos *= 0.0  
    # wrist_pos *= 0.0  
    qpos_retargeted = retargeter.retarget(
        ref_value, 
        fixed_qpos=np.zeros(len(retargeter.optimizer.fixed_joint_names))
    )
    
    optimizer_joint_names = retargeter.optimizer.robot.dof_joint_names  
    idx_pin2target = retargeter.optimizer.idx_pin2target.tolist()
    if isinstance(retargeter.optimizer.adaptor, MimicJointKinematicAdaptor):
        idx_pin2target.extend(retargeter.optimizer.adaptor.idx_pin2mimic.tolist()) 
    joint_vals = dict() 
    # hand_qpos = hand.init_qpos.clone() * 0.0 # tensor of shape (num_joints,)
    hand_qpos = hand_init_qpos.clone()
    wrist_idxs = []
    wrist_qpos = []
    for idx in idx_pin2target:
        val = qpos_retargeted[idx]
        jname = optimizer_joint_names[idx]
        target_jname = jname
        if 'tx' in jname or "_x" in jname:
            val += wrist_pos[0]
        elif 'ty' in jname or "_y" in jname:
            val += wrist_pos[1]
        elif 'tz' in jname or "_z" in jname: # for mano hand
            val += wrist_pos[2] 
        joint_idx = actuated_dof_names.index(target_jname)
        hand_qpos[:, joint_idx] = val
        joint_vals[target_jname] = val
        if 'forearm' in target_jname: # wrist qpos 
            wrist_qpos.append(val)
            wrist_idxs.append(actuated_dof_idxs[joint_idx]) 
    return joint_vals, hand_qpos, wrist_qpos, wrist_idxs

def control_hand(hand, hand_qpos, wrist_qpos, wrist_idxs, set_finger=False, set_wrist=False):
    """
    NOTE the hand here is genesis entity
    """
    hand_qpos = torch.tensor(hand_qpos).to(hand.init_qpos.device) 
    if set_finger and set_wrist:
        hand.set_joint_position(hand_qpos)
    elif set_wrist:
        hand.set_joint_position(
            torch.tensor(wrist_qpos)[None].to(hand_qpos.device),
            joint_idxs=wrist_idxs,
        ) 
    else:
        # pass 
        hand.control_joint_position(hand_qpos)
    return hand_qpos

def retarget_all_steps(
    dof_limits, # (lower, upper) tuple
    hand_init_qpos,
    actuated_dof_names,
    actuated_dof_idxs,
    retargeter, 
    num_steps, 
    joint_pos_demo,
    retarget_type,
    frame_start=0,
):
    """ do retarget only once, since the retargeter optimizer seems to have some non-deterministic behavior """
    # hand_init_qpos = hand.init_qpos.clone()
    # actuated_dof_names = hand.actuated_dof_names
    # actuated_dof_idxs = hand.actuated_dof_idxs
    result_keys = ['hand_qpos', 'wrist_qpos', 'wrist_idxs']
    all_ret = {key: [] for key in result_keys}
    for step in range(num_steps): 
        demo_step = step + frame_start
        joint_pos = joint_pos_demo[demo_step]
        wrist_pos = joint_pos[0]
        indices = retargeter.optimizer.target_link_human_indices
        ref_val = get_ref_val(joint_pos, indices)
        joint_vals, hand_qpos, wrist_qpos, wrist_idxs = retarget_one_hand(
            wrist_pos=wrist_pos, 
            retarget_type=retarget_type, 
            ref_value=ref_val, 
            retargeter=retargeter,  
            hand_init_qpos=hand_init_qpos,
            actuated_dof_names=actuated_dof_names,
            actuated_dof_idxs=actuated_dof_idxs,
        )
        # clamp 
        hand_qpos = torch.clamp(hand_qpos, dof_limits[0].to(hand_qpos.device), dof_limits[1].to(hand_qpos.device))
        all_ret['hand_qpos'].append(hand_qpos[0].cpu().numpy()) # only save the first env!
        all_ret['wrist_qpos'].append(wrist_qpos)
        all_ret['wrist_idxs'].append(wrist_idxs)
    all_ret = {key: np.stack(val) for key, val in all_ret.items()}
    return all_ret

