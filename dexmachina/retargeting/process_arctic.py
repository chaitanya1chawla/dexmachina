import numpy as np 
import pickle
import os
import torch
from mano_urdf.hand_model import HandModel45
from mano_urdf.hand_body import HandBody 
import argparse
from sklearn.neighbors import KDTree 
from copy import deepcopy
# add argparse arguments 
parser = argparse.ArgumentParser(description="Process ARCTIC data to get contact information")
# add arguments
parser.add_argument("--processed_fname", "-p", type=str, default="/home/mandi/arctic/outputs/processed_verts/seqs/s01/box_use_01.npy", help="Path to the processed ARCTIC data")
parser.add_argument("--skip_visualization", "-sv", action="store_true", help="Skip visualization")
parser.add_argument("--contact_threshold", "-ct", type=float, default=0.01, help="Contact threshold")
parser.add_argument("--max_contact_per_step", "-max", type=int, default=50, help="Max contact points per step")
parser.add_argument("--farthest_sample", "-fs", action="store_true", help="Sample farthest contact points")
parser.add_argument("--save", "-s", action="store_true", help="Save the processed data")
parser.add_argument("--overwrite", "-ow", action="store_true", help="Overwrite the processed data")
parser.add_argument("--debug", "-db", action="store_true", help="Visualize the processed data")
# parse the arguments
args_cli = parser.parse_args()

def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Source: https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
    Convert rotations given as axis/angle to quaternions.
    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.
    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions

def approximate_contact(verts1, verts2, dist_min=0, dist_max=100, threshold=0.01):
    """
    Find the closest points between two sets of mesh vertices as contact points.
    return these vertices on both verts1 and verts2, also return the idxs 
    """
    tree2 = KDTree(verts2)
    dist, idx = tree2.query(verts1)
    # dist is shape (n, 1)
    dist = np.clip(dist, dist_min, dist_max)
    contact_mask = dist < threshold # shape (N1, 1)
    # if all the distances are big, no contact is made
    if not np.any(contact_mask):
        return np.array([]), np.array([]), np.array([]), np.array([])

    contact_on_verts2 = verts2[idx[contact_mask]] 
    contact_on_verts1 = verts1[contact_mask[:, 0]]
    idxs1 = idx[contact_mask]
    idxs2 = np.arange(verts2.shape[0])[contact_mask]
    return contact_on_verts1, contact_on_verts2, idxs1, idxs2

def approximate_contact_with_id(obj_verts, obj_part_ids, hand_verts, threshold=0.01, dist_min=0, dist_max=100):
    """
    Because object vertices each has a part id (1 or 2), we want to returrn contact points with their part id 
    return contact points both on obj_verts, and use the part_id for the closest point for hand_verts
    returns (num_contact, 4), (num_contact, 4)
    """
    tree_hand = KDTree(hand_verts)
    dist, idx = tree_hand.query(obj_verts) # dist shape (N, 1), idx shape (N, 1)
    dist = np.clip(dist, dist_min, dist_max)
    contact_mask = dist < threshold
    if not np.any(contact_mask):
        return np.array([]), np.array([])
    contact_on_hand = hand_verts[idx[contact_mask]]
    # part_id_on_hand = obj_part_ids[idx[contact_mask]]
    contact_on_obj = obj_verts[contact_mask[:, 0]]
    part_id_on_obj = obj_part_ids[contact_mask[:, 0]]

    return np.concatenate([contact_on_obj, part_id_on_obj[:, None]], axis=-1), np.concatenate([contact_on_hand, part_id_on_obj[:, None]], axis=-1)
 
def farthest_sample_contact(contact_pts, num_sample): # shape (N, 3)
    """ 
    Sample every point that's farthest from the rest 
    If feature dim=4, i.e. last dim is part ID, then measure distance without the part ID
    """
    sampled = []  
    assert num_sample <= contact_pts.shape[0], f"num_sample {num_sample} must be less than contact_pts.shape[0] {contact_pts.shape[0]}"
    while len(sampled) < num_sample:
        if len(sampled) == 0:
            sampled.append(contact_pts[0]) 
        else:
            dists = np.linalg.norm(
                contact_pts[:, :3] - np.array(sampled)[:, :3][:, None],
                axis=-1)
            farthest = np.argmax(np.min(dists, axis=0))
            sampled.append(contact_pts[farthest]) 
    return np.array(sampled) 
 
MANO_HAND_LINKS=[
    ("palm", [0, 13, 1, 4, 10, 7]),
    ("thumb1", [13, 14]),
    ("thumb2", [14, 15]),
    ("thumb3", [15, 16]),
    ("index1", [1, 2]),
    ("index2", [2, 3]),
    ("index3", [3, 17]),
    ("middle1", [4, 5]),
    ("middle2", [5, 6]),
    ("middle3", [6, 18]),
    ("ring1", [10, 11]),
    ("ring2", [11, 12]),
    ("ring3", [12, 19]),
    ("pinky1", [7, 8]),
    ("pinky2", [8, 9]),
    ("pinky3", [9, 20]),    
]

def find_closest_link(contact_points, joint_points):
    """
    Given a list of (N, 3 or 4) contact points, find the distance between each point and each link by finding the avg distance to the link's joint points
    Then, return shape (21, 4): the averaged distances for each link, contact with part id IF there's valid contact, else that link is all 0s
    """
    all_avg_dists = []
    for idx, (link_name, joint_idxs) in enumerate(MANO_HAND_LINKS):
        dists = []
        for joint_idx in joint_idxs:
            joint_point = joint_points[joint_idx]
            dist = np.linalg.norm(contact_points[:, :3] - joint_point, axis=-1)
            dists.append(dist)
        avg_dist = np.mean(np.stack(dists, axis=0), axis=0)
        all_avg_dists.append(avg_dist) # shape (N,)
    all_avg_dists = np.stack(all_avg_dists, axis=0) # shape (num_links, N)
    # now for each point, find the closest link and its link index
    closest_link_idx = np.argmin(all_avg_dists, axis=0) # shape (N,)
    closest_link_dist = np.min(all_avg_dists, axis=0) # shape (N,)
    avg_contact_positions = [np.zeros(4) for _ in range(len(MANO_HAND_LINKS))]
    for idx, (link_name, joint_idxs) in enumerate(MANO_HAND_LINKS):
        if not np.any(closest_link_idx == idx):
            continue
        mask = closest_link_idx == idx
        part_ids = contact_points[mask, 3]
        # take the majority part id
        voted_part_id = np.argmax(np.bincount(part_ids.astype(int))) 
        weighted_avg_position = np.average(contact_points[mask][:, :3], axis=0, weights=1/closest_link_dist[mask])
        avg_contact_positions[idx] = np.concatenate([weighted_avg_position, [voted_part_id]])
    return avg_contact_positions 

SAVE_DIR = "assets/arctic/processed"
path_mean_r = "assets/mano-urdf/right_pose_mean.txt"
path_mean_l = "assets/mano-urdf/left_pose_mean.txt"
pose_mean_r = np.loadtxt(path_mean_r, dtype=np.float32)
pose_mean_l = np.loadtxt(path_mean_l, dtype=np.float32)

HOME_DIR="/home/mandiz"
if not os.path.exists(HOME_DIR):
    HOME_DIR="/home/mandi"
    assert os.path.exists(HOME_DIR), f"{HOME_DIR} does not exist"
fname = args_cli.processed_fname
assert os.path.exists(fname), f"{fname} does not exist"
arctic_data = np.load(fname, allow_pickle=True).item() # array of shape (T, 7)
world_data = arctic_data["world_coord"]
params = arctic_data["params"]
verts_right = world_data["verts.right"]
verts_left = world_data["verts.left"] # shape (T=605, 778, 3)
verts_obj = world_data["verts.object"] # shape (T=605, 3580, 3)

part_ids = world_data["parts_ids"] # shape (T, 3580)
num_steps = params["obj_trans"].shape[0]

if args_cli.debug:
    assert not args_cli.save, "Cannot save and debug at the same time"
    print(f"Only processing {num_steps} steps")
    num_steps = 200
# approx contact points
contact_thres = args_cli.contact_threshold
max_contact_per_step = args_cli.max_contact_per_step
contacts_left = np.zeros((num_steps, max_contact_per_step, 4))
contact_part_ids_left = np.zeros((num_steps, max_contact_per_step), dtype=int)

contacts_left_hand = np.zeros((num_steps, len(MANO_HAND_LINKS), 4)) # contact vertices on hand mesh
valid_contacts_left = np.zeros((num_steps, max_contact_per_step), dtype=bool)
contacts_right = np.zeros((num_steps, max_contact_per_step, 4)) 
contacts_right_hand = np.zeros((num_steps, len(MANO_HAND_LINKS), 4)) # contact vertices on hand mesh
valid_contacts_right = np.zeros((num_steps, max_contact_per_step), dtype=bool) 

for step in range(num_steps):

    for side in ["left", "right"]:
        verts = verts_left[step] if side == "left" else verts_right[step]
        # contacts_hand, contacts, idxs_hand, idxs_obj = approximate_contact(verts, verts_obj[step], threshold=contact_thres)
        contacts, contacts_hand = approximate_contact_with_id(
            verts_obj[step], part_ids[step], verts, threshold=contact_thres)
        if contacts.shape[0] == 0:
            continue
        num_contacts = min(max_contact_per_step, contacts.shape[0])
        if args_cli.farthest_sample:
            contacts_hand = farthest_sample_contact(contacts_hand, num_contacts)
            contacts  = farthest_sample_contact(contacts, num_contacts)
        else:
            contacts = contacts[:num_contacts]
            idxs_obj = idxs_obj[:num_contacts]
            contacts_hand = contacts_hand[:num_contacts] 

        fingertips = world_data[f"joints.{side}"][step]
        contact_links = find_closest_link(contacts_hand, fingertips)
        
        if side == "left":
            contacts_left[step, :num_contacts] = contacts 
            valid_contacts_left[step, :num_contacts] = True
            contacts_left_hand[step] = contact_links
        else:
            contacts_right[step, :num_contacts] = contacts 
            valid_contacts_right[step, :num_contacts] = True
            contacts_right_hand[step] = contact_links
    if step % 100 == 0:
        print(f"Processed {step} steps")

tosave = {
    "world_coord": {
        "joints.left": world_data["joints.left"],
        "joints.right": world_data["joints.right"],
        "contacts.left": contacts_left,
        "valid_contacts.left": valid_contacts_left,
        "contacts.right": contacts_right,
        "valid_contacts.right": valid_contacts_right,
        "contact_threshold": np.array(contact_thres),
        "contact_links_left": contacts_left_hand,
        "contact_links_right": contacts_right_hand,
    },
    "params": {
        "obj_trans": params["obj_trans"] / 1000.0, # convert to meters
        "obj_rot": params["obj_rot"].copy(),
        "obj_arti": params["obj_arti"].copy(),
        "trans_l": params["trans_l"].copy(),
        "trans_r": params["trans_r"].copy(),
        "rot_l": params["rot_l"].copy(),
        "rot_r": params["rot_r"].copy(),
        "shape_l": params["shape_l"].copy(),
        "shape_r": params["shape_r"].copy(),
    }, 
    }

MANO_MODELS_DIR=f"{HOME_DIR}/mano_v1_2/models"
left_hand_model = HandModel45(left_hand=True, models_dir=MANO_MODELS_DIR)
left_mano_hand = HandBody(
    left_hand_model, 
    shape_betas=arctic_data['params']['shape_l'][0], #left_mano["shape"],
    urdf_dir="examples/urdf/left",
    urdf_object_format="stl"
    )

right_hand_model = HandModel45(left_hand=False, models_dir=MANO_MODELS_DIR)
right_mano_hand = HandBody(
    right_hand_model, 
    shape_betas=arctic_data['params']['shape_r'][0], #right_mano["shape"],
    urdf_dir="examples/urdf/right",
    urdf_object_format="stl"
    )

configs_left, configs_right = [], []
obj_quats = []
left_quats, right_quats = [], []
for step in range(num_steps):
    # save the joint configurations
    obj_rot = params["obj_rot"][step]
    # obj_pos = params["obj_pos"][step]
    obj_quat = axis_angle_to_quaternion(torch.tensor(obj_rot)).numpy()
    obj_quats.append(obj_quat)
     
    trans_l = params["trans_l"][step]
    left_rot = params["rot_l"][step]
    left_quats.append(
        axis_angle_to_quaternion(torch.tensor(left_rot)).numpy()
    )
    left_joints = params["pose_l"][step] + pose_mean_l
    mano_pose = np.concatenate([left_rot, left_joints])
    left_configs = left_mano_hand.get_joint_config_from_mano(trans_l, mano_pose)
    configs_left.append(left_configs)

    trans_r = params["trans_r"][step]
    right_rot = params["rot_r"][step]
    right_quats.append(
        axis_angle_to_quaternion(torch.tensor(right_rot)).numpy()
    )
    right_joints = params["pose_r"][step] + pose_mean_r
    mano_pose = np.concatenate([right_rot, right_joints])
    right_configs = right_mano_hand.get_joint_config_from_mano(trans_r, mano_pose)
    configs_right.append(right_configs)
print(f"Processed {num_steps} steps")
# concatenate the joint configurations
tosave_config_left = dict()
for k in configs_left[0].keys():
    tosave_config_left[k] = np.stack([config[k] for config in configs_left], axis=0)
tosave["configs_left"] = tosave_config_left

tosave_config_right = dict()
for k in configs_right[0].keys():
    tosave_config_right[k] = np.stack([config[k] for config in configs_right], axis=0)
tosave["configs_right"] = tosave_config_right

tosave["params"]["obj_quat"] = np.stack(obj_quats, axis=0)
tosave["params"]["left_quat"] = np.stack(left_quats, axis=0)
tosave["params"]["right_quat"] = np.stack(right_quats, axis=0)
# save the data
filename = fname.split("/")[-1]
subject_name = fname.split("/")[-2]
save_fname = f"{SAVE_DIR}/{subject_name}/{filename}"  
if os.path.exists(save_fname) and args_cli.save and not args_cli.overwrite:
    print("WARNING -", f"{save_fname} already exists, overwrite?")
    breakpoint()
if args_cli.save:
    np.save(save_fname, tosave)
# try loading
loaded = np.load(save_fname, allow_pickle=True).item() # loaded to dict 
print(f"Saved to {save_fname}")
if args_cli.skip_visualization:
    print("Skipping visualization")
exit(0)
