# Dexterous Hand Asset Processing

0. Also need to run`retargeting/process_arctic.py` to get approximate contacts, this is done once for all object clips
## Inspect and Prepare URDFs
1. Clean up and inspect urdfs. Run `inspect_raw_urdf.py` to get a sense of collision geometry, default joint ranges
Note that we only did a rough pass on the URDFs to make sure the masses are approximately reasonable, but not the inertials -- we rely on the `recompute_inertia` flag [here](https://github.com/Genesis-Embodied-AI/Genesis/blob/ca97a18b6b81fb7998fb36ead69761daf23241b1/genesis/engine/solvers/rigid/collider_decomp.py#L87) from Genesis to estimate the uniform-density link inertials. 

```
python hand_processing/inspect_raw_urdf.py --urdf_path /home/mandi/chiral/assets/mimic_hand/mimic_left.urdf -v
```

2. Add 6dof wrist joints to the hand urdf. This is assuming the raw URDF file has a base link (typically the palm link, specified as `--base_link` in the script) and Run `add_wrist_dof.py`: 
```
python add_wrist_dof.py assets/xhand/xhand_left_copy.urdf xhand_left_6dof.urdf --dof_choices all --base_link left_hand_link
python add_wrist_dof.py assets/shadow_hand/shadow_hand_left_glb.urdf shadow_hand_left_6dof.urdf --dof_choices all --base_link forearm 
```
You can also use the same script but a flag to only print out joint names, this is useful for writing the retargeting config below.
```
python hand_processing/add_wrist_dof.py  /home/mandi/chiral/assets/mimic_hand/mimic_right_6dof.urdf filler.urdf --print_joint_names
```
3. Specify retargeting configs as a yaml file: this should include the fingertip mapping from robot fingers to MANO keypoints.
Do a short re-targeting run to get reference trajs, we need this for controller gain tuning 
```
python retarget_hands.py --clip box-20-100 --hand inspire_hand # add --vis flag for visualizing the retargeted traj
```
Note here that we use the fingertip indexing convention following [arctic](https://github.com/zc-alexfan/arctic/tree/master) 

4. Specify collision groups
```
python hand_processing/inspect_raw_urdf.py --gather_geoms  --num_envs 1 --urdf_path /home/mandiz/chiral/assets/schunk_hand/schunk_hand_left_6dof.urdf --base_link_name left_hand_e1  
```
This script will automatically sort through the kinematic chain and find all the collision links that belong in either the palm group or one of the fingers' groups. Note that here we actually make a strong assumption that the left and right hand kinematics are symmetric: this is due to how we implemented a new group-based collision filtering in Genesis and it only supports one set of collison group for both hands in the same scene. This is sufficient for the current hands setup but might need improvements in the future. 

5. Tuning controller gains 
See more details in section below. We default to give a 0.8 gravity compensation for all the hands. 

6. Run full-retargeting
This is our object-aware retargeting scheme that leverages the parallel sim. Running two stages so that first stage gives collision-free joints on individual steps, then second stage loads the result and roll-out in a single-threaded environment to get smoothed-out motions. 
```
OBJ=box
HAND=inspire_hand
CLIP=${OBJ}-0-600
python retargeting/parallel_retarget.py --clip $CLIP --hand ${HAND} --control_steps 2000 --save_name para --save -ow 
# then run:
python retargeting/smooth_retargets.py -lf $FNAME
# record video:
python tests/test_retarget.py -ugr -rn selfcol_smooth  --hand $HAND --clip $CLIP --skip_obj  -no_obj -ert 0 --record_video
```

and to visualize the effect
```
python hand_processing/inspect_raw_urdf.py --skip_wrist_interp -v 
```

## Controller gain tuning 

Exact steps and what to look for when you tune controller gains:

1. Tune kp first and set kv to 0: look at the speed at which the controller corrects itself to reach the target. You will see oscillation, but the amplitude and frequency of this vibration should be within a reasonable range of how fast the real robot can go, and the the joint should always center around the target.
2. Next tune kv: fix the kp and start with a _large_ range of kv: a typical first pass is [.1, 1, 100, 1000]. Now you don't want any oscillation: a good kv should make the controller steadily but still quickly go straight to the target. If it's too high the joint will be too slow or cannot fully reach the target, if it's too low the joint will still oscillate. 

Example command for single step tuning
```
python hand_processing/tune_gains.py --hand inspire -B 2 -v --joint_group finger  --kp 10 40   --kv 0 0  -fr 100.0 100.0 --clip ketchup-30-340 --freespace --step_interp 1 --step_response
```

Once the single step looks good, control the full trajectory to see if it follows the demo well
```
python hand_processing/tune_gains.py --hand inspire -B 2 -v --joint_group finger  --kp 20 20 --kv 1 1  -fr 100.0 100.0 --clip ketchup-30-340 --freespace
```
