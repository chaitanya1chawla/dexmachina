Source: https://github.com/unitreerobotics/unitree_ros/tree/master/robots/dexterous_hand_description/dex3_1

1. Removed `mujoco` compiler tag from urdf files.

2. Added a base `<root>` link and fixed joint between root and `palm_link`s

3. Manually add fingertip keypoints! For example:
```
<link name="right_thumb_tip">
  <visual>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <sphere radius="0.005"/>
    </geometry>
    <material name="">
      <color rgba="1 0 0 1"/>
    </material>
  </visual>
</link> 
<joint name="right_thumb_tip_joint" type="fixed">
  <origin xyz="0 0.045 0" rpy="0 0 0"/>
  <parent link="right_hand_thumb_2_link"/>
  <child link="right_thumb_tip"/>
</joint>
```

4. Add 6DoF floating wrist:
```
# right hand
python hand_proc/add_wrist_dof.py assets/dex3_hand/dex3_1_r.urdf dex3_right_6dof.urdf --dof_choices all --base_link right_hand_palm_link

# left hand
python hand_proc/add_wrist_dof.py assets/dex3_hand/dex3_1_l.urdf dex3_left_6dof.urdf --dof_choices all --base_link left_hand_palm_link
```

5. Full retargeting scripts:

```
OBJ=box
HAND=dex3_hand
CLIP=${OBJ}-0-600
python retargeting/parallel_retarget.py --clip $CLIP --hand ${HAND} --control_steps 2000 --save_name para --save -ow 

# record video:
python tests/test_retarget.py -ugr -rn selfcol_smooth  --hand $HAND --clip $CLIP --skip_obj  -no_obj -ert 0 --record_video
```