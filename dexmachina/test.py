import genesis as gs
import torch
import numpy as np
import pygame
import sys
import json
import time
import termios
import tty
import select
from pynput import keyboard
import threading
from scipy.spatial.transform import Rotation


import casadi
import pinocchio as pin
from pinocchio import casadi as cpin
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.utils import zero

# Global variable for target pose in world coordinates
world_target_pose = None

class WeightedMovingFilter:
    def __init__(self, weights, data_size = 14):
        self._window_size = len(weights)
        self._weights = np.array(weights)
        assert np.isclose(np.sum(self._weights), 1.0), "[WeightedMovingFilter] the sum of weights list must be 1.0!"
        self._data_size = data_size
        self._filtered_data = np.zeros(self._data_size)
        self._data_queue = []

    def _apply_filter(self):
        if len(self._data_queue) < self._window_size:
            return self._data_queue[-1]

        data_array = np.array(self._data_queue)
        temp_filtered_data = np.zeros(self._data_size)
        for i in range(self._data_size):
            temp_filtered_data[i] = np.convolve(data_array[:, i], self._weights, mode='valid')[-1]
        
        return temp_filtered_data

    def add_data(self, new_data):
        assert len(new_data) == self._data_size

        if len(self._data_queue) > 0 and np.array_equal(new_data, self._data_queue[-1]):
            return  # skip duplicate data
        
        if len(self._data_queue) >= self._window_size:
            self._data_queue.pop(0)

        self._data_queue.append(new_data)
        self._filtered_data = self._apply_filter()

    @property
    def filtered_data(self):
        return self._filtered_data

def on_press(key):
    global world_target_pose
    try:
        if key.char == 'w':
            world_target_pose.translation[2] += 0.01
        elif key.char == 's':
            world_target_pose.translation[2] -= 0.01
        elif key.char == 'a':
            world_target_pose.translation[1] -= 0.01
        elif key.char == 'd':
            world_target_pose.translation[1] += 0.01
        elif key.char == 'q':
            world_target_pose.translation[0] -= 0.01
        elif key.char == 'e':
            world_target_pose.translation[0] += 0.01
        print(f"Current world target pose: {world_target_pose.translation}")
    except AttributeError:
        pass  # Special keys like shift, ctrl, etc.

def start_listener():
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

class G1_29_ArmIK_unitree:
    def __init__(self):
        np.set_printoptions(precision=5, suppress=True, linewidth=200)

        # self.robot = pin.RobotWrapper.BuildFromURDF('dexmachina/assets/robots/g1/g1_body29_hand14.urdf', 'dexmachina/assets/robots/g1/')
        self.robot = pin.RobotWrapper.BuildFromURDF('dexmachina/assets/robots/g1/g1_26dof_old_fixedbase_fixedwaist.urdf', 'dexmachina/assets/robots/g1/')

        self.mixed_jointsToLockIDs = [
                                        "left_hip_pitch_joint" ,
                                        "left_hip_roll_joint" ,
                                        "left_hip_yaw_joint" ,
                                        "left_knee_joint" ,
                                        "left_ankle_pitch_joint" ,
                                        "left_ankle_roll_joint" ,
                                        "right_hip_pitch_joint" ,
                                        "right_hip_roll_joint" ,
                                        "right_hip_yaw_joint" ,
                                        "right_knee_joint" ,
                                        "right_ankle_pitch_joint" ,
                                        "right_ankle_roll_joint" ,
                                        
                                        # "waist_yaw_joint" ,  # already fixed in urdf
                                        # "waist_roll_joint" ,  #
                                        # "waist_pitch_joint" ,  #
                                        
                                        # "left_hand_thumb_0_joint" ,
                                        # "left_hand_thumb_1_joint" ,
                                        # "left_hand_thumb_2_joint" ,
                                        # "left_hand_middle_0_joint" ,
                                        # "left_hand_middle_1_joint" ,
                                        # "left_hand_index_0_joint" ,
                                        # "left_hand_index_1_joint" ,
                                        # 
                                        # "right_hand_thumb_0_joint" ,
                                        # "right_hand_thumb_1_joint" ,
                                        # "right_hand_thumb_2_joint" ,
                                        # "right_hand_index_0_joint" ,
                                        # "right_hand_index_1_joint" ,
                                        # "right_hand_middle_0_joint",
                                        # "right_hand_middle_1_joint"
                                    ]

        self.reduced_robot = self.robot.buildReducedRobot(
            list_of_joints_to_lock=self.mixed_jointsToLockIDs,
            reference_configuration=np.array([0.0] * self.robot.model.nq),
        )

        self.reduced_robot.model.addFrame(
            pin.Frame('L_ee',
                      self.reduced_robot.model.getJointId('left_wrist_yaw_joint'),
                      pin.SE3(np.eye(3),
                              np.array([0.05,0,0]).T),
                      pin.FrameType.OP_FRAME)
        )
        
        self.reduced_robot.model.addFrame(
            pin.Frame('R_ee',
                      self.reduced_robot.model.getJointId('right_wrist_yaw_joint'),
                      pin.SE3(np.eye(3),
                              np.array([0.05,0,0]).T),
                      pin.FrameType.OP_FRAME)
        )

        self.reduced_robot.data = self.reduced_robot.model.createData()
        
        # for i in range(self.reduced_robot.model.nframes):
        #     frame = self.reduced_robot.model.frames[i]
        #     frame_id = self.reduced_robot.model.getFrameId(frame.name)
        #     print(f"Frame ID: {frame_id}, Name: {frame.name}")
        for idx, name in enumerate(self.reduced_robot.model.names):
            print(f"{idx}: {name}")
        # Creating Casadi models and data for symbolic computing
        self.cmodel = cpin.Model(self.reduced_robot.model)
        self.cdata = self.cmodel.createData()

        # Creating symbolic variables
        self.cq = casadi.SX.sym("q", self.reduced_robot.model.nq, 1) 
        self.cTf_l = casadi.SX.sym("tf_l", 4, 4)
        self.cTf_r = casadi.SX.sym("tf_r", 4, 4)
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)

        # Get the hand joint ID and define the error function
        self.L_hand_id = self.reduced_robot.model.getFrameId("L_ee")
        self.R_hand_id = self.reduced_robot.model.getFrameId("R_ee")
        
        self.R_wrist_id = self.reduced_robot.model.getFrameId("right_wrist_yaw_joint")

        self.translational_error = casadi.Function(
            "translational_error",
            [self.cq, self.cTf_l, self.cTf_r],
            [
                casadi.vertcat(
                    self.cdata.oMf[self.L_hand_id].translation - self.cTf_l[:3,3],
                    self.cdata.oMf[self.R_wrist_id].translation - self.cTf_r[:3,3]
                )
            ],
        )
        self.rotational_error = casadi.Function(
            "rotational_error",
            [self.cq, self.cTf_l, self.cTf_r],
            [
                casadi.vertcat(
                    cpin.log3(self.cdata.oMf[self.L_hand_id].rotation @ self.cTf_l[:3,:3].T),
                    cpin.log3(self.cdata.oMf[self.R_hand_id].rotation @ self.cTf_r[:3,:3].T)
                )
            ],
        )

        # Defining the optimization problem
        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.reduced_robot.model.nq)
        self.var_q_last = self.opti.parameter(self.reduced_robot.model.nq)   # for smooth
        self.param_tf_l = self.opti.parameter(4, 4)
        self.param_tf_r = self.opti.parameter(4, 4)
        self.translational_cost = casadi.sumsqr(self.translational_error(self.var_q, self.param_tf_l, self.param_tf_r))
        self.rotation_cost = casadi.sumsqr(self.rotational_error(self.var_q, self.param_tf_l, self.param_tf_r))
        self.regularization_cost = casadi.sumsqr(self.var_q)
        self.smooth_cost = casadi.sumsqr(self.var_q - self.var_q_last)

        # Setting optimization constraints and goals
        self.opti.subject_to(self.opti.bounded(
            self.reduced_robot.model.lowerPositionLimit,
            self.var_q,
            self.reduced_robot.model.upperPositionLimit)
        )
        self.opti.minimize(50 * self.translational_cost + self.rotation_cost + 0.02 * self.regularization_cost + 0.1 * self.smooth_cost)

        opts = {
            'ipopt':{
                'print_level':0,
                'max_iter':50,
                'tol':1e-6
            },
            'print_time':False,# print or not
            'calc_lam_p':False # https://github.com/casadi/casadi/wiki/FAQ:-Why-am-I-getting-%22NaN-detected%22in-my-optimization%3F
        }
        self.opti.solver("ipopt", opts)

        self.init_data = np.zeros(self.reduced_robot.model.nq)
        self.smooth_filter = WeightedMovingFilter(np.array([0.4, 0.3, 0.2, 0.1]), 14)
        self.vis = None

    def solve_ik(self, left_wrist, right_wrist, current_lr_arm_motor_q = None, current_lr_arm_motor_dq = None):
        if current_lr_arm_motor_q is not None:
            self.init_data = current_lr_arm_motor_q
        self.opti.set_initial(self.var_q, self.init_data)

        self.opti.set_value(self.param_tf_l, left_wrist)
        self.opti.set_value(self.param_tf_r, right_wrist)
        self.opti.set_value(self.var_q_last, self.init_data) # for smooth

        try:
            sol = self.opti.solve()
            # sol = self.opti.solve_limited()

            sol_q = self.opti.value(self.var_q)
            self.smooth_filter.add_data(sol_q)
            sol_q = self.smooth_filter.filtered_data

            if current_lr_arm_motor_dq is not None:
                v = current_lr_arm_motor_dq * 0.0
            else:
                v = (sol_q - self.init_data) * 0.0

            self.init_data = sol_q

            sol_tauff = pin.rnea(self.reduced_robot.model, self.reduced_robot.data, np.array(sol_q).flatten(), v, np.zeros(self.reduced_robot.model.nv))

            pin.forwardKinematics(self.reduced_robot.model, self.reduced_robot.data, np.array(sol_q).flatten())
            pin.updateFramePlacement(self.reduced_robot.model, self.reduced_robot.data, self.L_hand_id)
            pin.updateFramePlacement(self.reduced_robot.model, self.reduced_robot.data, self.R_hand_id)

            return sol_q, sol_tauff
        
        except Exception as e:
            print(f"ERROR in convergence, plotting debug info.{e}")

            sol_q = self.opti.debug.value(self.var_q)
            self.smooth_filter.add_data(sol_q)
            sol_q = self.smooth_filter.filtered_data

            if current_lr_arm_motor_dq is not None:
                v = current_lr_arm_motor_dq * 0.0
            else:
                v = (sol_q - self.init_data) * 0.0

            self.init_data = sol_q

            sol_tauff = pin.rnea(self.reduced_robot.model, self.reduced_robot.data, sol_q, v, np.zeros(self.reduced_robot.model.nv))

            print(f"sol_q:{sol_q} \nmotorstate: \n{current_lr_arm_motor_q} \nleft_pose: \n{left_wrist} \nright_pose: \n{right_wrist}")

            # return sol_q, sol_tauff
            return current_lr_arm_motor_q, np.zeros(self.reduced_robot.model.nv)
        
def setup_scene():
    gs.init(backend=gs.gpu)
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.5, 1.5, 1.8),
            camera_lookat=(0.0, -0.15, 1.0),
            camera_fov=30,
        ),
        use_visualizer=True,
        show_viewer=True,
        show_FPS=False,
    )
    
    # scene.add_entity(
        # morph=gs.morphs.Plane(
            # pos=(0, 0, 0.2),  # Position of the plane
            # normal=(0, 0, 1),  # Normal vector pointing up
        # )
    # )
    
    # Add a marker at the target pose location in world coordinates
    marker_right = scene.add_entity(
        morph=gs.morphs.Cylinder(
            radius=0.01,
            height=0.1,
            collision=False
        ),
        surface=gs.surfaces.Smooth(color=(1.0, 1.0, 0.0, 1.0)),
    )
    marker_left = scene.add_entity(
        morph=gs.morphs.Cylinder(
            radius=0.01,
            height=0.1,
            collision=False
        ),
        surface=gs.surfaces.Smooth(color=(1.0, 1.0, 0.0, 1.0)),
    )
    
    R_ee_marker = scene.add_entity(
    gs.morphs.Sphere(
        radius=0.01,
    ),
    surface=gs.surfaces.Smooth(color=(0.0, 1.0, 0.0, 1.0)),  # Green
    )
    
    robot = scene.add_entity(
        gs.morphs.MJCF(
            file="dexmachina/assets/robots/g1/g1_26dof_old_fixedbase_fixedwaist.xml",
            pos=[0.0, 0.0, 0.0],
            quat=[1.0, 0.0, 0.0, 0.0],
        )
    )
    scene.build()

    return scene, marker_right, marker_left, robot, R_ee_marker

def main():
    # Start listener in a separate thread
    listener_thread = threading.Thread(target=start_listener, daemon=True)
    listener_thread.start()

    global world_target_pose
    # sim_robot = Robot()
    arm_ik = G1_29_ArmIK_unitree()
    # Define target pose in WORLD coordinates (not reduced model coordinates)
    # This is where you want the end-effector to be in the Genesis world
    world_target_pose = pin.SE3(np.array([[ 1.00000000,  0.00000000,  0.00000000, -0.14750616],
                                         [ 0.00000000,  1.00000000,  0.00000000, 0.10866881],  # 0.10866881 + 0.3 (robot base y offset)
                                         [ 0.00000000,  0.00000000,  1.00000000, 1.21660913],  # 0.12360913 + 1.093 (robot base z offset)
                                         [ 0.00000000,  0.00000000,  0.00000000,  1.00000000]]))

    # world_target_pose.translation = np.array([-0.2998, 0.0135, 1.0252])
    # world_target_pose.rotation = quaternion_to_rotation_matrix(np.array([-0.1221, 0.3194, -0.9393, 0.0264]))
    # world_target_pose.rotation = Rotation.from_quat([-0.1221, 0.3194, -0.9393, 0.0264]).as_matrix()
    
    world_target_pose.translation = np.array([0.25, 0.25, 0.1])
    
    scene, marker_right, marker_left, robot, R_ee_marker = setup_scene()

    # Use only elbow and wrist joints
    # arm_joints = np.array([19, 20, 21, 22, 23, 24, 25])
    dof_idx_local_right_arm = np.array([13, 15, 17, 19, 21, 23, 25])
    dof_idx_local_left_arm = np.array([12, 14, 16, 18, 20, 22, 24])

    joint = robot.get_joint("right_wrist_yaw_joint")
    joint_pos = joint.get_pos()  # or joint.pos
    joint_quat = joint.get_quat()

    # with open("pose_history_recorded.json", "r") as f:
        # pose_history = np.array(json.load(f)) # 2d list
    # pose_history = np.repeat(pose_history, repeats=10, axis=0)

    for i in range(10000):
        if i > 2000:
            i=0
    
        # Control right ee from keyboard
        target_pose_right = np.array([0.25, -0.25, 0.1]) # world_target_pose.translation
        target_pose_right_quat = np.array([1.0, 0.0, 0.0, 0.0])
        target_pose_left = world_target_pose.translation
        target_pose_left_quat = np.array([1.0, 0.0, 0.0, 0.0])

        ## Read from file
        # target_pose_right = pose_history[2*i, :3]
        # target_pose_right_quat = pose_history[2*i, 3:]
        # target_pose_left = pose_history[2*i+1, :3]
        # target_pose_left_quat = pose_history[2*i+1, 3:]
        

        # Updated right wrist marker pos: tensor([[-0.2998,  0.0135,  1.0252]], device='cuda:0') quat: tensor([[-0.1221,  0.3194, -0.9393,  0.0264]], device='cuda:0')

        ###### Genesis IK ######
        ###### Uncomment to use Genesis IK ######
        # qpos_right = robot.inverse_kinematics(
            # link=robot.get_link("right_wrist_ee"),
            # pos=target_pose_right, # world_target_pose.translation,
            # quat=target_pose_right_quat, # Rotation.from_matrix(world_target_pose.rotation).as_quat(),
            # dofs_idx_local=dof_idx_local_right_arm,
        # )
        # qpos_right[:12] = 0.0
        # qpos_right[dof_idx_local_left_arm] = 0.0

        # qpos_left = robot.inverse_kinematics(   
            # link=robot.get_link("left_wrist_ee"),
            # pos=target_pose_left,
            # quat=target_pose_left_quat,
            # dofs_idx_local=dof_idx_local_left_arm,
        # )
        # qpos_left[:12] = 0.0
        # qpos_left[dof_idx_local_right_arm] = 0.0
        # qpos = qpos_right + qpos_left
        ########################
        
        
        
        ###### PinocchioIK from Unitree ######
        ###### Uncomment to use Pinocchio IK ######
        curr_qpos = robot.get_qpos()
        current_lr_arm_motor_q = torch.cat([curr_qpos[dof_idx_local_left_arm], curr_qpos[dof_idx_local_right_arm]])
        curr_qvel = robot.get_dofs_velocity()
        current_lr_arm_motor_dq = torch.cat([curr_qvel[dof_idx_local_left_arm], curr_qvel[dof_idx_local_right_arm]])
        
        R_tf_target = pin.SE3(
            pin.Quaternion(target_pose_right_quat[0], target_pose_right_quat[1], target_pose_right_quat[2], target_pose_right_quat[3]),
            np.array(target_pose_right),
        )
        L_tf_target = pin.SE3(
            pin.Quaternion(target_pose_left_quat[0], target_pose_left_quat[1], target_pose_left_quat[2], target_pose_left_quat[3]),
            np.array(target_pose_left),
        )
        # sol_q, sol_tauff = arm_ik.solve_ik(L_tf_target.homogeneous, R_tf_target.homogeneous, current_lr_arm_motor_q.cpu().numpy(), None)
        sol_q, sol_tauff = arm_ik.solve_ik(L_tf_target.homogeneous, R_tf_target.homogeneous, None, None)

        # R_ee_id = arm_ik.reduced_robot.model.getFrameId("R_ee")
        # R_ee_pose = arm_ik.reduced_robot.data.oMf[R_ee_id]  # This is a pin.SE3 object
        # R_ee_pos = R_ee_pose.translation
        # R_ee_quat = Rotation.from_matrix(R_ee_pose.rotation).as_quat()  # 3x3 rotation matrix
        # R_ee_marker.set_pos(R_ee_pos)
        # R_ee_marker.set_quat(R_ee_quat)

        sol_q_left = sol_q[:7]
        sol_q_right = sol_q[-7:]
        
        qpos = np.zeros(26)
        # qpos[dof_idx_local_left_arm] = sol_q_left
        # qpos[dof_idx_local_right_arm] = sol_q_right
        qpos[8] = sol_q_left[0]
        qpos[12] = sol_q_left[1]
        qpos[16] = sol_q_left[2]
        qpos[18] = sol_q_left[3]
        qpos[20] = sol_q_left[4]
        qpos[22] = sol_q_left[5]
        qpos[24] = sol_q_left[6]
        # qpos[25] = -1.56 # shoulder --> 9--shoulder_pitch, 13--shoulder_roll, 17--shoulder_yaw, 19--elbow, 21--wrist_roll, 23--wrist_pitch, 25--wrist_yaw
        ########################
        
        robot.set_qpos(qpos, zero_velocity=True)
        
        if i % 50 == 0:
            joint_pos = robot.get_joint("right_wrist_yaw_joint").get_pos()
            print(f"joint_pos: {joint_pos}")
        
        # Update marker position directly in world coordinates
        marker_right.set_pos(target_pose_right)
        marker_right.set_quat(target_pose_right_quat)
        marker_left.set_pos(target_pose_left)
        marker_left.set_quat(target_pose_left_quat)
        scene.step()

if __name__ == "__main__":
    main()