import pinocchio as pin
import numpy as np
import torch

class RobotArmIK:
    def __init__(self):
        self.init_robot()

    def init_robot(self):
        # Load URDF
        model_path = "dexmachina/assets/robots/g1"
        urdf_filename = "g1_26dof_old_fixedbase_fixedwaist.urdf"
        urdf_model_path = model_path + "/" + urdf_filename

        self.model = pin.buildModelFromUrdf(urdf_model_path)
        self.dof = self.model.nq
        self.data = self.model.createData()

        right_arm_joints = ["right_shoulder_yaw_joint", "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"]
        indices_to_keep = [19, 20, 21, 22, 23, 24, 25]
        indices_to_keep = [i+1 for i in indices_to_keep]  # accounting for the universal joint

        q_full = np.array(pin.neutral(self.model))

        self.reduced_model = pin.buildReducedModel(self.model, [j.id for j in self.model.joints if j.id not in indices_to_keep], q_full)
        self.reduced_data = self.reduced_model.createData()

        ee_frame_name = "right_wrist_yaw_link"  # your robot's end-effector frame
        self.ee_id = self.reduced_model.getFrameId(ee_frame_name)
        self.current_q = pin.neutral(self.reduced_model)  # or use random configuration: pin.randomConfiguration(model)
    
    def ik(self, target_pose):
        max_iter = 10000
        tolerance = 1e-4

        q = self.current_q
        q = pin.neutral(self.reduced_model)
        
        # check if target pose is pin.se3
        if not isinstance(target_pose, pin.SE3) and target_pose.shape == (1, 7):
            target_pose = np.array(torch.Tensor.cpu(target_pose)).astype(np.float64)
            target_pose = pin.XYZQUATToSE3(target_pose[0])

        for i in range(max_iter):
            # self.current_q = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

            pin.forwardKinematics(self.reduced_model, self.reduced_data, q)
            pin.updateFramePlacements(self.reduced_model, self.reduced_data)
            current_pose = self.reduced_data.oMf[self.ee_id]

            # Compute SE(3) error (logarithmic map)
            error = pin.log6(current_pose.inverse() * target_pose)
            err_vec = pin.log6(current_pose.inverse() * target_pose).vector

            if np.linalg.norm(err_vec) < tolerance:
                # print(f"Converged in {i} iterations.")
                break

            # J = pin.computeJointJacobian(reduced_model, reduced_data, q, reduced_model.frames[ee_id].parent)
            J = pin.computeFrameJacobian(self.reduced_model, self.reduced_data, q, self.ee_id, pin.ReferenceFrame.LOCAL)

            # Solve for joint velocity using damped least squares
            alpha = 0.5
            dq = alpha * np.linalg.pinv(J) @ err_vec

            q = pin.integrate(self.reduced_model, q, dq)

        else:
            print("IK did not converge.")
            print("Selecting the closest qpos")
            # exit()

        body_q = np.zeros(self.dof - 14)
        left_arm_q = np.zeros(7)
        right_arm_q = q
        arms_q=np.column_stack((left_arm_q, right_arm_q)).flatten() # interleave left and right arm
        if np.all(body_q) != 0:
            print("Warning: body_q is not all zeros")
        qpos = np.append(body_q, arms_q)

        return qpos, right_arm_q
