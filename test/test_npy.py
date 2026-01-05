import numpy as np


filepath = "output/debug_1_dual_ur5_shadow/graspdata/mujoco_Hasbro_Monopoly_Hotels_Game/grasp_2_pose_0.npy"
data = np.load(filepath, allow_pickle=True).item()

a = 1
