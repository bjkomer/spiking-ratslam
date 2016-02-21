# Posecell network converted to Python

import rospy
import numpy as np

class PosecellNetwork(object):

    def __init__(self, pc_dim_xy=21, pc_dim_th=36, pc_w_e_dim=7,
                 pc_w_i_dim=5, pc_w_e_var=1, pc_w_i_var=2, pc_global_inhib=0.00002,
                 vt_active_decay=1.0, pc_vt_inject_energy=0.15, pc_cell_x_size=1.0,
                 exp_delta_pc_threshold=2.0, pc_vt_restore=0.05
                ):

        # Set up constant values
        self.pc_dim_xy = pc_din_xy
        self.pc_dim_th = pc_dim_th
        self.pc_w_e_dim = pc_w_e_dim
        self.pc_w_i_dim = pc_w_i_dim
        self.pc_w_e_var = pc_w_e_var
        self.pc_w_i_var = pc_w_i_var
        self.pc_global_inhib = pc_global_inhib
        self.vt_active_decay = vt_active_decay
        self.pc_vt_inject_energy = pc_vt_inject_energy
        self.pc_cell_x_size = pc_cell_x_size
        self.exp_delta_pc_threshold = exp_delta_pc_threshold
        self.pc_vt_restore = pc_vt_restore

        # Starting postion within the posecell network
        self.best_x = np.floor(self.pc_dim_xy / 2.0)
        self.best_y = np.floor(self.pc_dim_xy / 2.0)
        self.best_th = np.floor(self.pc_dim_th / 2.0)

        self.current_exp = 0
        self.current_vt = 0

        self.pose_cell_builder()

        self.odo_update = False
        self.vt_update = False

    def pose_cell_builder(self):

        self.pc_c_size_th = 2 * np.pi / self.pc_dim_th
        self.posecells = np.zeros((self.pc_dim_xy, self.pc_dim_xy, self.pc_dim_th))

    def inject(self, act_x, act_y, act_z, energy):

        if ((act_x < self.pc_dim_xy) & (act_x >= 0) & (act_y < self.pc_dim_xy) & (act_y >= 0) & (act_z < self.pc_dim_th) & (act_z >= 0)):
            self.posecells[act_x, act_y, act_z] += energy
        return True
