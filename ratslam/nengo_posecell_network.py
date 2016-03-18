from nengo_posecells import Posecells
from posecell_network import PosecellNetwork
import rospy
import numpy as np

class NengoPosecellNetwork(PosecellNetwork):

    def pose_cell_builder(self):

        pass

    def inject(self, act_x, act_y, act_z, energy):

        if ((act_x < PC_DIM_XY) & (act_x >= 0) & (act_y < PC_DIM_XY) & (act_y >= 0) & (act_z < PC_DIM_TH) & (act_z >= 0)):
            self.stim_x = ( act_x / PC_DIM_XY * 2 ) - 1
            self.stim_y = ( act_y / PC_DIM_XY * 2 ) - 1
            self.stim_th = ( act_z / PC_DIM_XY * 2 ) - np.pi
        return True

    def on_odo(self, vtrans, vrot):

        self.vtrans = vtrans
        self.vrot = vrot
        """
    def excite(self):

        pass

    def inhibit(self):

        pass

    def global_inhibit(self):

        pass

    def path_integration(self, vtrans, vrot):

        pass

    def find_best(self):

        pass
        """
    def __call__(self, t, values):
        """
        Takes in best_x, best_y, and best_th as input from the Nengo network
        Outputs vtrans, vrot, and injection stimuli to the Nengo network
        """
        self.best_x = int(((values[0] + 1) /2) * PC_DIM_XY)
        self.best_y = int(((values[1] + 1) /2) * PC_DIM_XY)
        self.best_th = int(((values[2] + np.pi) /2) * PC_DIM_TH)

        return [self.vtrans, self.vrot, self.stim_x, self.stim_y, self.stim_th]
