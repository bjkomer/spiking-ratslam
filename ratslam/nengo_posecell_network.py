#from nengo_posecells import Posecells
from posecell_network import PosecellNetwork
import rospy
import numpy as np

PC_DIM_XY=11#21
PC_DIM_TH=36
PC_CELL_X_SIZE=0.015#1.0

class NengoPosecellNetwork(PosecellNetwork):

    def __init__(self, disable_signals=False):
        self.vtrans = 0
        self.vrot = 0
        self.stim_x = 0 # x location of injection
        self.stim_y = 0 # y location of injection
        self.stim_th = 0 # z location of injection
        self.energy = 1 # amount of energy to be injected
        self.best_x = 0
        self.best_y = 0
        self.best_th = 0

        super(NengoPosecellNetwork, self).__init__(disable_signals=disable_signals)

    def pose_cell_builder(self):

        pass

    def inject(self, act_x, act_y, act_z, energy):

        if ((act_x < PC_DIM_XY) & (act_x >= 0) & (act_y < PC_DIM_XY) & (act_y >= 0) & (act_z < PC_DIM_TH) & (act_z >= 0)):
            self.stim_x = ( act_x / PC_DIM_XY * 2 ) - 1
            self.stim_y = ( act_y / PC_DIM_XY * 2 ) - 1
            self.stim_th = ( act_z / PC_DIM_XY * 2 ) - np.pi
            self.energy = energy
        return True

    def on_odo(self, vtrans, vrot):

        #self.vtrans = vtrans * (PC_DIM_XY / PC_CELL_X_SIZE) *.01 #FIXME multiplying by magic numbers
        self.vtrans = vtrans * 2#* (PC_DIM_XY / PC_CELL_X_SIZE) *.01 #FIXME multiplying by magic numbers
        self.vrot = vrot
        self.odo_update = True
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

        energy = self.energy
        self.energy *= .1 # decay on the energy being injected

        return [self.vtrans, self.vrot, 
                self.stim_x, self.stim_y, self.stim_th,
                energy]
