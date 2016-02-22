# Posecell network converted to Python

import rospy
import numpy as np
import itertools
from sensor_msgs.msg import Image, CompressedImage
from nav_msgs.msg import Odometry
from ratslam_ros.msg import ViewTemplate, TopologicalAction
import cv2

# TopologicalAction values
NO_ACTION=0
CREATE_NODE=1
CREATE_EDGE=2
SET_NODE=3

# Global Contants #TODO: fix the code to use these constants instead of attributes
PC_DIM_XY=21
PC_DIM_TH=36
PC_W_E_DIM=7
PC_W_I_DIM=5
PC_W_E_VAR=1
PC_W_I_VAR=2
PC_GLOBAL_INHIB=0.00002
VT_ACTIVE_DECAY=1.0
PC_VT_INJECT_ENERGY=0.15
PC_CELL_X_SIZE=1.0
EXP_DELTA_PC_THRESHOLD=2.0
PC_VT_RESTORE=0.05

# Contants taken from Renato de Pontes Pereira
def create_pc_weights(dim, var):
    dim_center = int(np.floor(dim/2.))
    
    weight = np.zeros([dim, dim, dim])
    for x, y, z in itertools.product(xrange(dim), xrange(dim), xrange(dim)):
        dx = -(x-dim_center)**2
        dy = -(y-dim_center)**2
        dz = -(z-dim_center)**2
        weight[x, y, z] = 1.0/(var*np.sqrt(2*np.pi))*np.exp((dx+dy+dz)/(2.*var**2))

    weight = weight/np.sum(weight)
    return weight
PC_W_EXCITE             = create_pc_weights(PC_W_E_DIM, PC_W_E_VAR)
PC_W_INHIB              = create_pc_weights(PC_W_I_DIM, PC_W_I_VAR)
PC_W_E_DIM_HALF         = int(np.floor(PC_W_E_DIM/2.))
PC_W_I_DIM_HALF         = int(np.floor(PC_W_I_DIM/2.))
PC_C_SIZE_TH            = (2.*np.pi)/PC_DIM_TH
PC_E_XY_WRAP            = range(PC_DIM_XY-PC_W_E_DIM_HALF, PC_DIM_XY) + range(PC_DIM_XY) + range(PC_W_E_DIM_HALF)
PC_E_TH_WRAP            = range(PC_DIM_TH-PC_W_E_DIM_HALF, PC_DIM_TH) + range(PC_DIM_TH) + range(PC_W_E_DIM_HALF)
PC_I_XY_WRAP            = range(PC_DIM_XY-PC_W_I_DIM_HALF, PC_DIM_XY) + range(PC_DIM_XY) + range(PC_W_I_DIM_HALF)
PC_I_TH_WRAP            = range(PC_DIM_TH-PC_W_I_DIM_HALF, PC_DIM_TH) + range(PC_DIM_TH) + range(PC_W_I_DIM_HALF)            
PC_XY_SUM_SIN_LOOKUP    = np.sin(np.multiply(range(1, PC_DIM_XY+1), (2*np.pi)/PC_DIM_XY))
PC_XY_SUM_COS_LOOKUP    = np.cos(np.multiply(range(1, PC_DIM_XY+1), (2*np.pi)/PC_DIM_XY))
PC_TH_SUM_SIN_LOOKUP    = np.sin(np.multiply(range(1, PC_DIM_TH+1), (2*np.pi)/PC_DIM_TH))
PC_TH_SUM_COS_LOOKUP    = np.cos(np.multiply(range(1, PC_DIM_TH+1), (2*np.pi)/PC_DIM_TH))
PC_CELLS_TO_AVG         = 3
PC_AVG_XY_WRAP          = range(PC_DIM_XY-PC_CELLS_TO_AVG, PC_DIM_XY) + range(PC_DIM_XY) + range(PC_CELLS_TO_AVG)
PC_AVG_TH_WRAP          = range(PC_DIM_TH-PC_CELLS_TO_AVG, PC_DIM_TH) + range(PC_DIM_TH) + range(PC_CELLS_TO_AVG)


class PosecellNetwork(object):

    def __init__(self, root='/irat_red',
                 pc_dim_xy=21, pc_dim_th=36, pc_w_e_dim=7,
                 pc_w_i_dim=5, pc_w_e_var=1, pc_w_i_var=2, pc_global_inhib=0.00002,
                 vt_active_decay=1.0, pc_vt_inject_energy=0.15, pc_cell_x_size=1.0,
                 exp_delta_pc_threshold=2.0, pc_vt_restore=0.05
                ):

        self.root = root
        self.pc_output = TopologicalAction()
        self.prev_time = rospy.Time() # defaults time to 0
        rospy.init_node('posecells', anonymous=True)

        self.pub_pc = rospy.Publisher( self.root + '/PoseCell/TopologicalAction',
                                       TopologicalAction
                                     )
        self.sub_odometry = rospy.Subscriber( self.root + '/odom',
                                         Odometry,
                                         self.odo_callback
                                       )
        self.sub_template = rospy.Subscriber( self.root + '/LocalView/Template',
                                        ViewTemplate,
                                        self.template_callback
                                      )

        # Set up constant values
        self.pc_dim_xy = pc_dim_xy
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

        # Initialize variables
        self.current_exp = 0
        self.current_vt = 0
        self.prev_vt = 0
        self.visual_templates = []
        self.experiences = []

        self.vt_delta_pc_th = 0

        self.pose_cell_builder()

        self.odo_update = False
        self.vt_update = False

    def odo_callback(self, odo):

        #NOTE: C version has a check to see if the time is greater than 0 here
        # Calculate time difference in seconds
        time_diff = (odo.header.stamp - self.prev_time).secs

        vtrans = odo.twist.twist.linear.x
        vrot = odo.twist.twist.angular.z

        self.pc_output.src_id = self.get_current_exp_id()

        self.on_odo(vtrans*time_diff, vrot*time_diff)
        self.pc_output.action = self.get_action()
        
        # Only publish a message if there is an action
        if self.pc_output.action != NO_ACTION:

            self.pc_output.header.stamp = rospy.Time.now()
            self.pc_output.header.seq += 1
            self.pc_output.dest_id = self.get_current_exp_id()
            self.pc_output.relative_rad = self.get_relative_rad()

            self.pub_pc.publish(self.pc_output)

        self.prev_time = odo.header.stamp

    def on_odo(self, vtrans, vrot):

        self.excite()
        self.inhibit()
        self.global_inhibit()
        self.normalize()
        self.path_integration(vtrans, vrot)
        self.find_best()
        self.odo_update = True

    def template_callback(self, vt):

        self.on_view_template(vt.current_id, vt.relative_rad)

    def on_view_template(self, vt, vt_rad):

        if vt >= len(self.visual_templates):
            # must be a new template
            self.create_view_template()
        else:
            # the template must exist
            pcvt = self.visual_templates[vt]

            # this prevents energy injected in recently created vt's
            if vt < len(self.visual_templates) - 10:
                if vt == self.current_vt:
                    pcvt['decay'] += self.vt_active_decay

                # magic line that michael knows about
                energy = self.pc_vt_inject_energy * 1.0 / 30.0 * (30.0 - np.exp(1.2 * pcvt['decay']))

                if energy > 0:
                    self.vt_delta_pc_th = vt_rad / (2.0*np.pi) * self.pc_dim_th
                    pc_th_corrected = pcvt['pc_th'] + vt_rad / (2.0*np.pi) * self.pc_dim_th
                    if pc_th_corrected < 0:
                        pc_th_corrected += self.pc_dim_th
                    if pc_th_corrected >= self.pc_dim_th:
                        pc_th_corrected -= self.pc_dim_th

                    # Inject energy into the posecell network
                    self.inject(pcvt['pc_x'], pcvt['pc_y'], pc_th_corrected, energy)

        for visual_template in self.visual_templates:
            visual_template['decay'] -= self.pc_vt_restore
            if visual_template['decay'] < self.vt_active_decay:
                visual_template['decay'] = self.vt_active_decay

        self.prev_vt = self.current_vt
        self.current_vt = vt

        self.vt_update = True

    def create_view_template(self):

        pcvt = {'pc_x': self.best_x,
                'pc_y': self.best_y,
                'pc_th': self.best_th,
                'decay': self.vt_active_decay,
                'exps':[]}

        self.visual_templates.append(pcvt)

    def create_experience(self):

        pcvt = self.visual_templates[self.current_vt]
        exp = {'x_pc': self.best_x,
               'y_pc': self.best_y,
               'th_pc': self.best_th,
               'vt_id': self.current_vt}
        self.experiences.append(exp)
        # Set the current experience to be the index of the latest experience
        self.current_exp = len(self.experiences)-1
        pcvt['exps'].append(self.current_exp) #TODO: make sure this is doing the referencing properly

    def pose_cell_builder(self):

        self.pc_c_size_th = 2 * np.pi / self.pc_dim_th
        self.posecells = np.zeros((self.pc_dim_xy, self.pc_dim_xy, self.pc_dim_th))

        self.posecells[np.floor(PC_DIM_XY/2), 
                       np.floor(PC_DIM_XY/2),
                       np.floor(PC_DIM_TH/2)] = 1
        #TODO: a lot more stuff might be missing in here

    def get_current_exp_id(self):

        return self.current_exp

    def get_relative_rad(self):

        return self.vt_delta_pc_th * 2.0 * np.pi / self.pc_dim_th

    def inject(self, act_x, act_y, act_z, energy):

        if ((act_x < self.pc_dim_xy) & (act_x >= 0) & (act_y < self.pc_dim_xy) & (act_y >= 0) & (act_z < self.pc_dim_th) & (act_z >= 0)):
            self.posecells[act_x, act_y, act_z] += energy
        return True
    
    def compute_activity_matrix(self, xywrap, thwrap, wdim, pcw): 
        """Compute the activation of pose cells. Taken from Renato de Pontes Pereira"""
        
        # The goal is to return an update matrix that can be added/subtracted
        # from the posecell matrix
        pca_new = np.zeros([PC_DIM_XY, PC_DIM_XY, PC_DIM_TH])
        
        # for nonzero posecell values  
        indices = np.nonzero(self.posecells)

        for i,j,k in itertools.izip(*indices):
            pca_new[np.ix_(xywrap[i:i+wdim], 
                           xywrap[j:j+wdim],
                           thwrap[k:k+wdim])] += self.posecells[i,j,k]*pcw
         
        return pca_new

    def excite(self):

        self.posecells = self.compute_activity_matrix(PC_E_XY_WRAP, 
                                                      PC_E_TH_WRAP, 
                                                      PC_W_E_DIM, 
                                                      PC_W_EXCITE)

    def inhibit(self):

        self.posecells = self.posecells - self.compute_activity_matrix(PC_I_XY_WRAP, 
                                                                       PC_I_TH_WRAP, 
                                                                       PC_W_I_DIM, 
                                                                       PC_W_INHIB)

    def global_inhibit(self):

        self.posecells[self.posecells < PC_GLOBAL_INHIB] = 0
        self.posecells[self.posecells >= PC_GLOBAL_INHIB] -= PC_GLOBAL_INHIB

    def normalize(self):

        total = np.sum(self.posecells)
        self.posecells /= total

    def path_integration(self, vtrans, vrot):

        # vtrans affects xy direction
        # shift in each th given by the th
        for dir_pc in xrange(PC_DIM_TH): 
            direction = np.float64(dir_pc-1) * PC_C_SIZE_TH
            # N,E,S,W are straightforward
            if (direction == 0):
                self.posecells[:,:,dir_pc] = \
                    self.posecells[:,:,dir_pc] * (1.0 - vtrans) + \
                    np.roll(self.posecells[:,:,dir_pc], 1, 1)*vtrans

            elif direction == np.pi/2:
                self.posecells[:,:,dir_pc] = \
                    self.posecells[:,:,dir_pc]*(1.0 - vtrans) + \
                    np.roll(self.posecells[:,:,dir_pc], 1, 0)*vtrans

            elif direction == np.pi:
                self.posecells[:,:,dir_pc] = \
                    self.posecells[:,:,dir_pc]*(1.0 - vtrans) + \
                    np.roll(self.posecells[:,:,dir_pc], -1, 1)*vtrans

            elif direction == 3*np.pi/2:
                self.posecells[:,:,dir_pc] = \
                    self.posecells[:,:,dir_pc]*(1.0 - vtrans) + \
                    np.roll(self.posecells[:,:,dir_pc], -1, 0)*vtrans

            else:
                pca90 = np.rot90(self.posecells[:,:,dir_pc], 
                              int(np.floor(direction *2/np.pi)))
                dir90 = direction - int(np.floor(direction*2/np.pi)) * np.pi/2


                # extend the Posecells one unit in each direction (max supported at the moment)
                # work out the weight contribution to the NE cell from the SW, NW, SE cells 
                # given vtrans and the direction
                # weight_sw = v * cos(th) * v * sin(th)
                # weight_se = (1 - v * cos(th)) * v * sin(th)
                # weight_nw = (1 - v * sin(th)) * v * sin(th)
                # weight_ne = 1 - weight_sw - weight_se - weight_nw
                # think in terms of NE divided into 4 rectangles with the sides
                # given by vtrans and the angle
                pca_new = np.zeros([PC_DIM_XY+2, PC_DIM_XY+2])   
                pca_new[1:-1, 1:-1] = pca90 
                
                weight_sw = (vtrans**2) *np.cos(dir90) * np.sin(dir90)
                weight_se = vtrans*np.sin(dir90) - \
                            (vtrans**2) * np.cos(dir90) * np.sin(dir90)
                weight_nw = vtrans*np.cos(dir90) - \
                            (vtrans**2) *np.cos(dir90) * np.sin(dir90)
                weight_ne = 1.0 - weight_sw - weight_se - weight_nw
          
                pca_new = pca_new*weight_ne + \
                          np.roll(pca_new, 1, 1) * weight_nw + \
                          np.roll(pca_new, 1, 0) * weight_se + \
                          np.roll(np.roll(pca_new, 1, 1), 1, 0) * weight_sw

                pca90 = pca_new[1:-1, 1:-1]
                pca90[1:, 0] = pca90[1:, 0] + pca_new[2:-1, -1]
                pca90[1, 1:] = pca90[1, 1:] + pca_new[-1, 2:-1]
                pca90[0, 0] = pca90[0, 0] + pca_new[-1, -1]

                #unrotate the pose cell xy layer
                self.posecells[:,:,dir_pc] = np.rot90(pca90, 
                                                   4 - int(np.floor(direction * 2/np.pi)))


        # Path Integration - Theta
        # Shift the pose cells +/- theta given by vrot
        if vrot != 0: 
            weight = (np.abs(vrot)/PC_C_SIZE_TH)%1
            if weight == 0:
                weight = 1.0

            shift1 = int(np.sign(vrot) * int(np.floor(abs(vrot)/PC_C_SIZE_TH)))
            shift2 = int(np.sign(vrot) * int(np.ceil(abs(vrot)/PC_C_SIZE_TH)))
            self.posecells = np.roll(self.posecells, shift1, 2) * (1.0 - weight) + \
                             np.roll(self.posecells, shift2, 2) * (weight)
        
    def find_best(self):
        '''Find the x, y, th center of the activity in the network.'''
        
        x, y, z = np.unravel_index(np.argmax(self.posecells), self.posecells.shape)
        
        z_posecells = np.zeros([PC_DIM_XY, PC_DIM_XY, PC_DIM_TH]) 
      
        zval = self.posecells[np.ix_(
            PC_AVG_XY_WRAP[x:x+PC_CELLS_TO_AVG*2], 
            PC_AVG_XY_WRAP[y:y+PC_CELLS_TO_AVG*2], 
            PC_AVG_TH_WRAP[z:z+PC_CELLS_TO_AVG*2]
        )]
        z_posecells[np.ix_(
            PC_AVG_XY_WRAP[x:x+PC_CELLS_TO_AVG*2], 
            PC_AVG_XY_WRAP[y:y+PC_CELLS_TO_AVG*2], 
            PC_AVG_TH_WRAP[z:z+PC_CELLS_TO_AVG*2]
        )] = zval
        
        # get the sums for each axis
        x_sums = np.sum(np.sum(z_posecells, 2), 1) 
        y_sums = np.sum(np.sum(z_posecells, 2), 0)
        th_sums = np.sum(np.sum(z_posecells, 1), 0)
        th_sums = th_sums[:]
        
        # now find the (x, y, th) using population vector decoding to handle 
        # the wrap around 
        x = (np.arctan2(np.sum(PC_XY_SUM_SIN_LOOKUP*x_sums), 
                        np.sum(PC_XY_SUM_COS_LOOKUP*x_sums)) * \
            PC_DIM_XY/(2*np.pi)) % (PC_DIM_XY)
            
        y = (np.arctan2(np.sum(PC_XY_SUM_SIN_LOOKUP*y_sums), 
                        np.sum(PC_XY_SUM_COS_LOOKUP*y_sums)) * \
            PC_DIM_XY/(2*np.pi)) % (PC_DIM_XY)
            
        th = (np.arctan2(np.sum(PC_TH_SUM_SIN_LOOKUP*th_sums), 
                         np.sum(PC_TH_SUM_COS_LOOKUP*th_sums)) * \
             PC_DIM_TH/(2*np.pi)) % (PC_DIM_TH)

        self.best_x = x
        self.best_y = y
        self.best_th = th

    def find_best_new(self):

        x, y, th = np.unravel_index(self.posecells.argmax(), self.posecells.shape)
        mx = self.posecells[x,y,th]

        # get the sums for each axis
        x_sums = np.zeros(PC_DIM_XY)
        y_sums = np.zeros(PC_DIM_XY)
        z_sums = np.zeros(PC_DIM_TH)

        for i in range(x - PC_CELLS_TO_AVG, x + PC_CELLS_TO_AVG + 1):
            for j in range(y - PC_CELLS_TO_AVG, y + PC_CELLS_TO_AVG + 1):
                for k in range(th - PC_CELLS_TO_AVG, th + PC_CELLS_TO_AVG + 1):
                    # Use modulo for wrapping
                    im = i % PC_DIM_XY
                    jm = j % PC_DIM_XY
                    km = k % PC_DIM_TH
                    x_sums[im] += self.posecells[im ,jm, km]
                    y_sums[jm] += self.posecells[im ,jm, km]
                    z_sums[km] += self.posecells[im ,jm, km]

        # now find the (x, y, th) using population vector decoding to handle the wrap around
        sum_x1 = 0
        sum_x2 = 0
        sum_y1 = 0
        sum_y2 = 0
        
        for i in range(PC_DIM_XY):
            sum_x1 += PC_XY_SUM_SIN_LOOKUP[i] * x_sums[i]
            sum_x2 += PC_XY_SUM_COS_LOOKUP[i] * x_sums[i]
            sum_y1 += PC_XY_SUM_SIN_LOOKUP[i] * y_sums[i]
            sum_y2 += PC_XY_SUM_COS_LOOKUP[i] * y_sums[i]

        x = np.arctan2(sum_x1, sum_x2) * PC_DIM_XY / (2.0 * np.pi) - 1.0
        while x < 0:
            x += PC_DIM_XY
        while x > PC_DIM_XY:
            x -= PC_DIM_XY

        y = np.arctan2(sum_y1, sum_y2) * PC_DIM_XY / (2.0 * np.pi) - 1.0
        while y < 0:
            y += PC_DIM_XY
        while x > PC_DIM_XY:
            y -= PC_DIM_XY

        sum_x1 = 0
        sum_x2 = 0
        for i in range(PC_DIM_TH):
            sum_x1 += PC_TH_SUM_SIN_LOOKUP[i] * z_sums[i]
            sum_x2 += PC_TH_SUM_COS_LOOKUP[i] * z_sums[i]

        th = np.arctan2(sum_x1, sum_x2) * PC_DIM_TH / (2.0 * np.pi) - 1.0
        while th < 0:
            th += PC_DIM_TH
        while x > PC_DIM_TH:
            th -= PC_DIM_TH

        self.best_x = x
        self.best_y = y
        self.best_th = th
        

    def get_action(self):

        delta_pc = 0
        action = NO_ACTION

        if (self.odo_update & self.vt_update):
            self.odo_update = False
            self.vt_update = False
        else:
            return action # NO_ACTION

        if len(self.visual_templates) == 0:
            return action # NO_ACTION
        
        if len(self.experiences) == 0:
            self.create_experience()
            action = CREATE_NODE
        else:
            experience = self.experiences[self.current_exp]

            delta_pc = self.get_delta_pc(experience['x_pc'], experience['y_pc'], experience['th_pc'])

            pcvt = self.visual_templates[self.current_vt]

            if len(pcvt['exps']) == 0:
                self.create_experience()
                action = CREATE_NODE
            elif delta_pc > EXP_DELTA_PC_THRESHOLD or self.current_vt != self.prev_vt:
                # go through all the exps associated with the current view and find the one with the closest delta_pc
                matched_exp_id = -1
                min_delta_id = -1
                #min_delta = int('infinity')
                min_delta = 100000 # should be very large number
                #delta_pc_tmp??

                # find the closest experience in cell space
                for i in range(len(pcvt['exps'])):
                    # make sure we aren't comparing to the current experience
                    if self.current_exp == pcvt['exps'][i]:
                        continue

                    experience = self.experiences[pcvt['exps'][i]]
                    delta_pc = self.get_delta_pc(experience['x_pc'], experience['y_pc'], experience['th_pc'])

                    if delta_pc < min_delta:
                        min_delta = delta_pc
                        min_delta_id = pcvt['exps'][i]

                # if an experience is closer than the thresh create a link
                if min_delta < EXP_DELTA_PC_THRESHOLD:
                    matched_exp_id = min_delta_id
                    action = CREATE_EDGE

                if self.current_exp != matched_exp_id:
                    if matched_exp_id == -1:
                        self.create_experience()
                        action = CREATE_NODE
                    else:
                        self.current_exp = matched_exp_id
                        if action == NO_ACTION:
                            action = SET_NODE


                elif self.current_vt == self.prev_vt:
                    self.create_experience()
                    action = CREATE_NODE

        return action

    def get_delta_pc(self, x, y, th):

        pc_th_corrected = self.best_th - self.vt_delta_pc_th
        
        if pc_th_corrected < 0:
            pc_th_corrected += PC_DIM_TH
        if pc_th_corrected >= PC_DIM_TH:
            pc_th_corrected -= PC_DIM_TH
        
        return np.sqrt(self.get_min_delta(self.best_x, x, PC_DIM_XY)**2 +\
                       self.get_min_delta(self.best_y, y, PC_DIM_XY)**2 +\
                       self.get_min_delta(self.best_th, th, PC_DIM_TH)**2
                      )

    def get_min_delta(self, d1, d2, mx):
        absval = abs(d1 - d2)
        return min(absval, mx - absval)





if __name__ == '__main__':
    posecells = PosecellNetwork()
    rospy.spin()
