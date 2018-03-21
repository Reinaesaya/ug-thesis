import copy
import numpy as np
from numpy import cos, sin

import random

from FFNN import FFNN_tanh
from ArmBase import link, ArmBase


# RR Manipulator robot

# Includes a forward model using a single-hidden layer feedforward neural network 
# to model the cerebellum (which is an error corrector)
# Also includes an inverse model using a 2-hidden layer feedforward neural network
# To model motor command generation in other parts of the brain


class RRModel(ArmBase):
    def __init__(self, params={}):
        """
        Class Constructor

        Pass in forward model path and inverse model path
        """
        self.params = {                                 # default parameters
            'FM_path': None,                            # location of forward model tensorflow model
            'IM_path': None,                            # location of inverse model tensorflow model
            'a1': 100,                                  # link 1 length
            'a2': 100,                                  # link 2 length
            'arm_direction': 'right',                   # right or left arm
            'FM_learnrate': 0.0001,
            'FM_learnsteps': 1,
            'IM_learnrate': 0.000005,
            'IM_learnsteps': 1,
        }

        for p in params.keys():                         # replace defaults
            self.params[p] = params[p]

        link1 = link('revolute', self.params['a1'], np.pi/2, 0, 0)
        link2 = link('revolute', self.params['a2'], 0, 0, 0)
        self.links = [link1, link2]
        self._formDH()

        self.setarmconstraints()
        self.initReach()

    def setconstraints(self, l1, l2):
        self.links[0].setconstraints(l1)    # revolute 1
        self.links[1].setconstraints(l2)    # revolute 2

    def setarmconstraints(self):
        if self.params['arm_direction'].lower() == 'right':
            self.setconstraints([-np.pi/4, np.pi/2], [0, 3*np.pi/4])
            self.restQ = np.array([-np.pi/6, np.pi/2])
        elif self.params['arm_direction'].lower() == 'left':
            self.setconstraints([np.pi/2, 5*np.pi/4], [-3*np.pi/4, 0])
            self.restQ = np.array([7*np.pi/6, -np.pi/2])

    def randgenQs(self, numQ):
        """
        Randomly generate joint angles within constraints
        """
        Q = []
        for n in range(numQ):
            q1 = random.random()*self.links[0].c_range + self.links[0].c_span[0]
            q2 = random.random()*self.links[1].c_range + self.links[1].c_span[0]
            Q.append(np.array([q1,q2]))
        return Q

    def mux(self, sets):
        """
        Multiplexer, to merge data so it may be passed into neural networks
        """
        merged = sets[0]
        for s in sets[1:]:
            merged = np.concatenate((merged,s),axis=1)
        return merged

### INTERNAL FORWARD MODEL NEURAL NETWORK (CEREBELLUM) ###
# Cerebellum internal forward model
# Inputs:
#   1) start position (R^2)
#   2) efference copy of generated command (R^2)
# Output:
#   estimate of resulting end position (R^2)
# Structure:
#   - one hidden layer (512 neurons)
#   - tanh activation in hidden layer
#   - sigmoid activation in output layer
#   - R2 regularization with coefficient of 0.001
#   - no dropout

    def initFM(self):
        self.FM = FFNN_tanh(hidden_layers=[512], num_inputs=4, num_outputs=2, drop=False, regbeta=0.001, model=self.params['FM_path'])

    def openSessionFM(self):
        self.FM.openSession()

    def closeSessionFM(self):
        self.FM.closeSession()

    def itrainFM(self, num_points=100000, learning_rate=0.001, num_steps=100000, batch_size=256, display_step=1000):

        Qs = self.randgenQs(num_points)                     # random starting link params
        Xs = np.array([self.solveFK(q)[0:2,3] for q in Qs])
        
        Qf = self.randgenQs(num_points)                     # random ending link params
        Xf = np.array([self.solveFK(q)[0:2,3] for q in Qf])

        C = np.array(Qf)-np.array(Qs)                       # corresponding commands

        self.FM.train(self.mux([Xs,C]),Xf,learning_rate,num_steps,batch_size,display_step)  # training

    def saveFM(self):
        self.FM.saveNN()

    def runFM(self, Xs, C):
        return self.FM.run(self.mux([[Xs],[C]]))[0]

#### INVERSE NEURAL NETWORK MODEL (BRAIN STEM) ###
# Inverse neural network kinematics model
# Input:
#   1) start point (R^2)
#   2) desired end point (R^2)
# Output:
#   motor command (R^2)
# Structure:
#   - two hidden layer (256, 256 neurons)
#   - tanh activation in hidden layers
#   - sigmoid activation in output layer
#   - R2 regularization with coefficient of 0.001
#   - no dropout

    def initIM(self):
        self.IM = FFNN_tanh(hidden_layers=[256, 256], num_inputs=4, num_outputs=2, drop=False, regbeta=0.001, model=self.params['IM_path'])

    def openSessionIM(self):
        self.IM.openSession()

    def closeSessionIM(self):
        self.IM.closeSession()

    def itrainIM(self, num_points=100000, learning_rate=0.001, num_steps=100000, batch_size=256, display_step=1000):

        Qs = self.randgenQs(num_points)                     # random starting link params
        Xs = np.array([self.solveFK(q)[0:2,3] for q in Qs])
        
        Qf = self.randgenQs(num_points)                     # random ending link params
        Xf = np.array([self.solveFK(q)[0:2,3] for q in Qf])

        C = np.array(Qf)-np.array(Qs)                       # corresponding commands

        self.IM.train(self.mux([Xs,Xf]),C,learning_rate,num_steps,batch_size,display_step)

    def saveIM(self):
        self.IM.saveNN()

    def runIM(self, Xs, Xf):
        return self.IM.run(self.mux([[Xs],[Xf]]))[0]


### RECURRENT ADAPTIVE NEURAL NETWORK CONTROL ###

    def initReach(self):
        """
        Initialize recursive block in recurrent structure
        """
        self.prevXd = np.array([0,0])
        self.Xd = np.array([0,0])
        self.prevE = np.array([0,0])


    def noFMReach(self, Qs, Xd):
        """
        Reach without recurrent control architecture (disregard cerebellar forward model)
        """
        Xs = np.array(self.solveFK(Qs)[0:2,3])
        C = self.runIM(Xs, Xd)
        Xf = self.runPlant(Qs, C, np.array([0,0]))
        return Xf

    def onlineReach(self, Qs, Xd, D, FM_learn=True, IM_learn=True):
        """
        Online reach with recurrent control and plant feedback
        """
        Xs = np.array(self.solveFK(Qs)[0:2,3])      # Convert start joint parameters to planar sensory coordinates

        Xd_changed = False                          # If reach target changes, reset history
        if np.linalg.norm(self.Xd - Xd) != 0:
            self.Xd = Xd
            Xd_changed = True
            self.prevXd = Xd

        self.prevXd = self.prevXd+self.prevE        # update recursion block
        C = self.runIM(Xs, self.prevXd)             # get command from inverse model
        Xf = self.runPlant(Qs, C, D)                # get actual plant output

        Xp = self.runFM(Xs, C)                      # get prediction from forward model
        self.prevE = Xd-Xp                          # update error for next recursive update

        # Online learning/adaptation
        if FM_learn:
            self.FM.train(self.mux([[Xs],[C]]),[Xf],self.params['FM_learnrate'],self.params['FM_learnsteps'],1)
        if IM_learn:
            self.IM.train(self.mux([[Xs],[Xf]]),[C],self.params['IM_learnrate'],self.params['IM_learnsteps'],1)

        return Xf   

    def offlineReach(self, Qs, Xd, D, FM_learn=True, IM_learn=True):
        """
        Offline reach with recurrent control and no plant feedback
        """
        Xs = np.array(self.solveFK(Qs)[0:2,3])  # Convert start joint parameters to planar sensory coordinates
        C = self.runIM(Xs, Xd)                  # generated command from inverse model
        Xp = self.runFM(Xs, C)                  # predicted output from forward model

        Xf = self.runPlant(Qs, C, D)            # plant output (doesn't actually happen - for visualization purposes online)

        # Offline learning/adaptation
        if FM_learn:
            self.FM.train(self.mux([[Xs],[C]]),[Xd],self.params['FM_learnrate'],self.params['FM_learnsteps'],1)
        if IM_learn:
            self.IM.train(self.mux([[Xs],[Xp]]),[C],self.params['IM_learnrate'],self.params['IM_learnsteps'],1)

        return Xf


if __name__ == "__main__":
    pass