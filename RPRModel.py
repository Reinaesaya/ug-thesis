import copy
import numpy as np
from numpy import cos, sin

import random

from FFNN import FFNN_tanh
from ArmBase import link, ArmBase
from RRModel import RRModel


# RR Manipulator robot

# Includes a forward model using a single-hidden layer feedforward neural network 
# to model the cerebellum (which is an error corrector)
# Also includes an inverse model using a 2-hidden layer feedforward neural network
# To model motor command generation in other parts of the brain


class RPRModel(RRModel):
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

        link1 = link('revolute', 0, np.pi/2, 0, np.pi/2)
        link2 = link('prismatic', 0, -np.pi/2, 0, 0)
        link3 = link('revolute', self.params['a2'], 0, 0, -np.pi/2)
        self.links = [link1, link2, link3]
        self._formDH()

        self.setarmconstraints()
        self.initReach()

    def setconstraints(self, l1, l2, l3):
        self.links[0].setconstraints(l1)    # revolute 1
        self.links[1].setconstraints(l2)    # prismatic 2
        self.links[2].setconstraints(l3)    # revolute 3

    def setarmconstraints(self):
        if self.params['arm_direction'].lower() == 'right':
            self.setconstraints([0, np.pi/2], [0,self.params['a1']], [-np.pi/2, 0])
            self.restQ = np.array([0, 30, -np.pi/4])
        elif self.params['arm_direction'].lower() == 'left':
            self.setconstraints([np.pi/2, np.pi], [0,self.params['a1']], [-np.pi, -np.pi/2])
            self.restQ = np.array([np.pi, 30, -3*np.pi/4])

    def randgenQs(self, numQ):
        """
        Randomly generate joint angles within constraints
        """
        Q = []
        for n in range(numQ):
            q1 = random.random()*self.links[0].c_range + self.links[0].c_span[0]
            q2 = random.random()*self.links[1].c_range + self.links[1].c_span[0]
            q3 = random.random()*self.links[2].c_range + self.links[2].c_span[0]
            Q.append(np.array([q1,q2,q3]))
        return Q


### INTERNAL FORWARD MODEL NEURAL NETWORK (CEREBELLUM) ###
# Cerebellum internal forward model
# Inputs:
#   1) start position (R^2)
#   2) efference copy of generated command (R^3)
# Output:
#   estimate of resulting end position (R^2)
# Structure:
#   - one hidden layer (512 neurons)
#   - tanh activation in hidden layer
#   - sigmoid activation in output layer
#   - R2 regularization with coefficient of 0.001
#   - no dropout

    def initFM(self):
        self.FM = FFNN_tanh(hidden_layers=[512], num_inputs=5, num_outputs=2, drop=False, regbeta=0.001, model=self.params['FM_path'])


#### INVERSE NEURAL NETWORK MODEL (BRAIN STEM) ###
# Inverse neural network kinematics model
# Input:
#   1) start point (R^2)
#   2) desired end point (R^2)
# Output:
#   motor command (R^3)
# Structure:
#   - two hidden layer (256, 256 neurons)
#   - tanh activation in hidden layers
#   - sigmoid activation in output layer
#   - R2 regularization with coefficient of 0.001
#   - no dropout

    def initIM(self):
        self.IM = FFNN_tanh(hidden_layers=[256, 256], num_inputs=4, num_outputs=3, drop=False, regbeta=0.001, model=self.params['IM_path'])


### RECURRENT ADAPTIVE NEURAL NETWORK CONTROL ###

    def noFMReach(self, Qs, Xd, variation):
        """
        Reach without recurrent control architecture (disregard cerebellar forward model)
        """
        Xs = np.array(self.solveFK(Qs)[0:2,3])
        C = self.runIM(Xs, Xd)
        Xf = self.runPlant(Qs, C, variation, np.array([0,0,0]))
        return Xf


if __name__ == "__main__":
    pass