import copy
import random
import numpy as np

from FFNN import FFNN_tanh


# Linear Plant Model

# Reach from Xs to Xf (R3)
# Intermediate command C = Xf-Xs+D (D = disturbance)

# Forward model is a single-hidden layer FFNN
# Inverse approximator is a 2-hidden layer FFNN


class LinearModel():
    def __init__(self, params={}):
        """
        Class Constructor

        Pass in forward model path and inverse model path
        """
        self.params = {                                 # default parameters
            'FM_path': None,                            # location of forward model tensorflow model
            'IM_path': None,                            # location of inverse model tensorflow model
            'constraints': [(0,100),(0,100),(0,100)],   # constraints in 3-dimensions
            'FM_learnrate': 0.0001,
            'FM_learnsteps': 1,
            'IM_learnrate': 0.000005,
            'IM_learnsteps': 1,
        }

        for p in params.keys():                         # replace defaults
            self.params[p] = params[p]

        self.initReach()

    def randgenXs(self, numX):
        """
        Randomly generate coordinates within constraints
        """
        X = []
        for n in range(numX):
            x = random.random()*(self.params['constraints'][0][1]-\
                self.params['constraints'][0][0])+self.params['constraints'][0][0]
            y = random.random()*(self.params['constraints'][1][1]-\
                self.params['constraints'][1][0])+self.params['constraints'][1][0]
            z = random.random()*(self.params['constraints'][2][1]-\
                self.params['constraints'][2][0])+self.params['constraints'][2][0]
            X.append(np.array([x,y,z]))
        return np.array(X)

    def mux(self, sets):
        """
        Multiplexer, to merge data so it may be passed into neural networks
        """
        merged = sets[0]
        for s in sets[1:]:
            merged = np.concatenate((merged,s),axis=1)
        return merged

    def runPlant(self, Xs, C, variation, D):
        """
        Get plant output
        """
        if variation.lower() == "linear":
            return Xs+C+D
        elif variation.lower() == "inverse":
            return Xs-C
        elif variation.lower() == "halfcommand":
            return Xs+C/2



### INTERNAL FORWARD MODEL NEURAL NETWORK (CEREBELLUM) ###
# Cerebellum internal forward model
# Inputs:
#   1) start position (R^3)
#   2) efference copy of generated command (R^3)
# Output:
#   estimate of resulting end position (R^3)
# Structure:
#   - one hidden layer (256 neurons)
#   - tanh activation in hidden layer
#   - sigmoid activation in output layer
#   - R2 regularization with coefficient of 0.001
#   - no dropout

    def initFM(self):
        self.FM = FFNN_tanh(hidden_layers=[256], num_inputs=6, num_outputs=3, drop=False, regbeta=0.001, model=self.params['FM_path'])

    def openSessionFM(self):
        self.FM.openSession()

    def closeSessionFM(self):
        self.FM.closeSession()

    def itrainFM(self, num_points=100000, learning_rate=0.001, num_steps=100000, batch_size=256, display_step=1000):
        Xs = self.randgenXs(num_points)     # random start points
        Xf = self.randgenXs(num_points)     # random end points
        C = Xf-Xs                           # corresponding commands

        self.FM.train(self.mux([Xs,C]),Xf,learning_rate,num_steps,batch_size,display_step)  # training

    def saveFM(self):
        self.FM.saveNN()

    def runFM(self, Xs, C):
        return self.FM.run(self.mux([[Xs],[C]]))[0]


#### INVERSE NEURAL NETWORK MODEL (BRAIN STEM) ###
# Inverse neural network kinematics model
# Input:
#   1) start point (R^3)
#   2) desired end point (R^3)
# Output:
#   command (R^3)
# Structure:
#   - two hidden layer (128, 128 neurons)
#   - tanh activation in hidden layers
#   - sigmoid activation in output layer
#   - R2 regularization with coefficient of 0.001
#   - no dropout

    def initIM(self):
        self.IM = FFNN_tanh(hidden_layers=[128, 128], num_inputs=6, num_outputs=3, drop=False, regbeta=0.001, model=self.params['IM_path'])

    def openSessionIM(self):
        self.IM.openSession()

    def closeSessionIM(self):
        self.IM.closeSession()

    def itrainIM(self, num_points=100000, learning_rate=0.001, num_steps=100000, batch_size=256, display_step=1000):
        Xs = self.randgenXs(num_points)
        Xf = self.randgenXs(num_points)
        C = Xf-Xs

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
        self.prevXd = np.array([0,0,0])
        self.Xd = np.array([0,0,0])
        self.prevE = np.array([0,0,0])


    def noFMReach(self, Xs, Xd, variation):
        """
        Reach without recurrent control architecture (disregard cerebellar forward model)
        """
        C = self.runIM(Xs, Xd)
        Xf = self.runPlant(Xs, C, variation, np.array([0,0,0]))
        return Xf

    def onlineReach(self, Xs, Xd, variation, D, FM_learn=True, IM_learn=True):
        """
        Online reach with recurrent control and plant feedback
        """

        Xd_changed = False                          # If reach target changes, reset history
        if np.linalg.norm(self.Xd - Xd) != 0:
            self.Xd = Xd
            Xd_changed = True
            self.prevXd = Xd

        self.prevXd = self.prevXd+self.prevE        # update recursion block
        C = self.runIM(Xs, self.prevXd)             # get command from inverse model
        Xf = self.runPlant(Xs, C, variation, D)     # get actual plant output

        Xp = self.runFM(Xs, C)                      # get prediction from forward model
        self.prevE = Xd-Xp                          # update error for next recursive update

        # Online learning/adaptation
        if FM_learn:
            self.FM.train(self.mux([[Xs],[C]]),[Xf],self.params['FM_learnrate'],self.params['FM_learnsteps'],1)
#        if IM_learn:
#            self.IM.train(self.mux([[Xs],[Xf]]),[C],self.params['IM_learnrate'],self.params['IM_learnsteps'],1)

        return Xf

    def offlineReach(self, Xs, Xd, variation, D, FM_learn=True, IM_learn=True):
        """
        Offline reach with recurrent control and no plant feedback
        """
        C = self.runIM(Xs, Xd)                      # generated command from inverse model
        Xp = self.runFM(Xs, C)                      # predicted output from forward model

        Xf = self.runPlant(Xs, C, variation, D)     # plant output (doesn't actually happen - for visualization purposes online)

        # Offline learning/adaptation
        if FM_learn:
            self.FM.train(self.mux([[Xs],[C]]),[Xd],self.params['FM_learnrate'],self.params['FM_learnsteps'],1)
        if IM_learn:
            self.IM.train(self.mux([[Xs],[Xp]]),[C],self.params['IM_learnrate'],self.params['IM_learnsteps'],1)

        return Xf


if __name__ == "__main__":
    pass