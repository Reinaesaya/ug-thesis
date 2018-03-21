import numpy as np
import random
import matplotlib.pyplot as plt
import sys

from LinearModel import LinearModel
from RRModel import RRModel
from RPRModel import RPRModel


class LinearSimulator():
    def __init__(self, FM_path, IM_path, FM_itrain=False, IM_itrain=False):
        """
        Initialize model
        """

        self.params = {                                 # default parameters
            'FM_path': FM_path,                         # location of forward model tensorflow model
            'IM_path': IM_path,                         # location of inverse model tensorflow model
            'constraints': [(0,100),(0,100),(0,100)],   # constraints in 3-dimensions
            'FM_learnrate': 0.0001,
            'FM_learnsteps': 1,
            'IM_learnrate': 0.000005,
            'IM_learnsteps': 1,
        }

        self.instantiateModel()
        self.initializeModel(FM_itrain, IM_itrain)

    def instantiateModel(self):
        self.model = LinearModel(self.params)       # Instantiate

    def initializeModel(self, FM_itrain, IM_itrain):
        self.model.initFM()                         # Initialize forward model if exists
        self.model.openSessionFM()                  # Open session and load forward model if exists
        if FM_itrain:                               # Train forward model (no disturbance)
            self.model.itrainFM()
            self.model.saveFM()

        self.model.initIM()                         # Initialize inverse model if exists
        self.model.openSessionIM()                  # Open session and load inverse model if exists
        if IM_itrain:                               # Train inverse model (no disturbance)
            self.model.itrainIM()
            self.model.saveIM()


    def LTreachexperiment(self, Xs, Xd, D, onlinereps, offlinereps, numsessions, saveafter=False):
        """
        Long term reach experiment

        Learn to reach from Xs to Xd with constant disturbance D

        numsessions --> number of practice sessions
        onlinereps --> reach repetitions per online session
        offlinereps --> reach repetitions per offline session
        """
        Xs = np.array(Xs)
        Xd = np.array(Xd)
        D = np.array(D)

        # For keeping track of rep number and graphing purposes
        online_r = []
        online_e = []

        offline_r = []
        offline_e = []

        total_r = []
        total_e = []

        rep = 0
        for s in range(numsessions):
            
            for r in range(onlinereps):
                rep += 1
                online_r.append(rep)
                total_r.append(rep)

                Xf = self.model.onlineReach(Xs, Xd, D)                  # Online reach
                E = np.linalg.norm(Xf-Xd)                               # Error
                online_e.append(E)
                total_e.append(E)
            
            for r in range(offlinereps):
                rep += 1
                offline_r.append(rep)
                total_r.append(rep)

                Xf = self.model.offlineReach(Xs, Xd, D)                 # Offline reach
                E = np.linalg.norm(Xf-Xd)                               # Error
                offline_e.append(E)
                total_e.append(E)

        if saveafter:
            self.model.saveFM()
            self.model.saveIM()

        plt.figure()                                                   # Plot error vs reps
        plt.plot(online_r, online_e, '.', offline_r, offline_e, '.')
        plt.ylabel('error')
        plt.legend(['Online', 'Offline'])
        plt.show()

    def close(self):
        self.model.closeSessionFM()                 # Close forward model session
        self.model.closeSessionIM()                 # Close inverse model session


class RRSimulator(LinearSimulator):
    def __init__(self, FM_path, IM_path, FM_itrain=False, IM_itrain=False):
        """
        Initialize model
        """

        self.params = {                                 # default parameters
            'FM_path': FM_path,                            # location of forward model tensorflow model
            'IM_path': IM_path,                            # location of inverse model tensorflow model
            'a1': 100,                                  # link 1 length
            'a2': 100,                                  # link 2 length
            'arm_direction': 'right',                   # right or left arm
            'FM_learnrate': 0.0001,
            'FM_learnsteps': 1,
            'IM_learnrate': 0.000002,
            'IM_learnsteps': 1,
        }

        self.instantiateModel()
        self.initializeModel(FM_itrain, IM_itrain)

    def instantiateModel(self):
        self.model = RRModel(self.params)       # Instantiate

    def LTreachexperiment(self, Qs, Xd, D, onlinereps, offlinereps, numsessions, saveafter=False):
        """
        Long term reach experiment

        Learn to reach from Xs (converted from Qs) to Xd with constant disturbance D

        numsessions --> number of practice sessions
        onlinereps --> reach repetitions per online session
        offlinereps --> reach repetitions per offline session
        """
        Qs = np.array(Qs)
        Xd = np.array(Xd)
        D = np.array(D)

        # For keeping track of rep number and graphing purposes
        online_r = []
        online_e = []

        offline_r = []
        offline_e = []

        total_r = []
        total_e = []

        rep = 0
        for s in range(numsessions):
            
            for r in range(onlinereps):
                rep += 1
                online_r.append(rep)
                total_r.append(rep)

                Xf = self.model.onlineReach(Qs, Xd, D)                  # Online reach
                E = np.linalg.norm(Xf-Xd)                               # Error
                online_e.append(E)
                total_e.append(E)
            
            for r in range(offlinereps):
                rep += 1
                offline_r.append(rep)
                total_r.append(rep)

                Xf = self.model.offlineReach(Qs, Xd, D)                 # Offline reach
                E = np.linalg.norm(Xf-Xd)                               # Error
                offline_e.append(E)
                total_e.append(E)

        if saveafter:
            self.model.saveFM()
            self.model.saveIM()

        plt.figure()                                                   # Plot error vs reps
        plt.plot(online_r, online_e, '.', offline_r, offline_e, '.')
        plt.ylabel('error')
        plt.legend(['Online', 'Offline'])
        plt.show()

    def close(self):
        self.model.closeSessionFM()                 # Close forward model session
        self.model.closeSessionIM()                 # Close inverse model session


class RPRSimulator(RRSimulator):
    def __init__(self, FM_path, IM_path, FM_itrain=False, IM_itrain=False):
        """
        Initialize model
        """

        self.params = {                                 # default parameters
            'FM_path': FM_path,                            # location of forward model tensorflow model
            'IM_path': IM_path,                            # location of inverse model tensorflow model
            'a1': 100,                                  # link 1 length
            'a2': 100,                                  # link 2 length
            'arm_direction': 'right',                   # right or left arm
            'FM_learnrate': 0.001,
            'FM_learnsteps': 1,
            'IM_learnrate': 0.0001,
            'IM_learnsteps': 1,
        }

        self.instantiateModel()
        self.initializeModel(FM_itrain, IM_itrain)


    def instantiateModel(self):
        self.model = RPRModel(self.params)       # Instantiate

if __name__ == "__main__":

# Linear Model Simulation
    myLinSim = LinearSimulator('./models/linearModel_FM.ckpt', './models/linearModel_IM.ckpt', False, False)
    myLinSim.LTreachexperiment([0,0,0], [20,30,40], [10,10,5], 50, 75, 5, False)
    myLinSim.close()

# RR Manipulator Model Simulation
    myRRSim = RRSimulator('./models/RRModel_FM.ckpt', './models/RRModel_IM.ckpt', False, False)
    myRRSim.LTreachexperiment(myRRSim.model.restQ, [40,60], [np.pi/8, np.pi/8], 50, 75, 5, False)
    myRRSim.close()

# RPR Manipulator Model Simulation
    myRPRSim = RPRSimulator('./models/RPRModel_FM.ckpt', './models/RPRModel_IM.ckpt', False, False)
    myRPRSim.LTreachexperiment(myRPRSim.model.restQ, [40,60], [np.pi/8, -10, np.pi/8], 50, 75, 5, False)
    myRPRSim.close()