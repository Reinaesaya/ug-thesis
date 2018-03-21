import copy
import numpy as np
from numpy import cos, sin

import random

# Robot manipulator link
class link():
    def __init__(self, tp, a, alpha, d, theta):
        self.tp = tp    # link type (revolute, prismatic)
        self.a = a
        self.alpha = alpha
        self.d = d
        self.theta = theta

    def setconstraints(self, constraints):
        self.c_span = [min(constraints), max(constraints)]
        self.c_range = self.c_span[1]-self.c_span[0]


# Robot manipulator class that defined DH tables, links, and forward kinematics
class ArmBase():
    def __init__(self, links):
        self.links = copy.deepcopy(links)
        self._formDH()
    
    def _formDH(self):
        DH = [] 
        for link in self.links:
            DH.append([link.a, link.alpha, link.d, link.theta])
        self.DH = np.array(DH)

    def solveFK(self, joints):
        """
        Return H matrix for forward kinematics
        """
        H = np.identity(4)
        for i in range(len(joints)):
            q = joints[i]
            if self.links[i].tp == "revolute":
                theta = q
                d = self.links[i].d
            elif self.links[i].tp == "prismatic":
                theta = self.links[i].theta
                d = q
            else:
                pass
            alpha = self.links[i].alpha
            a = self.links[i].a

            Hq = np.array([
                [cos(theta), -sin(theta)*cos(alpha), sin(theta)*sin(alpha), a*cos(theta)],
                [sin(theta), cos(theta)*cos(alpha), -cos(theta)*sin(alpha), a*sin(theta)],
                [0, sin(alpha), cos(alpha), d],
                [0, 0, 0, 1]
                ])

            H = np.matmul(H,Hq)
        return H

    def runPlant(self, Qs, C, D):
        """
        Get plant output (simple addition)
        """
        return self.solveFK(Qs+C+D)[0:2,3]

if __name__ == "__main__":
    pass