import copy
import numpy as np
from numpy import cos, sin

import random

from FFNN import FFNN_Regression

class link():
	def __init__(self, tp, a, alpha, d, theta):
		self.tp = tp	# link type (revolute, prismatic)
		self.a = a
		self.alpha = alpha
		self.d = d
		self.theta = theta

	def setconstraints(self, constraints):
		self.c_span = [min(constraints), max(constraints)]
		self.c_range = self.c_span[1]-self.c_span[0]

class myRobot():
	def __init__(self, links):
		self.links = copy.deepcopy(links)
		self._formDH()
	
	def _formDH(self):
		DH = []
		for link in self.links:
			DH.append([link.a, link.alpha, link.d, link.theta])
		self.DH = np.array(DH)

	def forward(self, joints):
		""" Return H matrix for forward kinematics """
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


class myRRrobot(myRobot):
	def __init__(self, link1len, link2len, elbow_dir):
		self.elbow_dir = elbow_dir # left or right

		link1 = link('revolute', link1len, 0, 0, 0)
		link2 = link('revolute', link2len, 0, 0, 0)
		self.links = [link1, link2]
		self._formDH()

		self.setdefaultconstraints()

	def setconstraints(self, l1, l2):
		self.links[0].setconstraints(l1)
		self.links[1].setconstraints(l2)

	def setdefaultconstraints(self):
		if self.elbow_dir == 'right':
			self.setconstraints([0, np.pi/2], [0, np.pi/2])
		elif self.elbow_dir == 'left':
			self.setconstraints([np.pi/2, np.pi], [-np.pi/2, 0])
		else:
			print('What is this elbow direction?')
			print(self.elbow_dir)

	def randgenQs(self, numQs):
		Qs = []
		for n in range(numQs):
			q1 = random.random()*self.links[0].c_range + self.links[0].c_span[0]
			q2 = random.random()*self.links[1].c_range + self.links[1].c_span[0]
			Qs.append(np.array([q1,q2]))
		return Qs

	def init_inverseNN(self):
		self.ffnn = FFNN_Regression(hidden_layers=[128], num_inputs=2, num_outputs=2)

	def train_inverseNN(self, num_points=100000, learning_rate=0.001, \
		num_steps=5000, batch_size=128,\
		display_step=100, model='./models/invkinRR.ckpt'):
		
		Qs = self.randgenQs(num_points)
		Xs = np.array([self.forward(q)[0:2,3] for q in Qs])
		self.ffnn.init_optimizer(learning_rate)
		self.ffnn.train(Xs, Qs, num_steps, batch_size, display_step, model)

	def test_inverseNN(self, Xs, Qs, model='./models/invkinRR.ckpt'):
		self.ffnn.test(Xs, Qs, model)

	def load_inverseNN(self, model='./models/invkinRR.ckpt'):
		# None is default
		self.ffnn.openSession(model)

	def close_inverseNN(self):
		self.ffnn.closeSession()

	def inverse(self, X):
		return self.ffnn.run([X])

	def inverseBatch(self, Xs, model='./models/invkinRR.ckpt'):
		self.load_inverseNN(model)
		nnQs = [self.inverse(X) for X in Xs]
		self.close_inverseNN()
		return nnQs



if __name__ == "__main__":
	myRR = myRRrobot(50,50,'right')
	myRR.init_inverseNN()
	myRR.train_inverseNN()