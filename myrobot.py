import copy
import numpy as np
from numpy import cos, sin

import random

from FFNN import FFNN_1

# Robot manipulator link
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


# Robot manipulator class that defined DH tables, links, and forward kinematics
class myRobot():
	def __init__(self, links):
		self.links = copy.deepcopy(links)
		self._formDH()
	
	def _formDH(self):
		DH = []
		for link in self.links:
			DH.append([link.a, link.alpha, link.d, link.theta])
		self.DH = np.array(DH)

	def forward(self, joints, disturbance=np.zeros((4,4))):
		""" Return H matrix for forward kinematics with possible disturbance"""
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
		H = H + disturbance
		return H

# RR Manipulator robot
# defines neural network mapping signals to joint angles
# signals are made with the simple assumption of 1-to-1 mapping to desired position
# so sort of like inverse kinematics, thus named (self.invmodel)
class myRRrobot(myRobot):
	def __init__(self, link1len, link2len, elbow_dir, invmodel):
		self.elbow_dir = elbow_dir # left or right
		self.invmodel = invmodel

		link1 = link('revolute', link1len, 0, 0, 0)
		link2 = link('revolute', link2len, 0, 0, 0)
		self.links = [link1, link2]
		self._formDH()

		self.setdefaultconstraints()

	def setconstraints(self, l1, l2):
		self.links[0].setconstraints(l1)
		self.links[1].setconstraints(l2)

	def setdefaultconstraints(self):
		if self.elbow_dir.lower() == 'right':
			self.setconstraints([0, np.pi/2], [0, np.pi])
		elif self.elbow_dir.lower() == 'left':
			self.setconstraints([np.pi/2, np.pi], [-np.pi, 0])
		else:
			print('What is this elbow direction?')
			print(self.elbow_dir)

	def randgenQs(self, numQs):
		# randomly generate certain number of joint angles
		Qs = []
		for n in range(numQs):
			q1 = random.random()*self.links[0].c_range + self.links[0].c_span[0]
			q2 = random.random()*self.links[1].c_range + self.links[1].c_span[0]
			Qs.append(np.array([q1,q2]))
		return Qs

	def randgenXs(self, numXs):
		# randomly generate certain number of desired points
		Xs = []
		for n in range(numXs):
			q1 = random.random()*self.links[0].c_range + self.links[0].c_span[0]
			q2 = random.random()*self.links[1].c_range + self.links[1].c_span[0]
			Xs.append(self.forward([q1,q2])[0:2,3])
		return Xs

	def randgenCs(self, numCs):
		# randomly generate certain number of commands
		# one to one f(x)=x relationship with desired points
		return self.randgenXs(numCs)

#### INVERSE KINEMATICS NEURAL NETWORK MODEL (BRAIN STEM) ###
# Inverse neural network kinematics model translates commands into joint angles (no disturbance)
# NN session open and closes
	def init_inverseNN(self):
		self.inv_ffnn = FFNN_1(hidden_layers=[128], num_inputs=2,\
			num_outputs=2, model=self.invmodel)

	def open_inverseNN_session(self):
		# None is default
		self.inv_ffnn.openSession()

	def close_inverseNN_session(self):
		self.inv_ffnn.closeSession()

# NN training
	def train_inverseNN(self, num_points=100000, learning_rate=0.001, \
		num_steps=100000, batch_size=128,\
		display_step=100):

		Qs = self.randgenQs(num_points)							# generate joint angle training set
		Xs = np.array([self.forward(q)[0:2,3] for q in Qs])		# corresponding points
		Cs = Xs 												# commands 1 to 1 relationship
		self.inv_ffnn.train(Cs, Qs, learning_rate, num_steps, batch_size, display_step)

	def save_inverseNN(self):
		self.inv_ffnn.saveNN()

# NN testing
	def test_inverseNN(self, num_points=10):
		Qs = self.randgenQs(num_points)
		Xs = np.array([self.forward(q)[0:2,3] for q in Qs])
		Cs = Xs
		print("Test commands:")
		print(Cs)
		loss = self.inv_ffnn.test(Cs, Qs)
		print(loss)

## NN inverse model
	def inverse(self, Cs, disturbance=[0,0]):
		# Run inverse model, with possible disturbance of angles
		Qs = self.inv_ffnn.run(Cs)
		Qs = [q+np.array(disturbance) for q in Qs]
		return Qs

	def inverseSingle(self, C, disturbance=[0,0]):
		Q = self.inverse([C], disturbance)
		return Q[0]

	def inverseBatch(self, Cs, disturbance=[0,0]):
		nnQs = self.inverse(Cs, disturbance)
		return nnQs


class RR_model1(myRRrobot):
# Follows a similar architecture to that in Porrill et al. 2014
# "Recurrent cerebellar architecture solves the motor-error problem"

# The forward model consists of one small neural network

	def __init__(self, link1len, link2len, elbow_dir, invmodel, internal_fmodel, internal_imodel):
		myRRrobot.__init__(self, link1len, link2len, elbow_dir, invmodel)
		self.internal_fmodel = internal_fmodel
		self.internal_imodel = internal_imodel
		self.initPIDcontrol()

### INTERNAL FORWARD MODEL NEURAL NETWORK (CEREBELLUM) ###
# Internal mapping of commands to end position
# named 'IF_NN' for Internal Forward Neural Network

# NN session open and closes
	def init_IF_NN(self):
		self.IF_nn = FFNN_1(hidden_layers=[64], num_inputs=2,\
			num_outputs=2, model=self.internal_fmodel)

	def open_IF_NN_session(self):
		# None is default
		self.IF_nn.openSession()

	def close_IF_NN_session(self):
		self.IF_nn.closeSession()

# IF_NN initial training (no disturbance)
	def train_IF_NN(self, num_points=100000, learning_rate=0.001, \
		num_steps=100000, batch_size=128,\
		display_step=100):

		Cs = self.randgenCs(num_points)
		Qs = self.inverseBatch(Cs)
		Ys = np.array([self.forward(q)[0:2,3] for q in Qs])
		self.IF_nn.train(Cs, Ys, learning_rate, num_steps, batch_size, display_step)

	def save_IF_NN(self):
		self.IF_nn.saveNN()

# IF_NN testing
	def test_IF_NN(self, num_points=10):
		Cs = self.randgenCs(num_points)
		Qs = self.inverseBatch(Cs)
		Ys = np.array([self.forward(q)[0:2,3] for q in Qs])
		print("Test Ys:")
		print(Ys)
		loss = self.IF_nn.test(Cs, Ys)
		print(loss)

## IF_NN forward run (single input only)
	def IF_run(self, C):
		Y = self.IF_nn.run([C])
		return Y[0]


### INTERNAL INVERSE MODEL NEURAL NETWORK (BRAIN STEM) ###
# Internal mapping of end position to command
# named 'II_NN' for Internal Inverse Neural Network

# NN session open and closes
	def init_II_NN(self):
		self.II_nn = FFNN_1(hidden_layers=[64], num_inputs=2,\
			num_outputs=2, model=self.internal_imodel)

	def open_II_NN_session(self):
		# None is default
		self.II_nn.openSession()

	def close_II_NN_session(self):
		self.II_nn.closeSession()

# II_NN initial training (no disturbance)
	def train_II_NN(self, num_points=100000, learning_rate=0.001, \
		num_steps=100000, batch_size=128,\
		display_step=100):

		Cs = self.randgenCs(num_points)
		Qs = self.inverseBatch(Cs)
		Ys = np.array([self.forward(q)[0:2,3] for q in Qs])
		self.II_nn.train(Ys, Cs, learning_rate, num_steps, batch_size, display_step)

	def save_II_NN(self):
		self.II_nn.saveNN()

# IF_NN testing
	def test_II_NN(self, num_points=10):
		Cs = self.randgenCs(num_points)
		Qs = self.inverseBatch(Cs)
		Ys = np.array([self.forward(q)[0:2,3] for q in Qs])
		print("Test Cs:")
		print(Cs)
		loss = self.II_nn.test(Ys, Cs)
		print(loss)

## IF_NN forward run (single input only)
	def II_run(self, Y):
		C = self.II_nn.run([Y])
		return C[0]

	def initPIDcontrol(self):
		self.PID = {
			'lasterror': 0,
			'accumerror': 0,
		}
		self.setPID(1,0,0)
	
	def setPID(self, P, I, D):
		self.PID['P'] = P
		self.PID['I'] = I
		self.PID['D'] = D

	def resetPID_I(self):
		self.PID['accumerror'] = 0

	def resetPID(self):
		self.PID['lasterror'] = 0
		self.PID['accumerror'] = 0

	def calculatePIDcontrol(self, error):
		derror = error - self.PID['lasterror']
		s = self.PID['P']*error + self.PID['I']*self.PID['accumerror'] + self.PID['D']*derror
		self.PID['lasterror'] = error
		self.PID['accumerror'] = self.PID['accumerror'] + error
		if np.linalg.norm(self.PID['accumerror']) > 1000:
			self.resetPID_I()
		return s

## Reach with control, with disturbances, with learning support ##
## Learning means training the internal forward model
	def reach(self, X, learn=True, learning_rate=0.01, num_steps=10, display_step=100,\
		Q_disturbance=np.array([0,0]), Y_disturbance=np.array([0,0])):

		H_disturbance = np.zeros((4,4))
		H_disturbance[0,3]=Y_disturbance[0]
		H_disturbance[1,3]=Y_disturbance[1]

		C = self.II_run(X)
		E = self.IF_run(C)-X
		S = self.calculatePIDcontrol(E)
		Q = self.inverseSingle(C-S, disturbance=Q_disturbance)
		Y = self.forward(Q, disturbance=H_disturbance)[0:2,3]

		if learn:
			self.IF_nn.train([C], [Y], learning_rate, num_steps, 1, display_step)

		return Y



if __name__ == "__main__":
	pass