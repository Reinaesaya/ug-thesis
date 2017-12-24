import copy
import numpy as np
from numpy import cos, sin

import random

from FFNN import FFNN_1

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
	def __init__(self, link1len, link2len, elbow_dir, invmodel, compmodel):
		self.elbow_dir = elbow_dir # left or right
		self.invmodel = invmodel
		self.compmodel = compmodel

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
		Qs = []
		for n in range(numQs):
			q1 = random.random()*self.links[0].c_range + self.links[0].c_span[0]
			q2 = random.random()*self.links[1].c_range + self.links[1].c_span[0]
			Qs.append(np.array([q1,q2]))
		return Qs

	def randgenXs(self, numXs):
		Xs = []
		for n in range(numXs):
			q1 = random.random()*self.links[0].c_range + self.links[0].c_span[0]
			q2 = random.random()*self.links[1].c_range + self.links[1].c_span[0]
			Xs.append(self.forward([q1,q2])[0:2,3])
		return Xs


#### INVERSE MODEL NEURAL NETWORK (BRAIN STEM) ###
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

		Qs = self.randgenQs(num_points)
		Xs = np.array([self.forward(q)[0:2,3] for q in Qs])
		self.inv_ffnn.train(Xs, Qs, learning_rate, num_steps, batch_size, display_step)

	def save_inverseNN(self):
		self.inv_ffnn.saveNN()

# NN testing
	def test_inverseNN(self, num_points=10):
		Qs = self.randgenQs(num_points)
		Xs = np.array([self.forward(q)[0:2,3] for q in Qs])
		print("Test Xs:")
		print(Xs)
		loss = self.inv_ffnn.test(Xs, Qs)
		print(loss)

## NN inverse model
	def inverse(self, Xs):
		return self.inv_ffnn.run(Xs)

	def inverseSingle(self, X):
		Q = self.inverse([X])
		return Q[0]

	def inverseBatch(self, Xs):
		nnQs = self.inverse(Xs)
		return nnQs


### COMPENSATION NEURAL NETWORK (CEREBELLUM) ###
# NN session open and closes
	def init_compNN(self):
		self.comp_ffnn = FFNN_1(hidden_layers=[128], num_inputs=2,\
			num_outputs=2, model=self.compmodel)

	def open_compNN_session(self):
		# None is default
		self.comp_ffnn.openSession()

	def close_compNN_session(self):
		self.comp_ffnn.closeSession()

# compensator NN initial training
	def train_compNN(self, num_points=100000, learning_rate=0.001, \
		num_steps=100000, batch_size=128,\
		display_step=100):

		Xs = self.randgenXs(num_points)
		Qs = self.inverseBatch(Xs)
		Ys = np.array([self.forward(q)[0:2,3] for q in Qs])
		Ds = Ys-Xs
		self.comp_ffnn.train(Qs, Ds, learning_rate, num_steps, batch_size, display_step)

	def save_compNN(self):
		self.comp_ffnn.saveNN()

# compensator NN testing
	def test_inverseNN(self, num_points=10):
		Xs = self.randgenXs(num_points)
		Qs = self.inverseBatch(Xs)
		Ys = np.array([self.forward(q)[0:2,3] for q in Qs])
		Ds = Ys-Xs
		print("Test Qs:")
		print(Qs)
		loss = self.comp_ffnn.test(Qs, Ds)
		print(loss)

## compensator NN forward run (single input only)
	def compensate(self, Q):
		D = self.comp_ffnn.run([Q])
		return D[0]


### Full loop for reaching, with learning ###
	def reach(self, X, learn=True, learning_rate=0.01, num_steps=10, display_step=100):
		Q = self.inverseSingle(X)
		D = self.compensate(Q)
		Q_comp = self.inverseSingle(X-D)
		Y = self.forward(Q_comp)[0:2,3]
		D_comp = Y-X

		if learn:
			self.comp_ffnn.train([Q], [D_comp], learning_rate, num_steps, 1, display_step)

		return Y


if __name__ == "__main__":
	myRR = myRRrobot(50,50,'right',\
		'./models/invRR_right/invkinRR_right.ckpt',\
		'./models/compRR_right/compensationRR_right.ckpt')
	myRR.init_inverseNN()
	myRR.open_inverseNN_session()
	#myRR.train_inverseNN(display_step=1000)
	#myRR.save_inverseNN()
	#myRR.test_inverseNN(10)

	myRR.init_compNN()
	myRR.open_compNN_session()
	#myRR.train_compNN(display_step=1000)
	#myRR.save_compNN()

	x = [30,60]
	noloop = myRR.forward(myRR.inverseSingle(x))[0:2,3]
	withloop = myRR.reach(x, learn=False)

	print(noloop)
	print(withloop)

	# learn a few times
	for i in range(1000):
		withloop = myRR.reach(x, learn=True, learning_rate=0.005)
		if i%100 == 0:
			print('Reach %d: %s' % (i, str(withloop)))

	print(myRR.reach(x, learn=False))

	# q = myRR.inverseSingle([10,80])
	# print(q)
	# y = myRR.forward(q)[0:2,3]

	# print(y)
	# qs = myRR.inverseBatch([[1,1],[2,2]])
	# print(qs)

	myRR.save_inverseNN()
	myRR.save_compNN()
	myRR.close_compNN_session()
	myRR.close_inverseNN_session()