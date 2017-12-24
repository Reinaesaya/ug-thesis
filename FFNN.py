"""
Feed forward neural network instantiations
"""

from __future__ import print_function
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import os

class FFNN_1():
	"""
	Feed forward neural network instantiation with
	tanh activation each layer except for output layer
	"""
	def __init__(self, hidden_layers, num_inputs, num_outputs, model=None):
		if len(hidden_layers) < 1:
			print ('No hidden layers???')

		self.hidden_layers = hidden_layers
		self.num_inputs = num_inputs
		self.num_outputs = num_outputs
		self.model = model
		self.sessopen = False

		self.init_ioplaceholders()
		self.init_weights()
		self.init_biases()
		self.construct_model()
		self.init_optimizer()

# Model constructions
	def init_ioplaceholders(self):
		self.X = tf.placeholder("float", [None, self.num_inputs])
		self.Y = tf.placeholder("float", [None, self.num_outputs])

	def init_weights(self):
		self.weights = {}
		self.weights['1'] = tf.Variable(tf.random_normal([self.num_inputs, self.hidden_layers[0]]))
		for i in range(len(self.hidden_layers)-1):
			self.weights[str(i+2)] = tf.Variable(\
				tf.random_normal([self.hidden_layers[i], self.hidden_layers[i+1]]))
		self.weights['out'] = tf.Variable(tf.random_normal([self.hidden_layers[-1], self.num_outputs]))

	def init_biases(self):
		self.biases = {}
		for i in range(len(self.hidden_layers)):
			self.biases[str(i+1)] = tf.Variable(tf.random_normal([self.hidden_layers[i]]))
		self.biases['out'] = tf.Variable(tf.random_normal([self.num_outputs]))

	def construct_model(self):
		self.layers = []
		self.layers.append(tf.tanh(tf.add(tf.matmul(self.X, self.weights['1']), self.biases['1'])))
		for i in range(len(self.hidden_layers)-1):
			self.layers.append(tf.tanh(tf.add(tf.matmul(self.layers[-1], \
				self.weights[str(i+2)]), self.biases[str(i+2)])))
		self.layers.append(tf.add(tf.matmul(self.layers[-1],\
			self.weights['out']), self.biases['out']))

		self.output = self.layers[-1]
		self.loss_op = tf.reduce_mean(tf.square(self.Y - self.output))

	def init_optimizer(self):
		self.learning_rate_placeholder = tf.placeholder(tf.float32, [], name='learning_rate')
		self.optimizer = tf.train.AdamOptimizer(\
			learning_rate=self.learning_rate_placeholder).minimize(self.loss_op)

# Session open and close
	def openSession(self):
		if self.sessopen:
			print('Session already open')
		else:
			self.gv_init = tf.global_variables_initializer()
			self.saver = tf.train.Saver()
			self.sess = tf.Session()

			if (self.model == None) or not os.path.exists(self.model+'.index'):
				self.sess.run(self.gv_init)
			else:
				self.saver.restore(self.sess, self.model)
				print('Model %s restored' % self.model)
			self.sessopen = True
			print('Session opened')

	def closeSession(self):
		if self.sessopen:
			self.sess.close()
			self.sessopen = False
			print('Session closed')
		else:
			print('Session already closed')

	def saveNN(self):
		self.save_path = self.saver.save(self.sess, self.model)
		print("Model saved in file: %s" % self.save_path)

# For training

	def train(self, dataX, dataY, learning_rate=0.001,\
		num_steps=100000, batch_size=128, display_step=100):
		# Initialize the variables (i.e. assign their default value)
		data_inds = [i for i in range(len(dataX))]

		# Start training
		batch_start = 0
		batch_end = batch_size

		for step in range(1, num_steps+1):
			batch_inds = data_inds[batch_start:batch_end]
			batch_x = [dataX[i] for i in batch_inds]
			batch_y = [dataY[i] for i in batch_inds]

			# Increment next batch
			if batch_end < len(dataX):
				batch_start += batch_size
				batch_end += batch_size
				batch_end = min(batch_end,len(dataX))
			else:
				# If reached the end of the batch, reshuffle and start from beginning
				random.shuffle(data_inds)
				batch_start = 0
				batch_end = batch_size

			# Run optimization op (backprop)
			loss, _ = self.sess.run([self.loss_op, self.optimizer],\
				feed_dict={self.X: batch_x, self.Y: batch_y,\
				self.learning_rate_placeholder: learning_rate})
			if step % display_step == 0:
				# Calculate batch loss and accuracy
				print("Step " + str(step) + ", Minibatch Loss= " + "{:.4f}".format(loss))

# For testing
	def test(self, dataX, dataY):
		loss = 0

		# Testing
		nn_output, loss = self.sess.run([self.output, self.loss_op],\
			feed_dict={self.X: dataX, self.Y: dataY})
		#print(dataY)
		#print(nn_output)
		return loss

# For forward runs
	def run(self, dataX):
		nn_output = self.sess.run(self.output, feed_dict={self.X: dataX})
		return nn_output