"""
Feed forward neural network instantiations
"""

from __future__ import print_function
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random

class FFNN_Regression():
	def __init__(self, hidden_layers, num_inputs, num_outputs):
		if len(hidden_layers) < 1:
			print ('No hidden layers???')

		self.hidden_layers = hidden_layers
		self.num_inputs = num_inputs
		self.num_outputs = num_outputs

		self.init_ioplaceholders()
		self.init_weights()
		self.init_biases()
		self.construct_model()

	def init_ioplaceholders(self):
		self.X = tf.placeholder("float", [None, self.num_inputs])
		self.Y = tf.placeholder("float", [None, self.num_outputs])
		print('got here')

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
		self.layers.append(tf.add(tf.matmul(self.X, self.weights['1']), self.biases['1']))
		for i in range(len(self.hidden_layers)-1):
			self.layers.append(tf.add(tf.matmul(self.layers[-1], \
				self.weights[str(i+2)]), self.biases[str(i+2)]))
		self.layers.append(tf.add(tf.matmul(self.layers[-1],\
			self.weights['out']), self.biases['out']))

		self.output = self.layers[-1]


	def init_sess(self):
		self.gv_init = tf.global_variables_initializer()
		self.saver = tf.train.Saver()


# For training
	def init_optimizer(self, learning_rate=0.001):
		self.loss_op = tf.reduce_mean(tf.square(self.Y - self.output))
		self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss_op)

	def train(self, dataX, dataY, num_steps=5000, batch_size=128, display_step=100, model=None):
		# Initialize the variables (i.e. assign their default value)
		data_inds = [i for i in range(len(dataX))]

		self.init_sess()

		# Start training
		with tf.Session() as sess:
			# Run the initializer
			if model == None:
				sess.run(self.gv_init)
			else:
				self.saver.restore(sess, model)
				print('Model %s restored' % model)

			batch_start = 0
			batch_end = batch_size

			for step in range(1, num_steps+1):
				batch_inds = data_inds[batch_start:batch_end]
				batch_x = [dataX[i] for i in batch_inds]
				batch_y = [dataY[i] for i in batch_inds]

				# Increment next batch
				if batch_end < num_training_data:
					batch_start += batch_size
					batch_end += batch_size
					batch_end = min(batch_end,num_training_data)
				else:
					# If reached the end of the batch, reshuffle and start from beginning
					random.shuffle(data_inds)
					batch_start = 0
					batch_end = batch_size

				# Run optimization op (backprop)
				loss, _ = sess.run([self.loss_op, self.optimizer], feed_dict={X: batch_x, Y:batch_x})
				if step % display_step == 0 or step == 1:
					# Calculate batch loss and accuracy
					print("Step " + str(step) + ", Minibatch Loss= " + "{:.4f}".format(loss))

			self.save_path = self.saver.save(sess, model)
			print("Model saved in file: %s" % self.save_path)


# For testing
	def test(self, dataX, dataY, model=None):
		self.init_sess

		with tf.Session() as sess:
			self.saver.restore(sess, model)
			print('Model %s restored' % model)

			# Testing
			for i in range(len(dataX)):
				nn_output, loss = sess.run([self.output, self.loss_op],\
					feed_dict={X: [dataX[i]], Y: [dataY[i]]})
				print(nn_output)
				print(dataY[i])
				#print(loss)

# For forward runs
	def openSession(self, model=None):
		self.init_sess()
		self.sess = tf.Session()
		self.saver.restore(self.sess, model)

	def closeSession(self):
		self.sess.close()

	def run(self, dataX):
		nn_output = self.sess.run(self.output, feed_dict={X: dataX})
		return nn_output