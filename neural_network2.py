#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 11:29:52 2019

@author: yourongzhang
"""

import tensorflow as tf 

class neural_network:
	def __init__(self,
				 n_input,
				 n_output,
		 		 n_hidden_units,
				 weight_initialization=tf.contrib.layers.xavier_initializer(), 
				 activation_hidden=tf.tanh):

		self.n_input = n_input
		self.n_output = n_output
		self.n_hidden_units = n_hidden_units
		self.weight_initialization = weight_initialization
		self.activation_hidden = activation_hidden

		self.weights = {}
		self.biases = {}
		self.number_of_layers = len(self.n_hidden_units)
		
		self.var_list = []

		# initialize the weights in the NNmodel
		for i in range(self.number_of_layers):
			if i == 0:
				self.weights['W_'+'0'] = tf.get_variable('weightW_' + str(0), shape=[self.n_input, self.n_hidden_units[0]], initializer=self.weight_initialization, dtype=tf.float64)
				self.var_list.append  ('weightW_' + str(0))
			else:
				self.weights['W_'+str(i)] = tf.get_variable('weightW_' + str(i), shape=[self.n_hidden_units[i-1], self.n_hidden_units[i], 4], initializer=self.weight_initialization, dtype=tf.float64)
				self.weights['U_'+str(i)] = tf.get_variable('weightU_' + str(i), shape=[self.n_input, self.n_hidden_units[i],4], initializer=self.weight_initialization, dtype=tf.float64)

				
			self.biases['bias' +str(i)] = tf.get_variable('bias_' + str(i), shape=[self.n_hidden_units[i],4], initializer=self.weight_initialization, dtype=tf.float64)

		self.weights['W_'+str(self.number_of_layers)] =  tf.get_variable('weightW_' + str(self.number_of_layers), shape=[self.n_hidden_units[-1], self.n_output], initializer=self.weight_initialization, dtype=tf.float64)
		self.biases['bias'+str(self.number_of_layers)] =tf.get_variable('bias_' + str(self.number_of_layers), shape=[self.n_output], initializer=self.weight_initialization, dtype=tf.float64)

		
		
	def value(self, input_var):
		for i in range(self.number_of_layers):
			# first layer
			if i == 0:	
				layer = tf.add(tf.matmul(input_var, self.weights['W_'+'0']) , self.biases['bias'+'0'][:,0])
				layer = self.activation_hidden(layer)
                
			else: 
				Z = tf.matmul(input_var, self.weights['U_'+str(i)][:,:,0])+ tf.matmul(layer, self.weights['W_'+str(i)][:,:,0]) + self.biases['bias'+str(i)][:,0]
				Z = self.activation_hidden(Z)
                
				G = tf.matmul(input_var, self.weights['U_'+str(i)][:,:,1])+ tf.matmul(layer, self.weights['W_'+str(i)][:,:,1]) + self.biases['bias'+str(i)][:,1]
				G = self.activation_hidden(G)
                
				R = tf.matmul(input_var, self.weights['U_'+str(i)][:,:,2])+ tf.matmul(layer, self.weights['W_'+str(i)][:,:,2]) + self.biases['bias'+str(i)][:,2]
				R = self.activation_hidden(R)
                
				intermediate = tf.multiply(layer,R)+ self.biases['bias'+str(i)][:,0]
				H = tf.matmul(input_var, self.weights['U_'+str(i)][:,:,3])+ tf.matmul(intermediate, self.weights['W_'+str(i)][:,:,0]) 
				H = self.activation_hidden(H)

				layer = tf.multiply (H, (1.0-G)) + tf.multiply(Z,layer)

		f = tf.matmul(layer, self.weights['W_'+str(self.number_of_layers)]) + self.biases['bias'+str(self.number_of_layers)]
		return f


	def first_derivatives(self, X):
		return tf.gradients(self.value(X), X)[0]

	def second_derivatives(self, X):
		grad = self.first_derivatives(X)
		grad_grad = []

		for i in range(self.n_input):
			grad_grad.append(tf.slice(tf.gradients(tf.slice(grad, [0, i], [tf.shape(X)[0], 1]), X)[0], [0, i],  [tf.shape(X)[0], 1]))

		return grad_grad