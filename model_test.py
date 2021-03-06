#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 14:16:02 2019

@author: yourongzhang
"""
import poisson_problem
import numpy as np
import tensorflow as tf
import neural_network2


def main(NUM_INPUTS,NUM_OUTPUTS,HIDDEN_UNITS, N, FILENAME):
	
	PROBLEM = poisson_problem.poisson_2d()
	neural_network = neural_network2.neural_network(NUM_INPUTS, NUM_OUTPUTS, HIDDEN_UNITS)
    
	int_var = tf.placeholder(tf.float64, [None, NUM_INPUTS]) 
	bou_var = tf.placeholder(tf.float64, [None, NUM_INPUTS]) 

	value_int = neural_network.value(int_var)
	value_bou = neural_network.value(bou_var)

	grad_grad= neural_network.second_derivatives(int_var)

	sol_int = tf.placeholder(tf.float64, [None, 1])
	sol_bou = tf.placeholder(tf.float64, [None, 1])

	sum_of_second_derivatives = 0.0
	for i in range(NUM_INPUTS):
		sum_of_second_derivatives += grad_grad[i]

	loss_int = tf.square(sum_of_second_derivatives + sol_int)
	loss_bou = tf.square(value_bou-sol_bou)

	loss = tf.reduce_mean(loss_int + loss_bou)
	saver = tf.train.Saver()

	init = tf.global_variables_initializer()

#	def compute_fd_error(session, problem, N):
#		x = np.linspace(problem.range[0], problem.range[1], N)
#		y = np.linspace(problem.range[0], problem.range[1], N)
#
#		mesh = np.array(np.meshgrid(x,y)).T.reshape(-1, 2)
#		fd_error = (session.run(value_int, feed_dict={int_var: mesh})- problem.velocity(mesh))**2
#
#		sum_fd_error = np.sum(fd_error)
#
#		return sum_fd_error/(N**2)


	def compute_fd_loss(session, problem, N):
		x = np.linspace(problem.range[0], problem.range[1], N)
		y = np.linspace(problem.range[0], problem.range[1], N)

		mesh = np.array(np.meshgrid(x,y)).T.reshape(-1, 2)
		sol = np.reshape(problem.rhs(mesh), ((N*N), 1))
		fd_loss = session.run(loss_int, feed_dict={int_var: mesh, sol_int: sol})

		sum_fd_loss = np.sum(fd_loss)

		return sum_fd_loss/(N**2)


	with tf.Session() as sess:
		sess.run(init)
		saver.restore(sess, FILENAME)
		print('Model restored.')

		#print('l2-loss:', np.sqrt(compute_fd_error(sess, PROBLEM, N)))
		print('l2-int-loss:', np.sqrt(compute_fd_loss(sess, PROBLEM, N)))

#		l_2_max, l_2_max_location = compute_fd_error_max(sess, PROBLEM, N)
#		l_2_int_max, l_2_int_max_location = compute_fd_loss_max(sess, PROBLEM, N)
#
#		print('l2-max:', np.sqrt(l_2_max), l_2_max_location)
#		print('l2-int-max:', np.sqrt(l_2_int_max), l_2_int_max_location)

if __name__ == '__main__':
	tf.reset_default_graph()
	main(2,1,[50], 100, 'mytest_model/1_layer_sq_loss_2000_m_iter_50000_rs_42')





