import tensorflow as tf 
tf.set_random_seed(42)

import numpy as np 
from scipy import integrate
import neural_network2 as neural_networks

class sampling_from_dataset:
	def __init__(self, filepath, total_samples):
		self.filepath = filepath
		self.total_samples = total_samples

		self.last_grab_int = 0
		self.last_grab_bou = 0

	def load_dataset(self):
		self.dataset = np.genfromtxt(self.filepath, delimiter=',')


	def interior_samples(self, batchsize):
		sampling_int_draw = self.dataset[0:batchsize ,:]


		return sampling_int_draw
    
	def boundary_samples(self, batchsize):
		sampling_bou_draw = self.dataset[batchsize:2*batchsize ,:]

		return sampling_bou_draw


def main():

	# DEFAULT
	#N_LAYERS = 2
	BATCHSIZE = 2000
	MAX_ITER = 50000
	DO_SAVE = True
	SEED = 42
	NUM_INPUTS = 2
	minibatch_size = 100
	HIDDEN_UNITS = [50]
	#for i in range(N_LAYERS):
	#	HIDDEN_UNITS.append(16)

	#problem = poisson_problem.poisson(N_dim)
	
	sampler = sampling_from_dataset('datasets/' + str(BATCHSIZE) + '_' + str(NUM_INPUTS), BATCHSIZE)
	sampler.load_dataset()


	neural_network = neural_networks.neural_network(NUM_INPUTS, 1, HIDDEN_UNITS)

    #setting up input from interior and boundry of domain
	int_var = tf.placeholder(tf.float64, [None, NUM_INPUTS]) 
	bou_var = tf.placeholder(tf.float64, [None, NUM_INPUTS]) 
	
	# neural_network predction of output 
	value_int = neural_network.value(int_var)
	value_bou = neural_network.value(bou_var)
	
	#pde solution
	sol_int = tf.placeholder(tf.float64, [None, 1])
	sol_bou = tf.placeholder(tf.float64, [None, 1])

	#loss calculation
	grad_grad= neural_network.second_derivatives(int_var)
	sum_of_second_derivatives = 0.0
	for i in range(NUM_INPUTS):
		sum_of_second_derivatives += grad_grad[i]


	loss_int = tf.square(sum_of_second_derivatives+sol_int)
	loss_bou = tf.square(value_bou-sol_bou)
	loss = tf.sqrt(tf.reduce_mean(loss_int + loss_bou))
	
	   
#	optimizer = tf.train.GradientDescentOptimizer(0.001)
#	train = optimizer.minimize(loss)
	# train the model
	train_scipy = tf.contrib.opt.ScipyOptimizerInterface(loss, method='BFGS', options={'gtol':1e-14, 'disp':True, 'maxiter':MAX_ITER})

	saver = tf.train.Saver()
	save_name = 'mytest_model/' + str(len(HIDDEN_UNITS)) + '_layer_sq_loss_' + str(BATCHSIZE) + '_m_iter_' + str(MAX_ITER) + '_rs_' + str(SEED) 


	init = tf.global_variables_initializer()    
	with tf.Session() as sess:
		sess.run(init)
		
		# getting sample data
		int_draw = sampler.interior_samples(BATCHSIZE)
		int_draw = np.array(int_draw) 
		print (np.shape(int_draw))
        
		bou_draw = sampler.boundary_samples(BATCHSIZE)
		bou_draw = np.array(bou_draw) 
		print (np.shape(bou_draw))
        
		# calculation of solution of the pde
		x = int_draw
		f = 2.0*np.pi**2 * np.sin(np.pi*x[:,0]) * np.sin(np.pi*x[:,1]) 
		f = np.reshape(np.array(f), (BATCHSIZE, 1))

		x = bou_draw
		bou = np.sin(np.pi*x[:,0]) * np.sin(np.pi*x[:,1])        
		bou = np.reshape(np.array(bou), (BATCHSIZE, 1))

		# feed in to graph
		train_scipy.minimize(sess, feed_dict={sol_int:f, sol_bou:bou, int_var:int_draw, bou_var:bou_draw})
# =============================================================================
# 		for epoch in range(MAX_ITER):
# 			out_c = 0.0            
# 			for i in range (int(BATCHSIZE/minibatch_size)):
# 				a = f[i*minibatch_size:(i+1)*minibatch_size] 
# 				b = bou[i*minibatch_size:(i+1)*minibatch_size]     
# 				c = int_draw[i*minibatch_size:(i+1)*minibatch_size]     
# 				d = bou_draw[i*minibatch_size:(i+1)*minibatch_size]               				
# 				_,c = sess.run ([train,loss],feed_dict={sol_int:a,sol_bou:b, int_var:c, bou_var:d})
# 				out_c += c			
#             #print (c)
# =============================================================================
		if DO_SAVE:
			save_path = saver.save(sess, save_name)
			print("Model saved in path: %s" % save_path)
		#print (out_c)


if __name__ == '__main__':
    tf.reset_default_graph()
    main()
