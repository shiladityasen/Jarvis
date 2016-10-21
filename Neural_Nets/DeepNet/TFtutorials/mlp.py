import tensorflow as tf
import numpy as np

# Neural network class
class neural_net:
    def __init__(self, layers=[784, 10], activations=[tf.nn.softmax], error=tf.contrib.losses.log_loss):
        self.n_layers = len(layers)
        self.n_inputs = layers[0]
        self.n_outputs = layers[-1]
        
        self.x = tf.placeholder('float', [None, self.n_inputs], name='x')
        self.y = tf.placeholder('float', [None, self.n_outputs], name='y')
        
        self.W = []
        self.B = []
        self.neural_layers = [self.x]
        self.trained_W = None
        self.trained_B = None
                
        for layer in xrange(self.n_layers-1):
            input_layer = self.neural_layers[-1]
            
            w = tf.Variable(tf.random_normal([layers[layer], layers[layer+1]], name='rng_W_'+str(layer)), 
                            name='W_'+str(layer))
            b = tf.Variable(tf.random_normal([layers[layer+1]], name='rng_B_'+str(layer)), name='B_'+str(layer))
            
            output_layer = tf.add(tf.matmul(input_layer, w, name='W_'+str(layer)+'times_x'), 
                                  b, name='W_'+str(layer)+'_times_x_plus_B_'+str(layer))
            if activations[layer] != None:
                output_layer = activations[layer](output_layer)
            self.neural_layers += [output_layer]
            
            self.W += [w]
            self.B += [b]
        
        self.error = tf.reduce_mean(error(self.neural_layers[-1], self.y))
        
    def train(self, train_x, train_y, step_size=0.01, batch_size=100, max_iters=1):
        train_step = tf.train.GradientDescentOptimizer(step_size, name='grad_desc').minimize(self.error)
        
        train_size, n_features = train_x.shape
        train_x = np.vsplit(train_x, train_size/batch_size)
        train_y = np.vsplit(train_y, train_size/batch_size)
                
        with tf.Session() as session:
            tf.train.SummaryWriter('mlp_logs', session.graph)
            session.run(tf.initialize_all_variables())
            
            for _ in xrange(max_iters):
                for batch_x, batch_y in zip(train_x, train_y):
                    session.run(train_step, feed_dict={self.x: batch_x, self.y: batch_y})
                                
            W = session.run(self.W)
            B = session.run(self.B)
        
        self.trained_W = W
        self.trained_B = B
        return (W, B)
        
    def predict(self, test_x):
        with tf.Session() as session:
            for i in xrange(self.n_layers-1):
                if self.trained_W == None and self.trained_B == None:
                    session.run(tf.initialize_all_variables())
                else:
                    session.run(self.W[i].assign(self.trained_W[i]))
                    session.run(self.B[i].assign(self.trained_B[i]))
            return session.run(self.neural_layers[-1], feed_dict={self.x: test_x})
        
    def unlearn(self):
        self.trained_W = None
        self.trained_B = None
        with tf.Session() as session:
            session.run(tf.initialize_all_variables)