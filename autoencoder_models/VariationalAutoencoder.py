import os
import json
import numpy as np
import tensorflow as tf

class VariationalAutoencoder(object):

    def __init__(self, n_layers=[784, 10], learning_rate=0.001):
        self.n_layers = n_layers
        self.learning_rate = learning_rate

        self.transfer = tf.nn.softplus 
        self.graph = tf.Graph()
        with self.graph.as_default():
            optimizer = tf.train.AdamOptimizer(learning_rate)
            network_weights = self._initialize_weights()
            self.weights = network_weights

            # model
            self.x = tf.placeholder(tf.float32, [None, self.n_layers[0]])
            self.hidden_encode = []
            h = self.x
            for layer in range(len(self.n_layers)-2):
                h = self.transfer(
                    tf.add(tf.matmul(h, self.weights['encode'][layer]['w']),
                        self.weights['encode'][layer]['b']))
                self.hidden_encode.append(h)
                

            self.z_mean = tf.add(tf.matmul(h, self.weights['mean']['w']), self.weights['mean']['b'])
            self.z_log_sigma_sq = tf.add(tf.matmul(h, self.weights['log_sigma']['w']), self.weights['log_sigma']['b'])

            # sample from gaussian distribution
            eps = tf.random_normal(tf.stack([tf.shape(self.x)[0], self.n_layers[-1]]), 0, 1, dtype = tf.float32)
            self.z = tf.add(self.z_mean, tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))

            self.hidden_recon = []
            h = self.z
            for layer in range(len(self.n_layers)-1):
                h = self.transfer(
                    tf.add(tf.matmul(h, self.weights['recon'][layer]['w']),
                        self.weights['recon'][layer]['b']))
                self.hidden_recon.append(h)
            self.reconstruction = self.hidden_recon[-1]

            # cost
            # squared loss
            self.reconstr_loss = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
            # KL divergence
            self.latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq
                                            - tf.square(self.z_mean)
                                            - tf.exp(self.z_log_sigma_sq), 1)
            self.cost = tf.reduce_mean(self.reconstr_loss + self.latent_loss)
            self.optimizer = optimizer.minimize(self.cost)

            # create a saver 
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
        
        self.sess = tf.Session(graph=self.graph)
        self.sess.run(init)

    def _initialize_weights(self):
        all_weights = dict()
        initializer = tf.contrib.layers.xavier_initializer()
        # Encoding network weights
        encoder_weights = []
        for layer in range(len(self.n_layers)-2):
            w = tf.Variable(
                initializer((self.n_layers[layer], self.n_layers[layer + 1]),
                            dtype=tf.float32))
            b = tf.Variable(
                tf.zeros([self.n_layers[layer + 1]], dtype=tf.float32))
            encoder_weights.append({'w': w, 'b': b})
        # Bottleneck layer weights
        w = tf.Variable(
            initializer((self.n_layers[-2], self.n_layers[-1]),
                        dtype=tf.float32))
        b = tf.Variable(
            tf.zeros([self.n_layers[-1]], dtype=tf.float32))
        all_weights['mean'] = {'w': w, 'b': b}
        w = tf.Variable(
            initializer((self.n_layers[-2], self.n_layers[-1]),
                        dtype=tf.float32))
        b = tf.Variable(
            tf.zeros([self.n_layers[-1]], dtype=tf.float32))
        all_weights['log_sigma'] = {'w': w, 'b': b}

        # Recon network weights
        recon_weights = []
        for layer in range(len(self.n_layers)-1, 0, -1):
            w = tf.Variable(
                initializer((self.n_layers[layer], self.n_layers[layer - 1]),
                            dtype=tf.float32))
            b = tf.Variable(
                tf.zeros([self.n_layers[layer - 1]], dtype=tf.float32))
            recon_weights.append({'w': w, 'b': b})
        all_weights['encode'] = encoder_weights
        all_weights['recon'] = recon_weights
        return all_weights

    def partial_fit(self, X):
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict={self.x: X})
        return cost

    def calc_total_cost(self, X):
        return self.sess.run(self.cost, feed_dict = {self.x: X})

    def calc_losses(self, X):
        reconstr_loss, latent_losses = self.sess.run([self.reconstr_loss, self.latent_loss], feed_dict={self.x: X})
        return reconstr_loss, np.mean(latent_losses)

    def transform(self, X):
        return self.sess.run(self.z_mean, feed_dict={self.x: X})

    def generate(self, hidden = None):
        if hidden is None:
            hidden = np.random.randn(1, self.n_layers[-1])
        return self.sess.run(self.reconstruction, feed_dict={self.z: hidden})

    def reconstruct(self, X):
        return self.sess.run(self.reconstruction, feed_dict={self.x: X})

    def save(self, path):
        '''To save trained model and its params.
        '''
        save_path = self.saver.save(self.sess, 
            os.path.join(path, 'model.ckpt'))
        # save parameters of the model
        params = {'n_layers': self.n_layers, 'learning_rate': self.learning_rate}
        json.dump(params, 
            open(os.path.join(path, 'model_params.json'), 'w'))
        return save_path

    def _restore(self, path):
        with self.graph.as_default():
            self.saver.restore(self.sess, path)

    @classmethod
    def load(cls, path):
        '''To restore a saved model.
        '''
        # load params of the model
        params = json.load(open(os.path.join(path, 'model_params.json'), 'r'))
        # init an instance of this class
        estimator = cls(**params)
        estimator._restore(os.path.join(path, 'model.ckpt'))
        return estimator
