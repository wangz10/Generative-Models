from __future__ import print_function, division

import os
import json
import numpy as np
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers import MaxPooling2D, concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import losses
from keras.utils import to_categorical
import keras.backend as K
from keras.models import load_model

class BiGAN():
    def __init__(self, g_n_layers=[784, 10], d_n_layers=[100, 10], learning_rate=0.001, build_model=True):
        '''
        BiGAN: Bidirectional Generative Adversarial Network
        g_n_layers(list): number of neurons for generator network, 
            the reverse is for the encoder network, the first element should be 
            the input dim, last element should be the latent dim.
        d_n_layers(list): number of hidden units, the first element is the first hidden layer, 
            the input dim will be g_n_layers[0] + g_n_layers[-1].
        '''
        self.g_n_layers = g_n_layers
        self.d_n_layers = d_n_layers
        self.input_shape = g_n_layers[0]
        self.latent_dim = g_n_layers[-1]
        self.learning_rate = learning_rate
        self.params = {
            'g_n_layers': g_n_layers,
            'd_n_layers': d_n_layers,
            'learning_rate': learning_rate
        }
        if build_model:
            self.build_gan()

    def build_gan(self):
        optimizer = Adam(self.learning_rate, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=['binary_crossentropy'],
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # Build the encoder
        self.encoder = self.build_encoder()

        # The part of the bigan that trains the discriminator and encoder
        self.discriminator.trainable = False

        # Generate image from sampled noise
        z = Input(shape=(self.latent_dim, ))
        img_ = self.generator(z)

        # Encode image
        img = Input(shape=(self.input_shape, ))
        z_ = self.encoder(img)

        # Latent -> img is fake, and img -> latent is valid
        fake = self.discriminator([z, img_])
        valid = self.discriminator([z_, img])

        # Set up and compile the combined model
        # Trains generator to fool the discriminator
        self.bigan_generator = Model([z, img], [fake, valid])
        self.bigan_generator.compile(loss=['binary_crossentropy', 'binary_crossentropy'],
            optimizer=optimizer)


    def build_encoder(self):
        '''Encoder model encodes input to latent dim: E(x) = z.'''
        model = Sequential()

        for i, n_layer in enumerate(self.g_n_layers[1:]):
            if i == 0:
                model.add(Dense(n_layer, input_dim=self.input_shape))
            else:
                model.add(Dense(n_layer))
            model.add(LeakyReLU(alpha=0.2))
            model.add(BatchNormalization(momentum=0.8))
        
        model.summary()

        img = Input(shape=(self.input_shape, ))
        z = model(img)

        return Model(img, z)

    def build_generator(self):
        model = Sequential()
        for i, n_layer in enumerate(self.g_n_layers[::-1][1:]):
            if i == 0:
                model.add(Dense(n_layer, input_dim=self.latent_dim))
                model.add(LeakyReLU(alpha=0.2))
                model.add(BatchNormalization(momentum=0.8))
            elif i == len(self.g_n_layers) - 2: # last layer
                model.add(Dense(n_layer, activation='tanh'))
            else:
                model.add(Dense(n_layer)) 
                model.add(LeakyReLU(alpha=0.2))
                model.add(BatchNormalization(momentum=0.8))

        model.summary()

        z = Input(shape=(self.latent_dim,))
        gen_img = model(z)

        return Model(z, gen_img)

    def build_discriminator(self):

        z = Input(shape=(self.latent_dim, ))
        img = Input(shape=(self.input_shape, ))
        d_in = concatenate([z, img])

        for i, n_layer in enumerate(self.d_n_layers):
            if i == 0:
                model = Dense(n_layer)(d_in)
                model = LeakyReLU(alpha=0.2)(model)
                model = Dropout(0.5)(model)

            else:
                model = Dense(n_layer)(model)        
                model = LeakyReLU(alpha=0.2)(model)
                model = Dropout(0.5)(model)
        
        validity = Dense(1, activation="sigmoid")(model)

        return Model([z, img], validity)

    def partial_fit(self, x_batch):
        '''Train G, E, D using a batch of data.'''
        # Adversarial ground truths
        batch_size = x_batch.shape[0]
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Sample noise and generate img
        z = np.random.normal(size=(batch_size, self.latent_dim))
        x_batch_gen = self.generator.predict(z)

        # Select a random batch of images and encode
        z_ = self.encoder.predict(x_batch)

        # Train the discriminator (x_batch -> z_ is valid, z -> x_batch_gen is fake)
        d_loss_real = self.discriminator.train_on_batch([z_, x_batch], valid)
        d_loss_fake = self.discriminator.train_on_batch([z, x_batch_gen], fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        # ---------------------
        #  Train Generator
        # ---------------------

        # Train the generator (z -> x_batch is valid and x_batch -> z is is invalid)
        g_loss = self.bigan_generator.train_on_batch([z, x_batch], [valid, fake])
        
        accuracy = d_loss[1]
        # scalers for loss and accuracy from the discriminator
        return d_loss[0], accuracy, g_loss[0]

    def transform(self, X):
        '''Run encoder to get the latent embedding: z = E(x).'''
        return self.encoder.predict(X)

    def generate(self, z = None):
        if z is None:
            z = np.random.normal(size=[1, self.latent_dim])
        return self.generator.predict(z)

    def save(self, path):
        '''Save trained models to files'''
        self.generator.save(os.path.join(path, "generator.h5"))
        self.discriminator.save(os.path.join(path, "discriminator.h5"))
        self.encoder.save(os.path.join(path, "encoder.h5"))
        self.bigan_generator.save(os.path.join(path, "bigan_generator.h5"))
        json.dump( self.params, open(os.path.join(path, 'params.json'), 'w') ) 

    @classmethod
    def load(cls, path):
        params = json.load(open(os.path.join(path, 'params.json'), 'r'))
        params['build_model'] = False
        gan = cls(**params)
        gan.generator = load_model(os.path.join(path, "generator.h5"))
        gan.discriminator = load_model(os.path.join(path, "discriminator.h5"))
        gan.encoder = load_model(os.path.join(path, "encoder.h5"))
        gan.bigan_generator = load_model(os.path.join(path, "bigan_generator.h5"))
        return gan

