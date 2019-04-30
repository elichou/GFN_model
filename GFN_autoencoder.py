#!/usr/bin/env python
"""
Gated Field Autoencoder model using keras and tf.

@author: Elias Aoun Durand
@version: 1.0

"""
from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow.keras import layers



class GFN_encoder(tf.keras.Model):

    def __init__(self, input_type, input_shape, latent_dim, factor_type = ['dense']):
        super(GFN_encoder, self).__init__(name ='GFN_encoder')

        self.input_type = input_type
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.factor_type = factor_type

        self.inputlayer = layers.Input(shape = self.input_shape, name = 'enc_in_img_pos')

        if (('posture' in self.input_type ) and ('image' in self.input_type)):
            self.xlayer = layers.Lambda(lambda x: x[:,0,:,:])
            self.ylayer = layers.Lambda(lambda x: x[:,1,:3,0])

        if (('posture' in self.input_type) and ( not ('image' in self.input_type))):
            self.xlayer = layers.Lambda(lambda x: x[:,0,:])
            self.ylayer = layers.Lambda(lambda x: x[:,1,:])

        else:
            self.xlayer = layers.Lambda(lambda x: x[:,0,:,:])
            self.ylayer = layers.Lambda(lambda x: x[:,1,:,:])

        self.xflatten = layers.Flatten()
        self.yflatten = layers.Flatten()

        if ('dense' in self.factor_type):
            self.xlayer_1 = layers.Dense(self.latent_dim, activation='relu')
            self.xlayer_2 = layers.Dense(self.latent_dim, activation='relu')
            self.xlayer_3 = layers.Dense(self.latent_dim, activation='relu')
            self.xreshape = layers.Reshape((1,self.latent_dim,))

            self.ylayer_1 = layers.Dense(self.latent_dim, activation='relu')
            self.ylayer_2 = layers.Dense(self.latent_dim, activation='relu')
            self.ylayer_3 = layers.Dense(self.latent_dim, activation='relu')
            self.yreshape = layers.Reshape((self.latent_dim,1,))
        else:
            self.xlayer_1 = layers.Dense(self.latent_dim, activation='relu')
            self.xlayer_2 = layers.Dense(self.latent_dim, activation='relu')
            self.xlayer_3 = layers.Dense(self.latent_dim, activation='relu')
            self.xreshape = layers.Reshape((1,self.latent_dim,))

            self.ylayer_1 = layers.Dense(self.latent_dim, activation='relu')
            self.ylayer_2 = layers.Dense(self.latent_dim, activation='relu')
            self.ylayer_3 = layers.Dense(self.latent_dim, activation='relu')
            self.yreshape = layers.Reshape((self.latent_dim,1,))

        self.matmul = layers.Multiply()

        self.hflatten = layers.Flatten()
        self.hlayer_1 = layers.Dense(self.latent_dim, activation='relu')
        self.hlayer_2 = layers.Dense(self.latent_dim, activation='relu')
        self.hlayer_3 = layers.Dense(self.latent_dim, activation='relu')
        self.hreshape = layers.Reshape((1, self.latent_dim,))

        self.concatenate = layers.Concatenate()

    def call(self, inputs):
        # forward pass
        input = self.inputlayer(inputs)

        x = self.xlayer(input)
        y = self.ylayer(input)

        fx = self.xflatten(x)
        fx = self.xlayer_1(fx)
        fx = self.xlayer_2(fx)
        fx = self.xlayer_3(fx)
        fx = self.xreshape(fx)

        fy = self.yflatten(y)
        fy = self.ylayer_1(fy)
        fy = self.ylayer_2(fy)
        fy = self.ylayer_3(fy)
        fy = self.yreshape(fy)

        matmul = self.matmul([fy, fx])

        fh = self.hflatten(matmul)
        fh = self.hlayer_1(fh)
        fh = self.hlayer_2(fh)
        fh = self.hlayer_3(fh)
        fh = self.hreshape(fh)

        return self.concatenate([fx, fh])

class GFN_decoder(tf.keras.Model):

    def __init__(self, input_type, input_shape, latent_dim, factor_type = ['dense']):
        super(GFN_decoder, self).__init__(name='GFN_decoder')

        self.input_type = input_type
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.factor_type = factor_type

        self.inputlayer = layers.Input(shape = self.input_shape, name = 'enc_in_img_pos')

        self.xlayer = layers.Lambda(lambda x: x[:,:,:self.latent_dim])
        self.hlayer = layers.Lambda(lambda x: x[:,:,self.latent_dim:])

        self.hflatten = layers.Flatten()
        self.xflatten = layers.Flatten()

        self.hlayer_1 = layers.Dense(self.latent_dim, activation='relu')
        self.hlayer_2 = layers.Dense(self.latent_dim, activation='relu')
        self.hlayer_3 = layers.Dense(self.latent_dim, activation='relu')
        self.hreshape = layers.Reshape((1, self.latent_dim,))

        self.xlayer_1 = layers.Dense(self.latent_dim, activation='relu')
        self.xlayer_2 = layers.Dense(self.latent_dim, activation='relu')
        self.xlayer_3 = layers.Dense(self.latent_dim, activation='relu')
        self.xreshape = layers.Reshape((self.latent_dim,1,))

        self.matmul = layers.Multiply()

        if ('dense' in self.factor_type):
            self.ylayer_1 = layers.Dense(self.latent_dim, activation='relu')
            self.ylayer_2 = layers.Dense(self.latent_dim, activation='relu')
            self.ylayer_3 = layers.Dense(self.latent_dim, activation='relu')

        else:
            self.ylayer_1 = layers.Dense(self.latent_dim, activation='relu')
            self.ylayer_2 = layers.Dense(self.latent_dim, activation='relu')
            self.ylayer_3 = layers.Dense(self.latent_dim, activation='relu')

        if (('posture' in self.input_type) and ( not ('image' in self.input_type))):
            self.yrecon = layers.Reshape((1, self.input_shape[1],))

        else:
            self.yrecon = layers.Reshape((1, self.input_shape[1]*self.input_shape[1],))

    def call(self, inputs):

        input = self.input(inputs)

        fx = self.xlayer(input)
        fh = self.ylayer(input)

        fx = self.xflatten(fx)
        fx = self.xlayer_1(fx)
        fx = self.xlayer_2(fx)
        fx = self.xlayer_3(fx)
        fx = self.xreshape(fx)

        fh = self.hflatten(fh)
        fh = self.hlayer_1(fh)
        fh = self.hlayer_2(fh)
        fh = self.hlayer_3(fh)
        fh = self.hreshape(fh)

        matmul = self.matmul([fh, fx])

        fy = self.yflatten(matmul)
        fy = self.ylayer_1(fy)
        fy = self.ylayer_2(fy)
        fy = self.ylayer_3(fy)

        return fy = self.yrecon(fy)


class GFN_autoencoder(tf.keras.Model):

    def __init__(self, input_type, input_shape, latent_dim, factor_type):
        super(GFN_autoencoder, self).__init__(name='GFN_autoencoder')
        """
        input_type = ['image', 'image'], ['image', 'posture'], ['posture', 'posture']
        """
        self.input_type = input_type
        self.input_shape = input_shape
        self.latent_dim  = latent_dim
        self.factor_type = factor_type

        self.encoder = GFN_encoder(self.input_type, self.input_shape, self.latent_dim, self.factor_type)
        self.decoder = GFN_decoder(self.input_type, self.input_shape, self.latent_dim, self.factor_type)

    def call(self, inputs):
        x = self.encoder(inputs)
        return self.decoder(x)
