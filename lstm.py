__author__ = 'hiroki'

import numpy as np
import theano
import theano.tensor as T

import utils
from utils import tanh, sample_weights
from layer import Layer, FirstLayer, LastLayer


class LSTM(object):
    def __init__(self,
                 w,
                 d,
                 n_layers,
                 vocab_size,
                 n_in=32,
                 n_h=32,
                 n_words=1000,
                 batch_size=32,
                 activation=tanh):

        self.w = w
        self.d = d

        """model parameters"""
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.n_in = n_in
        self.n_h = n_h
        self.n_y = vocab_size
        self.n_words = n_words
        self.batch_size = batch_size
        self.activation = activation

        """embeddings"""
        self.emb = theano.shared(sample_weights(self.vocab_size, self.n_in))

        """initial parameters"""
        self.x = self.emb[self.w]  # x: 1D: n_words * batch_size, 2D: n_in

        self.c0 = theano.shared(np.zeros((self.batch_size, n_h), dtype=theano.config.floatX))
        self.h0 = self.activation(self.c0)

        """layers and parameters"""
        self.layers, self.params, self.layer_output = self.layers(n_layers=n_layers)

        self.y = self.layer_output[-1]  # y: 1D: n_words, 2D: batch_size, 3D: vocab_size
        if n_layers % 2 == 0:
            self.y = self.y[::-1]

        self.p_y_given_x = self.y.dimshuffle((1, 0, 2)).reshape((n_words * batch_size, vocab_size))
        self.nll = -T.mean(T.log(self.p_y_given_x)[T.arange(d.shape[0]), d])
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.errors = T.neq(self.y_pred, d)

    def layers(self, n_layers=1):
        layers = []
        params = []
        layer_output = []
        for i in xrange(n_layers):
            if i == 0:
                layer_input = self.x.reshape((self.batch_size, self.n_words, self.n_in)).dimshuffle(1, 0, 2)  # 100 * 10 * 32
                layer = FirstLayer(n_i=self.n_in)
            else:
                layer_input = layer_output[-1][::-1]
                layer = Layer(n_i=self.n_in)
            [h, c], _ = theano.scan(fn=layer.forward,
                                    sequences=layer_input,
                                    outputs_info=[self.h0, self.c0])
            layers.append(layer)
            params.extend(layer.params)
            layer_output.append(h)

        layer_input = layer_output[-1]
        layer = LastLayer(n_i=self.n_in, n_h=self.n_y)
        y, _ = theano.scan(fn=layer.forward,
                           sequences=layer_input,
                           outputs_info=[None])
        layers.append(layer)
        params.extend(layer.params)
        layer_output.append(y)
        return layers, params, layer_output

    def set_params(self, params):
        k = 0
        self.c0.set_value(params[0])
        k += 1
        for i in xrange(len(self.layers)-1):
            layer = self.layers[i]
            for p in layer.params:
                p.set_value(params[k])
                k += 1
        for p in self.layers[-1].params:
            p.set_value(params[k])
            k += 1

    def load(self, init_emb, init_params):
        emb = utils.load_data(init_emb)
        self.emb.set_value(emb)
        params = utils.load_data(init_params)
        self.set_params(params)

    def save(self):
        utils.dump_data(data=self.dump_params(), fn='layers%d.emb%d.vocab%d' % (self.n_layers, self.n_in, self.vocab_size))
        utils.dump_data(data=self.emb.get_value(), fn='emb%d.vocab%d.layers%d' % (self.n_in, self.vocab_size, self.n_layers))

    def dump_params(self):
        return [p.get_value(True) for p in self.params]
