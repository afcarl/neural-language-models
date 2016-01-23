__author__ = 'hiroki'

from utils import sample_weights, sigmoid, tanh

import theano
import theano.tensor as T


class Layer(object):
    def __init__(self,
                 n_i=32,
                 n_h=32,
                 activation=tanh
                 ):

        self.activation = activation

        """input gate parameters"""
        self.W_xi = theano.shared(sample_weights(n_i, n_h))
        self.W_hi = theano.shared(sample_weights(n_h, n_h))

        """forget gate parameters"""
        self.W_xf = theano.shared(sample_weights(n_i, n_h))
        self.W_hf = theano.shared(sample_weights(n_h, n_h))

        """cell parameters"""
        self.W_xc = theano.shared(sample_weights(n_i, n_h))
        self.W_hc = theano.shared(sample_weights(n_h, n_h))

        """output gate parameters"""
        self.W_xo = theano.shared(sample_weights(n_i, n_h))
        self.W_ho = theano.shared(sample_weights(n_h, n_h))

        self.params = [self.W_xi, self.W_hi, self.W_xf, self.W_hf,
                       self.W_xc, self.W_hc, self.W_xo, self.W_ho]

    def forward(self, x_t, h_tm1, c_tm1):
        i_t = sigmoid(T.dot(x_t, self.W_xi) + T.dot(h_tm1, self.W_hi) + c_tm1)
        f_t = sigmoid(T.dot(x_t, self.W_xf) + T.dot(h_tm1, self.W_hf) + c_tm1)
        c_t = f_t * c_tm1 + i_t * self.activation(T.dot(x_t, self.W_xc) + T.dot(h_tm1, self.W_hc))
        o_t = sigmoid(T.dot(x_t, self.W_xo) + T.dot(h_tm1, self.W_ho) + c_t)
        h_t = o_t * self.activation(c_t)
        return h_t, c_t


class FirstLayer(object):
    def __init__(self,
                 n_i=32,
                 n_h=32,
                 activation=tanh
                 ):

        self.activation = activation

        """input gate parameters"""
        self.W_xi = theano.shared(sample_weights(n_i, n_h))
        self.W_hi = theano.shared(sample_weights(n_h, n_h))
        self.W_ci = theano.shared(sample_weights(n_h))

        """forget gate parameters"""
        self.W_xf = theano.shared(sample_weights(n_i, n_h))
        self.W_hf = theano.shared(sample_weights(n_h, n_h))
        self.W_cf = theano.shared(sample_weights(n_h))

        """cell parameters"""
        self.W_xc = theano.shared(sample_weights(n_i, n_h))
        self.W_hc = theano.shared(sample_weights(n_h, n_h))

        """output gate parameters"""
        self.W_xo = theano.shared(sample_weights(n_i, n_h))
        self.W_ho = theano.shared(sample_weights(n_h, n_h))
        self.W_co = theano.shared(sample_weights(n_h))

        self.params = [self.W_xi, self.W_hi, self.W_ci, self.W_xf, self.W_hf, self.W_cf,
                       self.W_xc, self.W_hc, self.W_xo, self.W_ho, self.W_co]

    def forward(self, x_t, h_tm1, c_tm1):
        i_t = sigmoid(T.dot(x_t, self.W_xi) + T.dot(h_tm1, self.W_hi) + c_tm1 * self.W_ci)
        f_t = sigmoid(T.dot(x_t, self.W_xf) + T.dot(h_tm1, self.W_hf) + c_tm1 * self.W_cf)
        c_t = f_t * c_tm1 + i_t * self.activation(T.dot(x_t, self.W_xc) + T.dot(h_tm1, self.W_hc))
        o_t = sigmoid(T.dot(x_t, self.W_xo) + T.dot(h_tm1, self.W_ho) + c_t * self.W_co)
        h_t = o_t * self.activation(c_t)
        return h_t, c_t


class LastLayer(object):
    def __init__(self,
                 n_i=32,
                 n_h=45
                 ):

        self.W = theano.shared(sample_weights(n_i, n_h))
        self.params = [self.W]

    def forward(self, h_t):
        y_t = T.nnet.softmax(T.dot(h_t, self.W))
        return y_t
