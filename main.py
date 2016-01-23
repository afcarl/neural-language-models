__author__ = 'hiroki'

import theano
import numpy as np

theano.config.floatX = 'float32'


np.random.seed(0)


if __name__ == '__main__':
    import argparse
    import train

    parser = argparse.ArgumentParser(description='Train/Test A Language Model.')

    parser.add_argument('-data',  help='path to data')

    # NN architecture
    parser.add_argument('--vocab',  type=int, default=100000000, help='vocabulary size')
    parser.add_argument('--emb',    type=int, default=32,        help='dimension of embeddings')
    parser.add_argument('--hidden', type=int, default=32,        help='dimension of hidden layer')
    parser.add_argument('--layer',  type=int, default=1,         help='number of layers')

    # Training Parameters
    parser.add_argument('--batch', type=int, default=10, help='batch size')
    parser.add_argument('--opt', default='adagrad', help='optimization method')
    parser.add_argument('--epoch', type=int, default=100, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--save', type=bool, default=False, help='parameters to be saved or not')
    parser.add_argument('--init_emb', default=None, help='Initial embedding to be loaded')
    parser.add_argument('--n_words',  type=int, default=100, help='number of words')

    train.main(parser.parse_args())
