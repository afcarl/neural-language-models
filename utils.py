__author__ = 'hiroki'

from collections import defaultdict

import numpy as np
import theano
import theano.tensor as T

UNK = u'<UNK>'
EOS = u'<EOS>'


class Vocab(object):

    def __init__(self):
        self.i2w = []
        self.w2i = {}

    def add_word(self, word):
        assert isinstance(word, unicode)
        if word not in self.w2i:
            new_id = self.size()
            self.i2w.append(word)
            self.w2i[word] = new_id

    def get_id(self, word):
        assert isinstance(word, unicode)
        return self.w2i.get(word)

    def get_word(self, w_id):
        return self.i2w[w_id]

    def has_key(self, word):
        return self.w2i.has_key(word)

    def size(self):
        return len(self.i2w)

    def save(self, path):
        with open(path, 'w') as f:
            for i, w in enumerate(self.i2w):
                print >> f, str(i) + '\t' + w.encode('utf-8')

    @classmethod
    def load(cls, path):
        vocab = Vocab()
        with open(path) as f:
            for line in f:
                w = line.strip().split('\t')[1].decode('utf-8')
                vocab.add_word(w)
        return vocab


def load_conll(path, vocab_size=100000000, file_encoding='utf-8'):
    corpus = []
    word_freqs = defaultdict(int)

    vocab_word = Vocab()
#    vocab_word.add_word(EOS)
    vocab_word.add_word(UNK)

    with open(path) as f:
        for line in f:
            es = line.rstrip().split()
            words = map(lambda w: w.decode(file_encoding), es)
            for w in words:
                word_freqs[w] += 1
            corpus.extend(words)

    for w, f in sorted(word_freqs.items(), key=lambda (k, v): -v):
        if f == 1:
            break
        if vocab_size is not None and vocab_word.size() < vocab_size:
            vocab_word.add_word(w)
        else:
            break

    return corpus, vocab_word


def convert_words_into_ids(corpus, vocab_word):
    id_corpus = []
    for w in corpus:
        w_id = vocab_word.get_id(w[0])
        if w_id is None:
            """ID for unknown word"""
            w_id = vocab_word.get_id(UNK)
        assert w_id is not None
        id_corpus.append(w_id)
    return id_corpus


def convert_data(id_corpus):
    return T.cast(theano.shared(np.asarray(id_corpus, dtype=theano.config.floatX),
                                borrow=True),
                  'int32')


def relu(x):
    return T.maximum(0, x)


def sigmoid(x):
    return T.nnet.sigmoid(x)


def tanh(x):
    return T.tanh(x)


def sample_weights(size_x, size_y=0, sig=False):
    if size_y == 0:
        W = np.asarray(np.random.uniform(low=-np.sqrt(6.0 / (size_x + size_y)),
                                         high=np.sqrt(6.0 / (size_x + size_y)),
                                         size=size_x),
                       dtype=theano.config.floatX)
    else:
        W = np.asarray(np.random.uniform(low=-np.sqrt(6.0 / (size_x + size_y)),
                                         high=np.sqrt(6.0 / (size_x + size_y)),
                                         size=(size_x, size_y)),
                       dtype=theano.config.floatX)
    if sig:
        W *= 4.0
    return W


def build_shared_zeros(shape):
    return theano.shared(
        value=np.zeros(shape, dtype=theano.config.floatX),
        borrow=True
    )

