__author__ = 'hiroki'

import sys
import time

import utils
import lstm
import optimizers

import numpy as np
import theano
import theano.tensor as T


def main(argv):

    print '\nSYSTEM START\n'
    print 'Emb Dim: %d\tHidden Dim: %d\tOptimization: %s\tLayer: %d\tEpoch: %d' %\
          (argv.emb, argv.hidden, argv.opt, argv.layer, argv.epoch)
    print 'Parameters to be saved: %s' % argv.save

    """data preprocessing"""
    print 'DATA Preprocessing...'
    corpus, vocab_word = utils.load_conll(argv.data)
    id_corpus = utils.convert_words_into_ids(corpus, vocab_word)
    train_samples = utils.convert_data(id_corpus)
    n_samples = len(id_corpus)
    print 'Samples: %d\tVocab: %d' % (n_samples, vocab_word.size())

    """symbol definition"""
    index = T.iscalar()
    w = T.ivector()
    d = T.ivector()
    n_hidden = argv.hidden
    n_words = argv.n_words
    batch_size = argv.batch

    """model setup"""
    print 'Compiling Theano Code...'
    model = lstm.LSTM(w=w, d=d, n_layers=argv.layer, vocab_size=vocab_word.size(), n_in=n_hidden, n_h=n_hidden,
                      n_words=n_words, batch_size=batch_size
                      )
    cost = model.nll
    opt = optimizers.main(name=argv.opt, cost=cost, params=model.params, emb=model.emb, x=model.x, w=model.w)

    """ train """
    def _train():
        train_model = theano.function(
            inputs=[index],
            outputs=[model.nll, model.errors],
            updates=opt,
            givens={
                w: train_samples[index * n_words * batch_size: (index+1) * n_words * batch_size],
                d: train_samples[index * n_words * batch_size + 1: (index+1) * n_words * batch_size + 1]
            },
            mode='FAST_RUN'
        )

        n_batch_samples = n_samples / n_words / batch_size
        print 'Vocabulary Size: %d\tBatch Sample Size: %d' % (vocab_word.size(), n_batch_samples)
        print '\nTrain START'

        for epoch in xrange(argv.epoch):
            print '\nEpoch: %d' % (epoch + 1)
            print '\tIndex: ',
            start = time.time()

            losses = []
            errors = []
            for b_index in xrange(n_batch_samples):
                if b_index % 100 == 0 and b_index != 0:
                    print b_index,
                    sys.stdout.flush()
                loss, error = train_model(b_index)
                losses.append(loss)
                errors.append(error)
            avg_loss = np.mean(losses)
            end = time.time()
            print '\tTime: %f seconds' % (end - start)
            print '\tAverage Negative Log Likelihood: %f' % avg_loss

            total = 0.0
            correct = 0
            for sent in errors:
                total += len(sent)
                for y_pred in sent:
                    if y_pred == 0:
                        correct += 1
            print '\tTrain Accuracy: %f' % (correct / total)
            if argv.save:
                model.save()

    _train()
