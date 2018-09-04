# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

from memn2n import MemN2N
from six.moves import range
from data import Vocab
import glob
import time
import shutil
import os
import data

import tensorflow as tf
import numpy as np

tf.flags.DEFINE_string('train_dir', 'baseline',
                       'training directory (models and summaries are saved there periodically)')
tf.flags.DEFINE_string("train_path", "tfrecord/training/*.tfrecord", "train_path")
tf.flags.DEFINE_string("validation_path", "tfrecord/validation/*.tfrecord", "validation_path")
tf.flags.DEFINE_float("learning_rate", 0.001, "Learning rate for SGD.")
tf.flags.DEFINE_float("anneal_rate", 6, "Number of epochs between halving the learnign rate.")
tf.flags.DEFINE_float("anneal_stop_epoch", 30, "Epoch number to end annealed lr schedule.")
tf.flags.DEFINE_float("max_grad_norm", 10.0, "Clip gradients to this norm.")
tf.flags.DEFINE_integer("batch_size", 64, "Batch size for training.")
tf.flags.DEFINE_integer("hops", 5, "Number of hops in the Memory Network.")
tf.flags.DEFINE_integer("epochs", 100, "Number of epochs to train for.")
tf.flags.DEFINE_integer("embedding_size", 100, "Embedding size for embedding matrices.")
tf.flags.DEFINE_integer("memory_size", 30, "Maximum size of memory.")
tf.flags.DEFINE_integer("random_state", None, "Random state.")
tf.flags.DEFINE_float("keep_prob", 0.5, "keep_prob.")
tf.flags.DEFINE_integer("sentence_size", 70, "sentence_size.")
tf.flags.DEFINE_integer('print_every', 10, 'how often to print current loss')
tf.flags.DEFINE_integer("record", 100, "evaluation_interval.")
tf.flags.DEFINE_float("l2", 0.00001, "l2.")
tf.flags.DEFINE_bool("word2vec_init", False, "word2vec_init")
tf.flags.DEFINE_integer("hidden_size", 200, "hidden_size")
tf.flags.DEFINE_integer("vocab", 100000, "hidden_size")


FLAGS = tf.flags.FLAGS

sentence_size = FLAGS.sentence_size
memory_size = FLAGS.memory_size

vocab_path = "vocab.txt"
vocab_size = FLAGS.vocab
vocab = Vocab(vocab_path, vocab_size)

# train/validation/test sets

tf.set_random_seed(FLAGS.random_state)
batch_size = FLAGS.batch_size


# dataset
def batcher(article_sentence_list, abstract, label_list, title, file_name):
    sentence_select_ground_truth = [int(l) for l in label_list.split(',')]
    article_sentence_list = article_sentence_list.split('/s/')
    abstract = abstract.split('/s/')
    enc_len = len(article_sentence_list)

    # article_sentence_list
    if enc_len > memory_size:
        article_sentence_list = article_sentence_list[:memory_size]
        sentence_select_ground_truth = sentence_select_ground_truth[:memory_size]

    enc_input = [[vocab._word2id(w) for i, w in enumerate(sentence.split(' ')) if i < sentence_size] for
                 sentence in article_sentence_list]

    # abstract
    if len(abstract) > memory_size:
        abstract = abstract[:memory_size]
    enc_abstract = [[vocab._word2id(w) for i, w in enumerate(single_abstract.split(' ')) if i < sentence_size] for
                    single_abstract in abstract]

    # title_input
    title_input = [vocab._word2id(w) for i, w in enumerate(title.split()) if i < sentence_size]

    # begin padding
    pad_id = vocab._word2id(data.PAD_TOKEN)  # 0

    # enc_input
    for sentence_index, sentence in enumerate(enc_input):
        while len(sentence) < sentence_size:
            sentence.append(pad_id)
        sentence = np.array(sentence, np.int32)
        enc_input[sentence_index] = sentence

    while len(enc_input) < memory_size:
        enc_input.append([pad_id for _ in xrange(sentence_size)])  # pad ten words a zero sentence
        sentence_select_ground_truth.append(0)

    # enc_abstract
    for sentence_index, sentence in enumerate(enc_abstract):
        while len(sentence) < sentence_size:
            sentence.append(pad_id)
        sentence = np.array(sentence, np.int32)
        enc_abstract[sentence_index] = sentence

    while len(enc_abstract) < memory_size:
        enc_abstract.append([pad_id for _ in xrange(sentence_size)])  # pad ten words a zero sentence

    while len(title_input) < sentence_size:
        title_input.append(pad_id)  # pad ten words a zero sentence

    enc_input = np.array(enc_input, np.int32)
    enc_abstract = np.array(enc_abstract, np.int32)
    title_input = np.array(title_input, np.int32)
    sentence_select_ground_truth = np.array(sentence_select_ground_truth, np.float32)

    return [enc_input, title_input, sentence_select_ground_truth, enc_abstract, file_name]


def parser(record):
    keys_to_features = {
        'article': tf.FixedLenFeature([], tf.string),
        'abstract': tf.FixedLenFeature([], tf.string),
        'label_list': tf.FixedLenFeature([], tf.string),
        'title': tf.FixedLenFeature([], tf.string),
        'file_name': tf.FixedLenFeature([], tf.string)
    }
    parsed = tf.parse_single_example(record, keys_to_features)
    article = parsed['article']
    abstract = parsed['abstract']
    label_list = parsed['label_list']
    title = parsed['title']
    file_name = parsed['file_name']
    return article, abstract, label_list, title, file_name


def ParseStory(story_file):
    f = open(story_file, "r")
    lines = f.readlines()
    lines = lines[2:]
    f.close()
    sentence_list = []
    abstract = []

    done = 0
    for line in lines:
        if line != '\n' and done == 0:
            line = line.split('\t\t\t')
            sentence_list.append(line[0])
        elif done == 0 and line == '\n':
            done = 1
        elif done == 1 and line != '\n':
            abstract.append(line[:-1])
        elif done == 1 and line == '\n':
            break

    return sentence_list, abstract


# dataset
filenames = tf.placeholder(tf.string, shape=[None])
dataset = tf.contrib.data.TFRecordDataset(filenames)
dataset = dataset.map(parser)
dataset = dataset.map(lambda article, abstract, label_list, title, file_name: tuple(
    tf.py_func(batcher, [article, abstract, label_list, title, file_name],
               [tf.int32, tf.int32, tf.float32, tf.int32, tf.string])))
dataset = dataset.batch(batch_size)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

train_filenames = glob.glob(FLAGS.train_path)
validation_filenames = glob.glob(FLAGS.validation_path)


with tf.Session() as sess:
    if os.path.exists(FLAGS.train_dir):
        shutil.rmtree(FLAGS.train_dir)
        os.mkdir(FLAGS.train_dir)
        print('Created training directory', FLAGS.train_dir)
    else:
        os.mkdir(FLAGS.train_dir)
        print('Created training directory', FLAGS.train_dir)

    model = MemN2N(batch_size, vocab_size, sentence_size, memory_size, FLAGS.embedding_size, next_element,
                   FLAGS.keep_prob, l2_para=FLAGS.l2,  word_init=FLAGS.word2vec_init,
                   hidden_size=FLAGS.hidden_size,
                   training=True,
                   session=sess,
                   hops=FLAGS.hops, max_grad_norm=FLAGS.max_grad_norm)
    print('created model')
    saver = tf.train.Saver(max_to_keep=40)

    train_writer = tf.summary.FileWriter(FLAGS.train_dir)
    summaries = tf.summary.merge_all()
    # create saver before creating more graph nodes, so that we do not save any vars defined below

    cnt = 0  # 总步数
    for epoch in range(1, FLAGS.epochs):
        # Stepped learning rate
        epoch_start_time = time.time()
        if epoch - 1 <= FLAGS.anneal_stop_epoch:
            anneal = 2.0 ** ((epoch - 1) // FLAGS.anneal_rate)
        else:
            anneal = 2.0 ** (FLAGS.anneal_stop_epoch // FLAGS.anneal_rate)
        lr = FLAGS.learning_rate / anneal

        total_cost = 0.0
        sess.run(iterator.initializer, feed_dict={filenames: train_filenames})

        # train
        while True:
            try:
                model._training = True
                cnt += 1
                feed_dict = {model._lr: lr}
                gradient, cost_t, _, l2loss = model._sess.run(
                    [summaries, model.loss, model.train_op, model.l2],
                    feed_dict=feed_dict)

                # print
                if cnt % 100 == 0:
                    print(
                        'epoch:%d step:%d, train_loss = %6.8f secs = %.4fs l2loss=%f' % (
                            epoch, cnt,
                            cost_t,
                            time.time() - epoch_start_time,
                            l2loss
                        ))
            except tf.errors.OutOfRangeError:
                break
            except tf.errors.InvalidArgumentError:
                break

        # # end of an epoch: validation
        # sess.run(iterator.initializer, feed_dict={filenames: validation_filenames})
        #
        # valid_loss_all = []
        # while True:
        #     try:
        #         model._training = False
        #         next_value, valid_loss, logits = model._sess.run(
        #             [next_element, model.loss, model.predict])  # (64,30)
        #         valid_loss_all.append(valid_loss)
        #
        #     except tf.errors.OutOfRangeError:
        #         break
        #     except  tf.errors.InvalidArgumentError:
        #         break
        #
        # valid_loss = np.mean(valid_loss_all)
        #
        # summary = tf.Summary(value=[
        #     tf.Summary.Value(tag="valid_loss", simple_value=valid_loss)
        # ])
        # train_writer.add_summary(summary, epoch)
        save_as = '%s/epoch%03d.model' % (FLAGS.train_dir, epoch)
        saver.save(sess, save_as)
        print('Saved model', save_as)
        print('Epoch training time:', time.time() - epoch_start_time, '\n')
