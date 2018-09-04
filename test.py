# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

from memn2n import MemN2N
from six.moves import range
from data import Vocab
import glob
import time
import shutil
import math
import data

import tensorflow as tf
import numpy as np
import os

tf.flags.DEFINE_string('train_dir', '', 'model')
tf.flags.DEFINE_string('test_dir', 'test', 'model')
tf.flags.DEFINE_float("learning_rate", 0.001, "Learning rate for SGD.")
tf.flags.DEFINE_float("max_grad_norm", 40.0, "Clip gradients to this norm.")
tf.flags.DEFINE_integer("batch_size", 64, "Batch size for training.")
tf.flags.DEFINE_integer("hops", 5, "Number of hops in the Memory Network.")
tf.flags.DEFINE_integer("epochs", 1, "Number of epochs to train for.")
tf.flags.DEFINE_integer("embedding_size", 100, "Embedding size for embedding matrices.")
tf.flags.DEFINE_integer("memory_size", 30, "Maximum size of memory.")
tf.flags.DEFINE_integer("random_state", None, "Random state.")
tf.flags.DEFINE_float("keep_prob", 0.5, "keep_prob.")
tf.flags.DEFINE_integer("sentence_size", 50, "sentence_size.")
tf.flags.DEFINE_integer("l2", 0.00001, "l2.")
tf.flags.DEFINE_bool("word2vec_init", False, "word2vec_init")
tf.flags.DEFINE_integer("hidden_size", 200, "hidden_size")
tf.flags.DEFINE_integer("vocab", 100000, "hidden_size")


FLAGS = tf.flags.FLAGS

test_path = "tfrecord/test/*.tfrecord"
sentence_size = FLAGS.sentence_size
memory_size = FLAGS.memory_size

vocab_path = "vocab.txt"
vocab_size = FLAGS.vocab

vocab = Vocab(vocab_path, vocab_size)



print("Longest sentence length", sentence_size)
print("Longest story length", memory_size)

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
    pad_id = vocab._word2id(data.PAD_TOKEN)

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

test_filenames = glob.glob(test_path)

with tf.Session() as sess:
    lead3_dir = os.path.join(FLAGS.test_dir, 'lead3')
    mem_dir = os.path.join(FLAGS.test_dir, 'mem')
    if os.path.exists(FLAGS.test_dir):
        shutil.rmtree(FLAGS.test_dir)
        os.mkdir(FLAGS.test_dir)
        os.mkdir(lead3_dir)
        os.mkdir(mem_dir)
        os.mkdir(lead3_dir + '/system')
        os.mkdir(lead3_dir + '/model')
        os.mkdir(mem_dir + '/system')
        os.mkdir(mem_dir + '/model')
        print('Created test directory', FLAGS.test_dir)
    else:
        os.mkdir(FLAGS.test_dir)
        os.mkdir(lead3_dir)
        os.mkdir(mem_dir)
        os.mkdir(lead3_dir + '/system')
        os.mkdir(lead3_dir + '/model')
        os.mkdir(mem_dir + '/system')
        os.mkdir(mem_dir + '/model')
        print('Created test directory', FLAGS.test_dir)

    if os.path.exists('se.txt'):
        os.remove('se.txt')

    model = MemN2N(batch_size, vocab_size, sentence_size, memory_size, FLAGS.embedding_size, next_element,
                   FLAGS.keep_prob, l2_para=FLAGS.l2,  word_init=FLAGS.word2vec_init,
                   hidden_size=FLAGS.hidden_size,
                   training=False,
                   session=sess,
                   hops=FLAGS.hops, max_grad_norm=FLAGS.max_grad_norm)
    print('created model')

    saver = tf.train.Saver()
    saver.restore(sess, FLAGS.train_dir)
    names=[]
    cnt = -1  # file index
    for epoch in range(1, FLAGS.epochs + 1):
        epoch_start_time = time.time()

        total_cost = 0.0
        sess.run(iterator.initializer, feed_dict={filenames: test_filenames})
        while True:
            try:
                label, cost_t, logits = model._sess.run([next_element, model.loss, model.predict])
                for i in xrange(batch_size):
                    f1 = open(lead3_dir + '/system/system' + str(cnt), 'w+')
                    f2 = open(lead3_dir + '/model/model' + str(cnt), 'w+')
                    f3 = open(mem_dir + '/system/system' + str(cnt), 'w+')
                    f4 = open(mem_dir + '/model/model' + str(cnt), 'w+')

                    sentence, abstract = ParseStory("path_to_original_data" + label[4][i])
                    cnt += 1
                    if len(sentence) == 1:
                        f1.write(sentence[0])
                        f3.write(sentence[0])
                    elif len(sentence) == 2: 
                        f1.write(sentence[0] + ' \n' + sentence[1])
                        f3.write(sentence[0] + ' \n' + sentence[1])
                    else:
                        # prediction
                        evaluate = ''
                        select = []
                        arg = np.argsort(-logits[i])
                        cnnt = 0
                        ftemp = open('se.txt', 'a+')
                        ftemp.write(label[4][i] + '\n')
                        for j in xrange(FLAGS.memory_size):
                            if arg[j] not in select and arg[j] < len(sentence) and cnnt < 3:
                                cnnt += 1
                                select.append(arg[j])
                                evaluate += sentence[arg[j]] + ' \n'
                            if cnnt == 3:
                                break
                        evaluate = evaluate[:-1]
                        f3.write(evaluate)
                        f3.close()
                        ftemp.write('\n')
                        ftemp.close()

                        # lead3
                        evaluate = ''
                        for j in xrange(3):
                            evaluate += sentence[j] + ' \n'
                        evaluate = evaluate[:-1]
                        f1.write(evaluate)
                        f1.close()

                    # standard
                    reference = ''
                    for each in abstract:
                        reference += str(each) + '\n'
                    reference = reference[:-1]
                    f2.write(reference)
                    f4.write(reference)
                    f2.close()
                    f2.close()

            except tf.errors.OutOfRangeError:
                break
            except tf.errors.InvalidArgumentError:
                break

print('Epoch training time:', time.time() - epoch_start_time, '\n')
