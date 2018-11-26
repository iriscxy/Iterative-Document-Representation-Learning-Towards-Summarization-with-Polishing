# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from data import Vocab

import data_utils as utils
import tensorflow as tf
from attention_gru_cell import AttentionGRUCell
from tensorflow.contrib import rnn
import numpy as np
from six.moves import range

def position_encoding(sentence_size, embedding_size):  # j d
    """
    Position Encoding described in section 4.1 [1]
    """
    encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
    ls = sentence_size + 1
    le = embedding_size + 1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i - 1, j - 1] = (i - (embedding_size + 1) / 2) * (j - (sentence_size + 1) / 2)
    encoding = 1 + 4 * encoding / embedding_size / sentence_size
    # Make position encoding of time words identity to avoid modifying them 
    # encoding[:, -1] = 1.0
    return np.transpose(encoding)


class MemN2N(object):
    def __init__(self, batch_size, vocab_size, sentence_size, memory_size, embedding_size, value,
                 keep_prob, l2_para,  word_init, hidden_size,
                 training,
                 hops=3,
                 max_grad_norm=40.0,
                 initializer=tf.random_normal_initializer(stddev=0.1),
                 encoding=position_encoding,
                 session=tf.Session(),
                 name='MemN2N'):
        self._batch_size = batch_size
        self._vocab_size = vocab_size
        self._sentence_size = sentence_size
        self._memory_size = memory_size
        self._embedding_size = embedding_size
        self._hops = hops
        self._max_grad_norm = max_grad_norm
        self._init = initializer
        self._name = name
        self._value = value
        self._keep_prob = keep_prob
        self._training = training
        self.l2_para = l2_para
        self.word_init = word_init
        self.hidden_size = hidden_size

        self._build_inputs()
        self._build_vars()

        self._opt = tf.train.AdamOptimizer(learning_rate=self._lr)

        self._encoding = tf.cast(tf.constant(encoding(self._sentence_size, self._embedding_size), name="encoding"),
                                 tf.float32)

        # cross entropy
        self.logits = self._inference(self._stories)  # (batch_size, vocab_size)

        lossL2 = 0
        for v in tf.trainable_variables():
            if not 'bias' in v.name.lower():
                lossL2 += self.l2_para * tf.nn.l2_loss(v)

        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits,
                                                                labels=tf.cast(self._summary, tf.float32),
                                                                name="cross_entropy")

        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        if self._training:
            self.cross_original = cross_entropy_mean
            self.l2 = lossL2
            cross_entropy_mean += lossL2

        # gradient pipeline
        grads_and_vars = self._opt.compute_gradients(cross_entropy_mean)

        for grad, var in grads_and_vars:
            if grad != None and var != None:
                tf.summary.histogram(var.name + '/gradient', grad)

        gvs = [(tf.clip_by_norm(grad, self._max_grad_norm), var) for grad, var in grads_and_vars if
               grad != None and var != None]
        train_op = self._opt.apply_gradients(gvs, name="train_op")
        self.grads = gvs

        # assign ops
        self.loss = cross_entropy_mean
        self.predict = self.logits
        self.train_op = train_op

        init_op = tf.global_variables_initializer()
        self._sess = session
        self._sess.run(init_op)

    def get_document_representation(self, hidden_states):
        """Get summary vectors via embedding and GRU"""
        hidden_states = tf.concat([hidden_states[0], hidden_states[1]], 2)
        hidden = tf.reduce_mean(hidden_states, 1)
        q_vec = tf.contrib.layers.fully_connected(hidden,
                                                  self.hidden_size,
                                                  activation_fn=tf.nn.tanh)
        return q_vec  # （64，300）

    def get_sentence_representation(self, stories):
        """Get fact (sentence) vectors via embedding, positional encoding and bi-directional GRU"""
        # get word vectors from embedding
        inputs = tf.nn.embedding_lookup(self.embeddings, stories)

        # use encoding to get sentence representation
        inputs = tf.reduce_sum(inputs * self._encoding, 2)
        forward_gru_cell = tf.contrib.rnn.GRUCell(self.hidden_size)
        backward_gru_cell = tf.contrib.rnn.GRUCell(self.hidden_size)
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(
            forward_gru_cell,
            backward_gru_cell,
            inputs,
            dtype=np.float32
        )

        # sum forward and backward output vectors
        fact_vecs = tf.reduce_sum(tf.stack(outputs), axis=0)

        # apply dropout
        if self._training:
            fact_vecs = tf.nn.dropout(fact_vecs, self._keep_prob)

        return fact_vecs, outputs  # （64，30，300）

    def get_attention(self, q_vec, prev_memory, fact_vec, reuse):
        """Use question vector and previous memory to create scalar attention for current fact"""
        with tf.variable_scope("attention", reuse=reuse):
            features = [
                fact_vec * prev_memory,
                fact_vec,
                prev_memory]

            feature_vec = tf.concat(features, 1)
            attention = tf.contrib.layers.fully_connected(feature_vec,
                                                          self._embedding_size,
                                                          activation_fn=tf.nn.tanh,
                                                          reuse=reuse, scope="fc1")
            attention = tf.contrib.layers.fully_connected(attention,
                                                          1,
                                                          activation_fn=None,
                                                          reuse=reuse, scope="fc2")
        return attention

    def generate_episode(self, memory, q_vec, fact_vecs, hop_index):
        """Generate episode by applying attention to current fact vectors through a modified GRU"""
        attentions = [tf.squeeze(
            self.get_attention(q_vec, memory, fv, bool(hop_index) or bool(i)), axis=1)
            for i, fv in enumerate(tf.unstack(fact_vecs, axis=1))]

        attentions = tf.transpose(tf.stack(attentions))
        self.attentions.append(attentions)
        attentions = tf.nn.softmax(attentions)
        attentions = tf.expand_dims(attentions, axis=-1)

        reuse = True if hop_index > 0 else False

        # concatenate fact vectors and attentions for input into attGRU
        gru_inputs = tf.concat([fact_vecs, attentions], 2)

        with tf.variable_scope('attention_gru', reuse=reuse):
            _, episode = tf.nn.dynamic_rnn(AttentionGRUCell(self.hidden_size),
                                           gru_inputs,
                                           dtype=np.float32
                                           )
        return episode

    def _build_inputs(self):
        self._stories = self._value[0]
        self._stories.set_shape([self._batch_size, self._memory_size, self._sentence_size])
        self._summary = self._value[2]
        self._summary.set_shape([self._batch_size, self._memory_size])
        self._lr = tf.placeholder(tf.float32, [], name="learning_rate")

    def _build_vars(self):
        with tf.variable_scope(self._name):
            vocab_path = "vocab.txt"
            vocab = Vocab(vocab_path, self._vocab_size)
            if self.word_init:
                word2vec = utils.load_glove(self._embedding_size)
                self.embeddings = utils.create_embedding(self, word2vec, vocab.id_to_word, self._embedding_size)
            else:
                self.embeddings = np.random.uniform(-0.5, 0.5,
                                                    (len(vocab.id_to_word), self._embedding_size))
            self.embeddings = tf.Variable(self.embeddings.astype(np.float32), name="Embedding")

    def _inference(self, stories):
        with tf.variable_scope(self._name):
            # input fusion module

            with tf.variable_scope("article", initializer=tf.contrib.layers.xavier_initializer()):
                print('==> get article representation')
                fact_vecs, hidden_states = self.get_sentence_representation(stories)

            with tf.variable_scope("summary", initializer=tf.contrib.layers.xavier_initializer()):
                print('==> get summary representation')
                q_vec = self.get_document_representation(hidden_states)

            self.attentions = []
            hop3_out = []

            # memory module
            with tf.variable_scope("memory", initializer=tf.contrib.layers.xavier_initializer()):
                print('==> build episodic memory')

                # generate n_hops episodes
                prev_memory = q_vec
                gru_cell = tf.contrib.rnn.GRUCell(self.hidden_size)

                for i in range(self._hops):
                    # get a new episode
                    print('==> generating episode', i)
                    episode = self.generate_episode(prev_memory, q_vec, fact_vecs, i)

                    # untied weights for memory update
                    with tf.variable_scope("hop_%d" % i):

                        _,prev_memory=gru_cell(inputs=episode,state=prev_memory)

                        hop3_out.append(prev_memory)

            with tf.variable_scope('bidirectional_rnn'):
                gru_fw_cell = tf.contrib.rnn.GRUCell(self.hidden_size)
                gru_bw_cell = tf.contrib.rnn.GRUCell(self.hidden_size)
                if self._training:
                    gru_fw_cell = rnn.DropoutWrapper(gru_fw_cell, output_keep_prob=self._keep_prob)
                    gru_bw_cell = rnn.DropoutWrapper(gru_bw_cell, output_keep_prob=self._keep_prob)

            outputs = []
            output = []

            for hopn in range(self._hops):
                temp = tf.nn.bidirectional_dynamic_rnn(gru_fw_cell, gru_bw_cell,
                                                       inputs=fact_vecs, initial_state_fw=hop3_out[hopn],
                                                       initial_state_bw=hop3_out[hopn],
                                                       dtype=tf.float32)
                outputs.append(temp[0])
                output.append(tf.concat([outputs[hopn][0], outputs[hopn][1]], 2))  # (64,30,400)
                if self._training:
                    output[hopn] = tf.nn.dropout(output[hopn], self._keep_prob)

            concat_layer = tf.concat([output[0], output[1], output[2], output[3], output[4]], 2)
            logits = tf.layers.dense(concat_layer, 70, activation=tf.tanh)
            logits = tf.layers.dense(logits, 1, None)
            logits = tf.squeeze(logits, [-1])
            return logits

