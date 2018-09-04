# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Script for reading and processing the train/eval/test data
"""

PAD_TOKEN = '<PAD>'
UNKNOWN_TOKEN = '<UNK>'


class Vocab(object):
    """Vocabulary class for mapping between words and ids (integers)
    """

    def __init__(self, vocab_file, max_size):
        """
        """
        print(max_size)
        self.word_to_id = {}
        self.id_to_word = {}
        self.count = 0

        for w in [PAD_TOKEN, UNKNOWN_TOKEN]:
            self.word_to_id[w] = self.count
            self.id_to_word[self.count] = w
            self.count += 1

        with open(vocab_file, 'r') as vocab_f:
            for line in vocab_f:
                pieces = line.split()
                if len(pieces) != 2:
                    print('WARNING: incorrectly formatted line in vocabulary file: {}'.format(line))
                    continue
                if pieces[0] in self.word_to_id:
                    raise ValueError('Duplicated word in vocabulary file: {}.'.format(pieces[0]))
                self.word_to_id[pieces[0]] = self.count
                self.id_to_word[self.count] = pieces[0]
                self.count += 1
                if max_size != 0 and self.count >= max_size:
                    break
        print(max_size)
        print('INFO: Finished reading {} of {} words in vocab, last word added: {}'.format(self.count, max_size,
                                                                                           self.id_to_word[
                                                                                               self.count - 1]))

    def _word2id(self, word):
        """Returns the id (integer) of a word (string). Returns <UNK> id if word is OOV.
        """
        if word not in self.word_to_id:
            return self.word_to_id[UNKNOWN_TOKEN]
        return self.word_to_id[word]

    def _id2word(self, word_id):
        """Returns the word (string) corresponding to an id (integer).
        """
        if word_id not in self.id_to_word:
            raise ValueError('Id not found in vocab: %d' % word_id)
        return self.id_to_word[word_id]

    def _size(self):
        """Returns the total size of the vocabulary
        """
        return self.count
