from __future__ import print_function
import argparse
from collections import Counter
from itertools import chain
import os



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_vocab_size', default=100000, type=int, help='source vocabulary size')
    parser.add_argument('--src', default='../newdata/training/0a0b41be7d1612150b3cbcccfa40094b7565780d.summary', help='file of source sentences')
    parser.add_argument('--output', default='vocab.txt', type=str, help='output vocabulary file')

    args = parser.parse_args()


    data = []
    for file in os.listdir('../newdata/training'):
        for line in open('../newdata/training/'+file):
            sent = line.strip().split(' ')
            data.append(sent)
    for file in os.listdir('../newdata/validation'):
        for line in open('../newdata/validation/'+file):
            sent = line.strip().split(' ')
            data.append(sent)
    for file in os.listdir('../newdata/test'):
        for line in open('../newdata/test/'+file):
            sent = line.strip().split(' ')
            data.append(sent)


    word_freq = Counter(chain(*data))
    non_singletons = [w for w in word_freq if word_freq[w] > 1]
    print('number of word types: %d, number of word types w/ frequency > 1: %d' % (len(word_freq),
                                                                                   len(non_singletons)))

    top_k_words = sorted(word_freq.keys(), reverse=True, key=word_freq.get)[:150000]
    f = open('vocab.txt', 'w')
    for word in top_k_words:
        if word !='' and len(word.split())==1:
            f.write(word + ' ' + str(word_freq[word]) + '\n')

    a = 1
