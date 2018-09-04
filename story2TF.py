# encoding=utf-8
import getopt
import os
import sys
from os import listdir
from os.path import isfile, join

import tensorflow as tf
from numpy.random import shuffle as random_shuffle
from tensorflow.core.example import example_pb2



input_dir = "input_path"
output_dir = "output_path"
CHUNK_SIZE = 256


def ParseStory(story_file):
    print(story_file+'\n')
    f = open(story_file, "r")
    story_file=story_file.split('/')
    story_file=story_file[-1]
    story_file=str(story_file)
    lines = f.readlines()
    title = ' '.join(lines[0].split('/')[-1].split('.')[0].split('-'))
    lines = lines[2:]
    f.close()
    sentence_list = []
    label_list = []
    abstract = []

    done=0
    for line in lines:
        if line != '\n' and done==0:
            line = line.split('\t\t\t')
            label = int(line[1])
            sentence_list.append(line[0])
            # 1 means need to extract
            label_list.append('1' if label == 1 or label == 2 else '0')
        elif done==0 and line=='\n':
            done=1
        elif done==1 and line!='\n':
            abstract.append(line[:-1])
        elif done==1 and line=='\n':
            break
    return bytes('/s/'.join(sentence_list)), bytes('/s/'.join(abstract)), bytes(','.join(label_list)), bytes(title),bytes(story_file)


def WriteTFrecords(stories_directory, tf_directory, outfiles, fraction):
    stories = [f for f in listdir(stories_directory) if isfile(join(stories_directory, f))]
    random_shuffle(stories)

    print("Writing TFrecords files")
    print('story dir {} has {} stories'.format(stories_directory, len(stories)))

    index_start = 0
    for index, outfile in enumerate(outfiles):
        counts = int(len(stories) * fraction[index])
        index_stop = min(index_start + counts, len(stories))
        index1 = index_start
        fileindex = 0

        while index1 < index_stop:
            index1 = min(index_start + CHUNK_SIZE, index_stop)
            story_files = stories[index_start:index1]

            writer = tf.python_io.TFRecordWriter(join(tf_directory, outfile + '_' + str(fileindex) + '.tfrecord'))
            for story in story_files:
                try:
                    article_sentence_list, abstract, label_list, title,file_name = ParseStory(join(stories_directory, story))
                except:
                    continue
                tf_example = example_pb2.Example()
                tf_example.features.feature['article'].bytes_list.value.extend([article_sentence_list])
                tf_example.features.feature['abstract'].bytes_list.value.extend([abstract])
                tf_example.features.feature['label_list'].bytes_list.value.extend([label_list])
                tf_example.features.feature['title'].bytes_list.value.extend([title])
                tf_example.features.feature['file_name'].bytes_list.value.extend([file_name])
                tf_example_str = tf_example.SerializeToString()
                writer.write(tf_example_str)
            writer.close()
            fileindex += 1
            index_start = index1
    print("Done writing TFrecords file to directory \"%s\" " % tf_directory)

if __name__ == '__main__':
    dataset_list = ['training','validation','test']
    for dataset in dataset_list:
        stories_directory = os.path.join(input_dir, dataset)
        outdir = os.path.join(output_dir, dataset)
        print(outdir)
        if not os.path.exists(outdir): os.makedirs(outdir)
        WriteTFrecords(stories_directory, outdir, [dataset], [1.0])