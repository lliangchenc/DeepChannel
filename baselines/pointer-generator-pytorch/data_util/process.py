"""
Custom preprocess for pointer_generator.
The output format is the same with https://github.com/abisee/cnn-dailymail, including one vocab and three splits.
Differences are as follows:
    - we use spacy, a more easily-used python tokenizer instead of stanford parser
    - we process documents more carefully
    - we support both cnn, dailymail and duc with a unified interface
    - we remove chunk
"""

import os
import re
import argparse
import numpy as np
import spacy
import random
import hashlib
import copy
from tqdm import tqdm
from collections import Counter
#from IPython import embed
import struct
from tensorflow.core.example import example_pb2

random.seed(666)

pattern_of_num = re.compile(r'[0-9]+')
nlp = None
def process_document(d, sentence_len_limit):
    '''
    functions:
        - lowercase
        - tokenize
        - replace numbers with 'zero'
        - remove sentences ending with ':' or '--'
        - remove sentences whose length <= sentence_len_limit
    '''
    global nlp
    if nlp is None:
        nlp = spacy.load('en')
    d = d.lower()
    tokenize_d = nlp(d)
    results = []
    for s in tokenize_d.sents:
        if not s.text.strip():
            continue
        sentence = []
        for w in s:
            if not w.text.strip():
                continue
            if pattern_of_num.match(w.text):
                sentence.append('zero')
            else:
                sentence.append(w.text)
        if sentence[-1]==':' or sentence[-1]=='--' or len(sentence)<=sentence_len_limit:
            continue
        results.append(' '.join(sentence))
    return results 


def hashhex(s):
    """Returns a heximal formated SHA1 hash of the input string."""
    h = hashlib.sha1()
    h.update(s)
    return h.hexdigest()


def read_cnn_dailymail(args):
    def key2File(data_dir, key2file):
        for k in os.listdir(data_dir):
            f = os.path.join(data_dir, k)
            k = k[:-6] # exclude suffix '.story'
            key2file[k] = f

    assert args.split_dir is not None and os.path.exists(args.split_dir), \
        "valid path of url_lists must be given to args.split_dir"
    data_type, data_dir = args.data_type, args.data_dir
    data = [[], [], []]
    length = [[], [], []] # length of each sentence, whose index is corresponding to data
    key2file = {} # key: hashkey, excluding the suffix '.story'; file: absolute file path.
    if data_type == 'cnn+dailymail':
        data_dir = data_dir.split(';')
        assert len(data_dir)==2
        for i in range(2):
            key2File(data_dir[i], key2file) # update key2file
        prefix = 'all_'
    elif data_type == 'cnn':
        key2File(data_dir, key2file)
        prefix = 'cnn_'
    elif data_type == 'daily':
        key2File(data_dir, key2file)
        prefix = 'dailymail_'
    prefix = os.path.join(args.split_dir, prefix) # complete prefix path
    for i, split in enumerate(('train', 'val', 'test')):
        url_file = prefix + split + '.txt'
        for line in tqdm(open(url_file).readlines()):
            k = hashhex(line.strip().encode())
            f = key2file[k] # file path
            parts = open(f, encoding='latin1').read().split('@highlight')
            docu = process_document(parts[0], 5)
            summ = process_document('.'.join(parts[1:]) + '.', 3)
            if len(docu)==0 or len(summ)==0:
                continue
            docu_len = [len(s) for s in docu]
            summ_len = [len(s) for s in summ]
            data[i].append([docu, summ])
            length[i].append([docu_len, summ_len])
    return data, length


def read_duc2007(args):
    data_type, data_dir = args.data_type, args.data_dir
    data = [[], [], []]
    length = [[], [], []]
    for i in range(45):
        doc_names = []
        for filename in os.listdir(data_dir):
            if int(filename.split(".")[0][1:]) == 701 + i:
                doc_names.append(filename)
        doc = ""
        summs = []
        for filename in doc_names:
            corpus = open(os.path.join(data_dir, filename)).read()
            is_summ = len(pattern_of_num.findall(filename.split(".")[-1])) == 0
            if is_summ:
                summs.append(corpus)
            else:
                doc += corpus
        doc_ = process_document(doc, 3)
        summs_ = [process_document(s, 3) for s in summs]
        for summ_ in summs_:
            temp_doc_ = copy.deepcopy(doc_)
            docu_len = [len(s) for s in temp_doc_]
            summ_len = [len(s) for s in summ_]
            data[2].append([temp_doc_,summ_])
            length[2].append([docu_len, summ_len])
    return data, length


def main():
    datasets = {
            'cnn+dailymail': read_cnn_dailymail,
            'cnn': read_cnn_dailymail,
            'daily': read_cnn_dailymail,
            'duc2007': read_duc2007,
            }
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', default='cnn+dailymail', choices=datasets.keys())
    parser.add_argument('--data_dir', default='/data/share/cnn_stories/stories;/data/share/dailymail_stories/stories', help='If data=cnn+dailimail, then data-dir must contain two paths for cnn and dailymail seperated by ;.')
    parser.add_argument('--split_dir', help='the split path for cnn or dailymail (url_lists provided by abisee)')
    parser.add_argument('--save_path', required=True)
    parser.add_argument('--max_word_num', type=int, default=50000)
    args = parser.parse_args()

    print('Reading data......')
    data, length = datasets[args.data_type](args)
    print('train/valid/test: %d/%d/%d' % tuple([len(_) for _ in data]))
    
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    wtof = {}
    if(args.data_type == 'duc2007'):
        print('DUC should use the vocab of training set, such as cnndaily')
    else:
        print('Build vocab. Count word frequency only from train set')
        # build vocab !
        for j in range(len(data[0])): # j-th sample of train set
            for k in range(2): # 0: content, 1: summary
                for l in range(len(data[0][j][k])): # l-th sentence
                    for word in data[0][j][k][l]:
                        wtof[word] = wtof.get(word, 0) + 1
        wtof = Counter(wtof).most_common(args.max_word_num)
        print('Preserve word num: %d. Examples: %s %s' % (len(needed_words), wtof[0][0], wtof[1][0]))
        with open(os.path.join(args.save_path, "vocab"), 'w') as f:
            for word, freq in wtof:
                f.write('%s %d\n' % (word, freq))
        print("Save vocab count into %s" % os.path.join(args.save_path, "vocab"))

    for i, split in enumerate(('train', 'val', 'test')):
        if len(data[i]) == 0:
            continue
        out_file = os.path.join(args.save_path, split)
        with open(out_file, 'wb') as f:
            for j in range(len(data[i])):
                article, abstract = data[i][j] # document, summary
                ##### convert to string then to byte
                article = ' '.join(article)
                abstract = ' '.join(["%s %s %s" % ('<s>', sent, '</s>') for sent in abstract])
                article = article.encode('utf-8')
                abstract = abstract.encode('utf-8')
                #####
                tf_example = example_pb2.Example()
                tf_example.features.feature['article'].bytes_list.value.extend([article])
                tf_example.features.feature['abstract'].bytes_list.value.extend([abstract])
                tf_example_str = tf_example.SerializeToString()
                str_len = len(tf_example_str)
                f.write(struct.pack('q', str_len))
                f.write(struct.pack('%ds' % str_len, tf_example_str))


if __name__ == "__main__":
    main()
