#encoding=utf-8
import os
import re
import sys
import argparse
import numpy as np
import pickle
import spacy
import hashlib
import copy
import codecs
import json
from tqdm import tqdm
from collections import Counter
#from IPython import embed
import xml.etree.ElementTree as et
from data import Dataset

pattern_of_num = re.compile(r'[0-9]+')
pattern_unicode = re.compile(r'\\u[\w]{4}')
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
        results.append(sentence)
    return results 

def hashhex(s):
    """Returns a heximal formated SHA1 hash of the input string."""
    h = hashlib.sha1()
    h.update(s)
    return h.hexdigest()

def read_cnn_dailymail(data_type, data_dir):
    def key2File(data_dir, key2file):
        for k in os.listdir(data_dir):
            f = os.path.join(data_dir, k)
            k = k[:-6] # exclude suffix '.story'
            key2file[k] = f

    data = [[], [], []]
    length = [[], [], []] # length of each sentence, whose index is corresponding to data
    key2file = {} # key: hashkey, excluding the suffix '.story'; file: absolute file path.
    if data_type == 'cnn+dailymail':
        data_dir = data_dir.split(';')
        assert len(data_dir)==2
        for i in range(2):
            key2File(data_dir[i], key2file) # update key2file
        prefix = './cnndaily_url_splits/all_'
    elif data_type == 'cnn':
        key2File(data_dir, key2file)
        prefix = './cnndaily_url_splits/cnn_'
    elif data_type == 'daily':
        key2File(data_dir, key2file)
        prefix = './cnndaily_url_splits/dailymail_'
    for i, split in enumerate(['train', 'val', 'test']):
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


def read_duc2007(data_type, data_dir):
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
    parser.add_argument('--glove', default='/data/sjx/glove.6B.100d.py36.pkl', help='pickle file of glove')
    parser.add_argument('--data', default='cnn+dailymail', choices=datasets.keys())
    parser.add_argument('--data-dir', default='/data/share/cnn_stories/stories;/data/share/dailymail_stories/stories', help='If data=cnn+dailimail, then data-dir must contain two paths for cnn and dailymail seperated by ;.')
    parser.add_argument('--save-path', required=True)
    parser.add_argument('--max-word-num', type=int, default=50000)
    args = parser.parse_args()

    print('Loading glove......')
    glove = pickle.load(open(args.glove, 'rb'))
    word_dim = len(glove['the'])
    print('Word dim = %d' % word_dim)

    print('Reading data......')
    data, length = datasets[args.data](args.data, args.data_dir)
    print('train/valid/test: %d/%d/%d' % tuple([len(_) for _ in data]))
    

    print('Count word frequency only from train set......')
    wtof = {}
    if(args.data == 'duc2007'):
        pass
    else:
        for j in range(len(data[0])): # j-th sample of train set
            for k in range(2): # 0: content, 1: summary
                for l in range(len(data[0][j][k])): # l-th sentence
                    for word in data[0][j][k][l]:
                        wtof[word] = wtof.get(word, 0) + 1
        wtof = Counter(wtof).most_common(args.max_word_num)
        needed_words = { w[0]: w[1] for w in wtof }
        # print('Preserve word num: %d. Examples: %s %s' % (len(needed_words), wtof[0][0], wtof[1][0]))


    itow = ['<pad>', '<unk>']
    wtoi = {'<pad>': 0, '<unk>': 1}
    count = 2
    glove['<pad>'] = np.zeros((word_dim, ))
    glove['<unk>'] = np.zeros((word_dim, ))
    missing_word_neighbors = {}

    print('Replace word string with word index......')
    if(args.data == 'duc2007'):
        cnn_data = Dataset(path='/data/c-liang/data/cnndaily_5w_100d.pkl')
        needed_words = cnn_data.wtoi
        wtoi = cnn_data.wtoi
        itow = cnn_data.itow
        for i in range(len(data)):
            for j in range(len(data[i])):
                for k in range(2): # 0: content, 1: summary
                    max_len = max([len(s) for s in data[i][j][k]]) # max length of sentences for padding
                    for l in range(len(data[i][j][k])): # l-th sentence
                        for m, word in enumerate(data[i][j][k][l]): # m-th word
                            if word not in wtoi:
                                word = '<unk>'
                            data[i][j][k][l][m] = wtoi[word]
                        data[i][j][k][l] += [0]*(max_len - len(data[i][j][k][l])) # padding l-th sentence
                    data[i][j][k] = np.asarray(data[i][j][k], dtype='int32')
                    length[i][j][k] = np.asarray(length[i][j][k], dtype='int32')
                    # np.array for all documents/summaries
                    # shape of each document/summary: (# sentence, max length)
    else:
        for i in range(len(data)):
            for j in range(len(data[i])):
                for k in range(2): # 0: content, 1: summary
                    max_len = max([len(s) for s in data[i][j][k]]) # max length of sentences for padding
                    for l in range(len(data[i][j][k])): # l-th sentence
                        for m, word in enumerate(data[i][j][k][l]): # m-th word
                            if word not in needed_words:
                                word = '<unk>'
                            elif word not in wtoi:
                                itow.append(word)
                                wtoi[word] = count
                                count += 1
                            #print(word)
                            data[i][j][k][l][m] = wtoi[word]
                            # Find neighbor vectors for those words not in glove
                            if word not in glove:
                                if word not in missing_word_neighbors:
                                    missing_word_neighbors[word] = []
                                for neighbor in data[i][j][k][l][m-5:m+6]: # window size: 10
                                    if neighbor in glove:
                                        missing_word_neighbors[word].append(glove[neighbor])
                        if(max_len > len(data[i][j][k][l])):
                            data[i][j][k][l] += [0]*int(max_len - len(data[i][j][k][l])) # padding l-th sentence
                    data[i][j][k] = np.asarray(data[i][j][k], dtype='int32')
                    length[i][j][k] = np.asarray(length[i][j][k], dtype='int32')
                    # np.array for all documents/summaries
                    # shape of each document/summary: (# sentence, max length)
    print('Calculate vectors for missing words by averaging neighbors......')
    #print(data)
    if(args.data == 'duc2007'):
        weight_matrix = cnn_data.weight
    else:
        for word in missing_word_neighbors:
            vectors = missing_word_neighbors[word]
            if len(vectors) > 0:
                glove[word] = sum(vectors) / len(vectors)
            else:
                glove[word] = np.zeros((word_dim, ))
        weight_matrix = np.vstack([glove[w] for w in itow])
    print('Shape of weight matrix:')
    print(weight_matrix.shape)

    print('Dumping......')
    #print(data[2][0][0], data[2][1][0])
    save_file = open(args.save_path, 'wb')
    pickle.dump(data, save_file)
    pickle.dump(length, save_file)
    pickle.dump(weight_matrix, save_file)
    pickle.dump(wtoi, save_file)
    pickle.dump(itow, save_file)
    save_file.close()


if __name__ == "__main__":
    main()
