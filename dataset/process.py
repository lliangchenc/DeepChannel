import os
import re
import argparse
import numpy as np
import pickle
import spacy
import random
from tqdm import tqdm
from collections import Counter
from IPython import embed
random.seed(666)

pattern_of_num = re.compile(r'[0-9]+')
tokenizer = None
def sent_dealer(s):
    global tokenizer
    if tokenizer is None:
        tokenizer = spacy.load('en')
    words = [w.text for w in tokenizer(s) if w.text.strip()]
    words = ['zero' if pattern_of_num.match(w) else w for w in words]
    return words



#def read_official_split(data_dir):
#    data = [[], [], []] # train/valid/test
#    split_dirs = [os.path.join(data_dir, d) for d in ['train', 'valid', 'test']]
#    for i, dir in enumerate(split_dirs):
#        for f in tqdm(os.listdir(dir)):
#            f = open(os.path.join(dir, f)).read().replace('\n', ' ').split('@highlight') # (content, summary)
#            data[i].append(list(map(sent_dealer, [f[0], ' '.join(f[1:])])))
#    return data



def read_cnn_dailymail(data_type, data_dir):
    data = [[], [], []]
    length = [[], [], []] # length of each sentence, whose index is corresponding to data
    if data_type == 'cnn+dailymail':
        data_dir = data_dir.split(';')
        assert len(data_dir)==2
        fs = [os.path.join(data_dir[0], _) for _ in os.listdir(data_dir[0])] + [os.path.join(data_dir[1], _) for _ in os.listdir(data_dir[1])]
    elif data_type == 'cnn' or data_type == 'daily':
        assert ';' not in data_dir
        fs = [os.path.join(data_dir, _) for _ in os.listdir(data_dir)]
    random.shuffle(fs)
    split_idx = [0, int(len(fs)*0.8), int(len(fs)*0.9), len(fs)]  # split index for train:valid:test, ratio is 8:1:1
    for i in range(3):
        for f in tqdm(fs[split_idx[i]: split_idx[i+1]]):
            parts = open(f).read().split('@highlight')
            docu = [sent_dealer(s.strip()) for s in parts[0].split('\n') if s.strip()] 
            # document is list of sentences, sentence is list of word tokens
            summ = [sent_dealer(s.strip()) for s in parts[1:]]
            docu_len = [len(s) for s in docu]
            summ_len = [len(s) for s in summ]
            data[i].append([docu, summ])
            length[i].append([docu_len, summ_len])
            # return data, length
    return data, length


def read_duc2007(data_dir):
    pass


def main():
    datasets = {
            'cnn+dailymail': read_cnn_dailymail,
            'cnn': read_cnn_dailymail,
            'daily': read_cnn_dailymail,
            'duc2007': read_duc2007,
            }
    parser = argparse.ArgumentParser()
    parser.add_argument('--glove', default='/data/sjx/glove.840B.300d.py36.pkl', help='pickle file of glove300d')
    parser.add_argument('--data', default='cnn+dailymail', choices=datasets.keys())
    parser.add_argument('--data-dir', default='/data/share/cnn_stories/stories;/data/share/dailymail_stories/stories', help='If data=cnn+dailimail, then data-dir must contain two paths for cnn and dailymail seperated by ;.')
    parser.add_argument('--save-path', required=True)
    parser.add_argument('--max-word-num', type=int, default=150000)
    args = parser.parse_args()

    print('Loading glove......')
    glove = pickle.load(open(args.glove, 'rb'))
    word_dim = len(glove['the'])

    print('Reading data......')
    data, length = datasets[args.data](args.data, args.data_dir)
    print('train/valid/test: %d/%d/%d' % tuple([len(_) for _ in data]))

    print('Count word frequency......')
    wtof = {}
    for i in range(len(data)): # train/valid/test
        for j in range(len(data[i])): # j-th sample
            for k in range(2): # 0: content, 1: summary
                for l in range(len(data[i][j][k])): # l-th sentence
                    for word in data[i][j][k][l]:
                        wtof[word] = wtof.get(word, 0) + 1
    wtof = Counter(wtof).most_common(args.max_word_num)
    needed_words = { w[0]: w[1] for w in wtof }
    print('Preserve word num: %d. Examples: %s %s' % (len(needed_words), wtof[0][0], wtof[1][0]))


    itow = ['<pad>', '<unk>']
    wtoi = {'<pad>': 0, '<unk>': 1}
    count = 2
    glove['<pad>'] = np.zeros((word_dim, ))
    glove['<unk>'] = np.zeros((word_dim, ))
    missing_word_neighbors = {}

    print('Replace word string with word index......')
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
                        data[i][j][k][l][m] = wtoi[word]
                        # Find neighbor vectors for those words not in glove
                        if word not in glove:
                            if word not in missing_word_neighbors:
                                missing_word_neighbors[word] = []
                            for neighbor in data[i][j][k][l][m-5:m+6]: # window size: 10
                                if neighbor in glove:
                                    missing_word_neighbors[word].append(glove[neighbor])
                    data[i][j][k][l] += [0]*(max_len - len(data[i][j][k][l])) # padding l-th sentence
                data[i][j][k] = np.asarray(data[i][j][k], dtype='int32')
                length[i][j][k] = np.asarray(length[i][j][k], dtype='int32')
                # np.array for all documents/summaries
                # shape of each document/summary: (# sentence, max length)
    print('Calculate vectors for missing words by averaging neighbors......')
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
    save_file = open(args.save_path, 'wb')
    pickle.dump(data, save_file)
    pickle.dump(length, save_file)
    pickle.dump(weight_matrix, save_file)
    pickle.dump(wtoi, save_file)
    pickle.dump(itow, save_file)
    save_file.close()


if __name__ == "__main__":
    main()
