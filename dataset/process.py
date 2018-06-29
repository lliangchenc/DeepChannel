import os
import re
import argparse
import numpy as np
import pickle
import spacy
import random
from tqdm import tqdm
from collections import Counter
#from IPython import embed
import xml.etree.ElementTree as et

random.seed(666)

pattern_of_num = re.compile(r'[0-9]+')
nlp = None
def doc_dealer(d):
    global nlp
    if nlp is None:
        nlp = spacy.load('en')
    words = [['zero' if pattern_of_num.match(w.text) else w.text for w in s if w.text.strip()] \
            for s in nlp(d.lower()).sents if s.text.strip()]
    return words
    # return a list of sentences, sentence is list of words



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
        print(data_dir)
        assert ';' not in data_dir
        fs = [os.path.join(data_dir, _) for _ in os.listdir(data_dir)]
    random.shuffle(fs)
    split_idx = [0, int(len(fs)*0.8), int(len(fs)*0.9), len(fs)]  # split index for train:valid:test, ratio is 8:1:1
    for i in range(3):
        for f in tqdm(fs[split_idx[i]: split_idx[i+1]]):
            parts = open(f).read().split('@highlight')
            docu = doc_dealer(parts[0]) 
            summ = doc_dealer('.'.join(parts[1:]) + '.')
            docu_ = []
            summ_ = []
            for sent in docu:
                if(len(sent) > 5 and not(sent[-1] == ":" or sent[-1] == "--")):
                    docu_.append(sent)
            for sent in summ:
                if(len(sent) > 3 and not(sent[-1] == ":" or sent[-1] == "--")):
                    summ_.append(sent)
            if len(docu_)==0 or len(summ_)==0:
                continue
            docu_len = [len(s) for s in docu_]
            summ_len = [len(s) for s in summ_]
            data[i].append([docu_, summ_])
            length[i].append([docu_len, summ_len])
            # return data, length
    return data, length


def read_duc2007(data_dir):
    duc2007_doc_dir = data_dir + "/main/"
    duc2007_eval_dir = data_dir + "/mainEval/mainEval/ROUGE/models/"
    doc_dirs = [duc2007_doc_dir + x + "/" for x in os.listdir(duc2007_doc_dir)]

    data = [[], [], []]
    length = [[], [], []]

    raw_data = []
    d_length = []
    raw_summ = [[],[],[],[]]
    s_length = [[],[],[],[]]
    for d in doc_dirs:
        filenames = os.listdir(d)
        temp = []
        for filename in filenames:
            content = open(d + filename).read().replace("&","")
            root = et.fromstring(content)
            raw = "".join([x.text.replace("\n", "").replace("\t", "") for x in root.findall("BODY/TEXT/P")])
            doc = [sent_dealer(x.strip(" ")) for x in raw.split(".")]
            
        raw_data.append(doc)
        d_length.append([len(x) for x in doc])

        temp = []
        temp_l = []
        for filename in os.listdir(duc2007_eval_dir):
            if(filename[:5] == prefix):
                raw = open(duc2007_eval_dir + filename).read().replace("\n")
                doc = [sent_dealer(x.strip(" ")) for x in raw.split(".")]
                temp.append(open(duc2007_eval_dir + filename).read().replace("\n"))
                temp_l.append([len(x) for x in temp[-1]])

        for i in range(4):
            raw_summ[i].append(temp[i])
            s_length[i].append(temp_l[i])

    data[2].append(raw_data).extend(raw_summ)
    length[2].append(d_length).extend(s_length)

    return data, length


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
