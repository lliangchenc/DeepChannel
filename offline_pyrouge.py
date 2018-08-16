# python3.6 offline_pyrouge.py --data-path /data/c-liang/data/cnndaily_5w_100d.pkl --save-path /data/sjx/Summarization-Exp/offline_pyrouge_max_index.json

import multiprocessing as mp
from multiprocessing import Pool
from rouge import Rouge
from pyrouge import Rouge155
from dataset.data import Dataset
import numpy as np
import argparse
from tqdm import tqdm
from tempfile import mkdtemp
import time
from IPython import embed
import json

r1 = Rouge()
topn = 5 #-1 means take all document sentences as candidates

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', required=True, help='pickle file obtained by dataset dump or datadir for torchtext')
    parser.add_argument('--save-path', type=str, required=True, help='path to save results')
    args = parser.parse_args()
    return args


def job(doc_summ_pair):
    tmp = mkdtemp()
    r2 = Rouge155(fast=True, tmp=tmp)
    doc, summ = doc_summ_pair
    max_pyrouge_index = []
    for s in summ:        
        if topn == -1 or topn > len(doc):
            candidate_index = list(range(len(doc)))
        else:
            rouge_scores = []
            for d in doc:
                rouge_scores.append(r1.get_scores(d, s)[0]['rouge-1']['f'])
            candidate_index = np.argpartition(rouge_scores, -topn)[-topn:]
        pyrouge_scores = []
        for j in candidate_index:
            d = doc[j]
            pyrouge_scores.append(r2.score(d, {'A':s})['rouge_1_f_score'])
        max_index = int(candidate_index[np.argmax(pyrouge_scores)])
        max_pyrouge_index.append(max_index)
        # len = len(summs[i])
    r2.clear()
    return max_pyrouge_index



if __name__ == '__main__':
    mp.set_start_method('forkserver', force=True) # use fork server to take in charge of fork every time
    args = parse_args()
    data = Dataset(path=args.data_path, fraction=1)
    doc_summ_pairs = []
    # must not shuffle ! keep the sort for visiting by index
    for batch_iter, train_batch in tqdm(enumerate(data.gen_train_minibatch(shuffle=False))):
        doc, summ, doc_len, summ_len = train_batch
        doc, doc_len = doc.numpy(), doc_len.numpy()
        summ, summ_len = summ[0].numpy(), summ_len[0].numpy()
        doc_ = []
        summ_ = []
        for i in range(np.shape(doc)[0]):
            doc_.append(" ".join([data.itow[x] for x in doc[i]][:doc_len[i]]))
        for i in range(np.shape(summ)[0]):
            summ_.append(" ".join([data.itow[x] for x in summ[i]][:summ_len[i]]))
        doc_summ_pairs.append((doc_, summ_))

    print("Start working")
    tic = time.time()
    with Pool(50) as p:
        results = p.map(job, doc_summ_pairs)
    print('='*40)
    print(time.time() - tic)
    print('='*40)
    print("Writing to %s" % args.save_path)
    with open(args.save_path, 'w') as f:
        json.dump(results, f, indent=2)
    # embed()
