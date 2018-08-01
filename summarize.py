import torch
import time
import argparse
import random
import shutil
import os
from model.noisyChannel import ChannelModel
from model.sentence import SentenceEmbedding
from dataset.data import Dataset
import numpy as np
from utils import recursive_to_device, visualize_tensor, genSubset
from rouge import Rouge
from pyrouge.rouge import Rouge155
from train import rouge_atten_matrix
import copy
from tqdm import tqdm
from IPython import embed

def evalLead3(args):
    data = Dataset(path=args.data_path)
    Rouge_list, Rouge155_list = [], []
    Rouge155_obj = Rouge155(stem=True, tmp='./tmp2')
    for batch_iter, valid_batch in tqdm(enumerate(data.gen_test_minibatch()), total=data.test_size):
        doc, sums, doc_len, sums_len = valid_batch
        selected_indexs = range(min(doc.size(0), 3))
        doc_matrix = doc.data.numpy()
        doc_len_arr = doc_len.data.numpy()
        golden_summ_matrix = sums[0].data.numpy()
        golden_summ_len_arr = sums_len[0].data.numpy()
        doc_arr = []
        for i in range(np.shape(doc_matrix)[0]):
            temp_sent = " ".join([data.itow[x] for x in doc_matrix[i]][:doc_len_arr[i]])
            doc_arr.append(temp_sent)

        golden_summ_arr = []
        for i in range(np.shape(golden_summ_matrix)[0]):
            temp_sent = " ".join([data.itow[x] for x in golden_summ_matrix[i]][:golden_summ_len_arr[i]])
            golden_summ_arr.append(temp_sent)

        summ_matrix = torch.stack([doc[x] for x in selected_indexs]).data.numpy()
        summ_len_arr = torch.stack([doc_len[x] for x in selected_indexs]).data.numpy()
        
        summ_arr = []
        for i in range(np.shape(summ_matrix)[0]):
            temp_sent = " ".join([data.itow[x] for x in summ_matrix[i]][:summ_len_arr[i]])
            summ_arr.append(temp_sent)
        score_Rouge = Rouge().get_scores(" ".join(summ_arr), " ".join(golden_summ_arr))
        #score_Rouge155 = Rouge155_obj.score(summ_arr, {'A':golden_summ_arr})
        #score_Rouge155 = Rouge155_obj.score(' '.join(summ_arr), {'A':' '.join(golden_summ_arr)})
        Rouge_list.append(score_Rouge[0]['rouge-1']['f'])
        #Rouge155_list.append(score_Rouge155['rouge_1_f_score'])
        print(Rouge_list[-1])
        #print(Rouge155_list[-1])
        #embed()
        #print('-----')
    print('='*60)
    print(np.mean(Rouge_list))
    #print(np.mean(Rouge155_list))


def genSentences(args):
    np.set_printoptions(threshold=1e10) 
    print('Loading data......')
    data = Dataset(path=args.data_path)
    print('Building model......')
    args.num_words = len(data.weight) # number of words
    sentenceEncoder = SentenceEmbedding(**vars(args))
    args.se_dim = sentenceEncoder.getDim() # sentence embedding dim
    channelModel = ChannelModel(**vars(args))
    print('Initializing word embeddings......')
    sentenceEncoder.word_embedding.weight.data.set_(data.weight)
    sentenceEncoder.word_embedding.weight.requires_grad = False
    print('Fix word embeddings')
    device = torch.device('cuda' if args.cuda else 'cpu')
    if args.cuda:
        print('Transfer models to cuda......')
    sentenceEncoder, channelModel = sentenceEncoder.to(device), channelModel.to(device)
    identityMatrix = torch.eye(100).to(device)

    print('Initializing optimizer and summary writer......')
    params = [p for p in sentenceEncoder.parameters() if p.requires_grad] +\
            [p for p in channelModel.parameters() if p.requires_grad]

    sentenceEncoder.load_state_dict(torch.load(os.path.join(args.save_dir, 'se.pkl')))
    channelModel.load_state_dict(torch.load(os.path.join(args.save_dir, 'channel.pkl')))

    valid_count = 0
    Rouge_list, Rouge155_list = [], []
    Rouge_list_2, Rouge_list_l = [], []
    Rouge155_list_2, Rouge155_list_l = [], []
    total_score = None
    Rouge155_obj = Rouge155(n_bytes=75, stem=True, tmp='.tmp')
    best_rouge1_arr = []
    for batch_iter, valid_batch in tqdm(enumerate(data.gen_test_minibatch()), total = data.test_size):
        #print(valid_count)
        sentenceEncoder.eval(); channelModel.eval()
        doc, sums, doc_len, sums_len = recursive_to_device(device, *valid_batch)
        num_sent_of_sum = sums[0].size(0)
        D = sentenceEncoder(doc, doc_len)
        l = D.size(0)
        doc_matrix = doc.cpu().data.numpy()
        doc_len_arr = doc_len.cpu().data.numpy()
        golden_summ_matrix = sums[0].cpu().data.numpy()
        golden_summ_len_arr = sums_len[0].cpu().data.numpy()
        
        candidate_indexes = [i for i in range(len(doc_len_arr)) if doc_len_arr[i] >=0 and doc_len_arr[i] <= 10000]
        
        if(len(candidate_indexes) < 3):
            continue
        
        doc_ = ""
        doc_arr = []
        for i in range(np.shape(doc_matrix)[0]):
            temp_sent = " ".join([data.itow[x] for x in doc_matrix[i]][:doc_len_arr[i]])
            doc_ += str(i) + ": " + temp_sent + "\n\n"
            doc_arr.append(temp_sent)

        golden_summ_ = ""
        golden_summ_arr = []
        for i in range(np.shape(golden_summ_matrix)[0]):
            temp_sent = " ".join([data.itow[x] for x in golden_summ_matrix[i]][:golden_summ_len_arr[i]])
            golden_summ_ += str(i) + ": " + temp_sent + "\n\n"
            golden_summ_arr.append(temp_sent)

        selected_indexs = []
        #probs_arr = []

        if args.method == 'iterative':
            for _ in range(3):
                probs = np.zeros([l]) - 100000
                for i in candidate_indexes:
                    temp = [D[x] for x in selected_indexs]
                    temp.append(D[i])
                    temp_prob, addition = channelModel(D, torch.stack(temp))
                    probs[i] = temp_prob.item()
                    #print(i, selected_indexs, probs)
                #probs_arr.append(probs)
                best_index = np.argmax(probs)
                while(best_index in selected_indexs):
                    probs[best_index] = -100000
                    best_index = np.argmax(probs)
                selected_indexs.append(best_index)

        if(args.method == 'iterative-delete'):
            current_sent_set = range(l)
            best_index = -1
            doc_rouge_matrix = rouge_atten_matrix(doc_arr, doc_arr)
            for i_ in range(num_sent_of_sum):
                D_ = torch.stack([D[x] for x in current_sent_set])
                probs = []
                print(i_, current_sent_set)
                for i in current_sent_set:
                    temp_prob, addition = channelModel(D_, torch.stack([D[i]]))
                    probs.append(temp_prob.item())
                best_index = np.argmax(probs)
                print(current_sent_set[best_index])
                selected_indexs.append(current_sent_set[best_index])
                temp = []
                for i in current_sent_set:
                    if(doc_rouge_matrix[current_sent_set[best_index], i] < 0.9):
                        temp.append(i)
                if(len(temp) == 0):
                    break
                current_sent_set = temp

        #if(args.method == 'random-replace'):
        

        probs_arr = []
        if args.method == 'top-k-simple':
            for i in range(3):
                temp_prob, addition = channelModel(D, torch.stack([D[i]]))
                probs_arr.append(temp_prob.item())
            for _ in range(num_sent_of_sum):
                best_index = np.argmax(probs_arr)
                probs_arr[best_index] = - 1000000
                selected_indexs.append(best_index)

        if args.method == 'top-k':
            k_subset = genSubset(range(l), 3)
            probs = []
            for subset in k_subset:
                temp_prob, addition = channelModel(D, torch.stack([D[i] for i in subset]))
                probs.append(temp_prob.item())
            index = np.argmax(probs)
            selected_indexs = k_subset[index]

        if args.method == 'random':
            selected_indexs = random.sample(range(l), min(3, l))
        
        summ_matrix = torch.stack([doc[x] for x in selected_indexs]).cpu().data.numpy()
        summ_len_arr = torch.stack([doc_len[x] for x in selected_indexs]).cpu().data.numpy()
        
        summ_ = ""
        summ_arr = []
        for i in range(np.shape(summ_matrix)[0]):
            temp_sent = " ".join([data.itow[x] for x in summ_matrix[i]][:summ_len_arr[i]])
            summ_ += str(i) + ": " + temp_sent + "\n\n"
            summ_arr.append(temp_sent)
        '''
        best_rouge_summ_arr = []
        for s in golden_summ_arr:
            temp = []
            for d in doc_arr:
                temp.append(Rouge().get_scores(s, d)[0]['rouge-1']['f'])
            index = np.argmax(temp)
            best_rouge_summ_arr.append(doc_arr[index])
        '''
        #print("\nsample case %d:\n\ndocument:\n\n%s\n\ngolden summary:\n\n%s\n\nmy summary:\n\n%s\n\n"%(valid_count, doc_, golden_summ_, summ_))
        #print("PROB_ARR: ", str(probs_arr))
        #print(rouge_atten_matrix(doc_arr, golden_summ_arr))
        #print(rouge_atten_matrix(doc_arr, summ_arr))
        
        score_Rouge = Rouge().get_scores(" ".join(summ_arr), " ".join(golden_summ_arr))
        #score_Rouge155 = Rouge155_obj.score(summ_arr, {'A':golden_summ_arr})
        
        Rouge_list.append(score_Rouge[0]['rouge-1']['f'])
        Rouge_list_2.append(score_Rouge[0]['rouge-2']['f'])
        Rouge_list_l.append(score_Rouge[0]['rouge-l']['f'])

        #os.system("clear")
        print(Rouge_list[-1], Rouge_list_2[-1], Rouge_list_l[-1])
        #print(Rouge155_list[-1], Rouge155_list_2[-1], Rouge155_list_l[-1])
        '''
        if total_score is None:
            total_score = score_Rouge155
        else:
            for k in score_Rouge155:
                total_score[k] += score_Rouge155[k]
        valid_count += 1
        '''

    print('='*60)
    #for k in total_score:
    #    total_score[k] /= valid_count
    #print(total_score)
    print(np.mean(Rouge_list), np.mean(Rouge_list_2), np.mean(Rouge_list_l))
    #print(np.mean(Rouge155_list), np.mean(Rouge155_list_2), np.mean(Rouge155_list_l))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--SE-type', default='GRU', choices=['GRU', 'BiGRU', 'AVG'])
    parser.add_argument('--neg-case', default = 'max', choices=['max', 'random'])
    parser.add_argument('--neg-sample', default = 'mix', choices=['mix', 'delete', 'replace'])
    parser.add_argument('--method', default = 'random', choices=['random', 'top-k-simple', 'top-k', 'iterative', 'iterative-delete', 'lead-3'])
    parser.add_argument('--word-dim', type=int, default=300, help='dimension of word embeddings')
    parser.add_argument('--hidden-dim', type=int, default=300, help='dimension of hidden units per layer')
    parser.add_argument('--num-layers', type=int, default=1, help='number of layers in LSTM/BiLSTM')
    parser.add_argument('--kernel-num', type=int, default=64, help='kernel num/ output dim in CNN')
    parser.add_argument('--dropout', type=float, default=0)

    parser.add_argument('--cuda', action='store_true', default=True)

    parser.add_argument('--data-path', required=True, help='pickle file obtained by dataset dump or datadir for torchtext')
    parser.add_argument('--save-dir', type=str, help='path to save checkpoints and logs')
    args = parser.parse_args()
    return args


def prepare():
    args = parse_args()
    for k, v in vars(args).items():
        print(k+':'+str(v))
    return args

def main():
    args = prepare()
    if args.method == 'lead-3':
        evalLead3(args)
    else:
        genSentences(args)

if __name__ == "__main__":
    main()


