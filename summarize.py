import torch
import time
import argparse
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
logFormatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
rootLogger = logging.getLogger()
import random
import shutil
import os
from model.noisyChannel import ChannelModel
from model.sentence import SentenceEmbedding
from dataset.data import Dataset
from torch import nn
from torch import nn, optim
from torch.autograd import Variable
import numpy as np
from tensorboardX import SummaryWriter
from utils import recursive_to_device, visualize_tensor, genSubset
from rouge import Rouge
from pyrouge import Rouge155
from dataset.rouge_not_a_wrapper import rouge_n
from train import rouge_atten_matrix
import copy
#from IPython import embed


def genSentences(args):
    np.set_printoptions(threshold=1e10) 
    print('Loading data......')
    data = Dataset(path=args.data_path)
    print('Building model......')
    args.num_words = len(data.weight) # number of words
    sentenceEncoder = SentenceEmbedding(**vars(args))
    args.se_dim = sentenceEncoder.getDim() # sentence embedding dim
    channelModel = ChannelModel(**vars(args))
    logging.info(sentenceEncoder)
    logging.info(channelModel)
    print('Initializing word embeddings......')
    sentenceEncoder.word_embedding.weight.data.set_(data.weight)
    if not args.tune_word_embedding:
        sentenceEncoder.word_embedding.weight.requires_grad = False
        print('Fix word embeddings')
    else:
        print('Tune word embeddings')
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
    rouge_arr = [[],[],[]]
    another_rouge_arr = []
    best_rouge1_arr = []
    for batch_iter, valid_batch in enumerate(data.gen_valid_minibatch()):
        if(not(valid_count % 100 == 1)):
            valid_count += 1
            continue
        print(valid_count)
        sentenceEncoder.eval(); channelModel.eval()
        doc, sums, doc_len, sums_len = recursive_to_device(device, *valid_batch)
        num_sent_of_sum = sums[0].size(0)
        D = sentenceEncoder(doc, doc_len)
        l = D.size(0)
        if(l < num_sent_of_sum):
            continue
        doc_matrix = doc.cpu().data.numpy()
        doc_len_arr = doc_len.cpu().data.numpy()
        golden_summ_matrix = sums[0].cpu().data.numpy()
        golden_summ_len_arr = sums_len[0].cpu().data.numpy()

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
        probs_arr = []

        if args.method == 'iterative':
            for _ in range(num_sent_of_sum):
                probs = []
                for i in range(l):
                    temp = [D[x] for x in selected_indexs]
                    temp.append(D[i])
                    temp_prob, addition = channelModel(D, torch.stack(temp))
                    probs.append(temp_prob.item())
                    #print(i, selected_indexs, probs)
                probs_arr.append(probs)
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
        


        if args.method == 'top-k-simple':
            for i in range(l):
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
            selected_indexs = random.sample(range(l), min(num_sent_of_sum, l))

        if args.method == 'lead-3':
            if(l < 3):
                selected_indexs = range(l)
            else:
                selected_indexs = [0,1,2]
        
        summ_matrix = torch.stack([doc[x] for x in selected_indexs]).cpu().data.numpy()
        summ_len_arr = torch.stack([doc_len[x] for x in selected_indexs]).cpu().data.numpy()
        
        summ_ = ""
        summ_arr = []
        for i in range(np.shape(summ_matrix)[0]):
            temp_sent = " ".join([data.itow[x] for x in summ_matrix[i]][:summ_len_arr[i]])
            summ_ += str(i) + ": " + temp_sent + "\n\n"
            summ_arr.append(temp_sent)
        
        best_rouge_summ_arr = []
        for s in golden_summ_arr:
            temp = []
            for d in doc_arr:
                temp.append(Rouge().get_scores(s, d)[0]['rouge-1']['f'])
            index = np.argmax(temp)
            best_rouge_summ_arr.append(doc_arr[index])

        logging.info("\nsample case %d:\n\ndocument:\n\n%s\n\ngolden summary:\n\n%s\n\nmy summary:\n\n%s\n\n"%(valid_count, doc_, golden_summ_, summ_))
        print("PROB_ARR: ", str(probs_arr))
        print(rouge_atten_matrix(doc_arr, golden_summ_arr))
        print(rouge_atten_matrix(doc_arr, summ_arr))
        score = Rouge().get_scores(" ".join(summ_arr), " ".join(golden_summ_arr))
        another_score = rouge_n(best_rouge_summ_arr, golden_summ_arr, 1)
        
        score_1 = Rouge155(n_words=100, average="sentence", stem=True).score_summary([sent.split() for sent in summ_arr],{'A':[sent.split() for sent in golden_summ_arr]})
        print(score_1)
        #logging.info("\nsample case %d:\n\ndocument:\n\n%s\n\ngolden summary:\n\n%s\n\nrouge summary:\n\n%s\n\n"%(valid_count, doc_, golden_summ_, "\n\n".join(best_rouge_summ_arr)))riiieturn
        rouge_arr[0].append(score[0]['rouge-1']['f'])
        rouge_arr[1].append(score[0]['rouge-2']['f'])
        rouge_arr[2].append(score[0]['rouge-l']['f'])
        another_rouge_arr.append(another_score[0])
        if(score_1['rouge_1_f_score'] > 0):
            another_rouge_arr.append(score_1['rouge_1_f_score'])
        else:
            return
        print("ROUGE: ",score, another_score)
        valid_count += 1
        #return
    print("ROUGE : ", np.mean(rouge_arr,axis = 1))
    print("ROUGE : ", np.max(rouge_arr,axis = 1))

    print("ROUGE : ", np.mean(another_rouge_arr))
    print("ROUGE : ", np.max(another_rouge_arr))

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
    parser.add_argument('--margin', type=float, default=1, help='margin of hinge loss, must >= 0')

    parser.add_argument('--max-epoch', type=int, default=100)
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--optimizer', default='adam', choices=['adam', 'sgd', 'adadelta'])
    parser.add_argument('--batch-size', type=int, default=1, help='batch size for training, not used now')
    parser.add_argument('--tune-word-embedding', action='store_true', help='specified to fine tune glove vectors')
    parser.add_argument('--anneal', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--seed', type=int, default=666, help='random seed')
    parser.add_argument('--alpha', type=float, default=0.1, help='weight of regularization term')

    parser.add_argument('--data-path', required=True, help='pickle file obtained by dataset dump or datadir for torchtext')
    parser.add_argument('--save-dir', type=str, required=True, help='path to save checkpoints and logs')
    parser.add_argument('--load-previous-model', action='store_true')
    parser.add_argument('--validation', action='store_true')
    args = parser.parse_args()
    return args


def prepare():
    # dir preparation
    args = parse_args()
    if not args.load_previous_model:
        if os.path.isdir(args.save_dir):
            shutil.rmtree(args.save_dir)
            os.mkdir(args.save_dir)
    # seed setting
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(args.seed)
    # make logging.info display into both shell and file
    fileHandler = logging.FileHandler(os.path.join(args.save_dir, 'stdout.log'))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    # args display
    for k, v in vars(args).items():
        logging.info(k+':'+str(v))
    return args

def main():
    args = prepare()
    genSentences(args)

if __name__ == "__main__":
    main()


