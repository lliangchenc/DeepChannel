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
from utils import recursive_to_device, visualize_tensor

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
    for batch_iter, valid_batch in enumerate(data.gen_valid_minibatch()):
        sentenceEncoder.eval(); channelModel.eval()
        doc, sums, doc_len, sums_len = recursive_to_device(device, *valid_batch)
        num_sent_of_sum = sums[0].size(0)
        D = sentenceEncoder(doc, doc_len)
        l = D.size(0)
        
        selected_indexs = []
        for _ in range(num_sent_of_sum):
            probs = []
            for i in range(l):
                temp = [D[x] for x in selected_indexs]
                temp.append(D[i])
                temp_prob, addition = channelModel(D, torch.stack(temp))
                probs.append(temp_prob.item())
            best_index = np.argmax(probs)
            selected_indexs.append(best_index)

        valid_count += 1
        if(valid_count % 10 == 0):
            doc_matrix = doc.cpu().data.numpy()
            doc_len_arr = doc_len.cpu().data.numpy()
            golden_summ_matrix = sums[0].cpu().data.numpy()
            golden_summ_len_arr = sums_len[0].cpu().data.numpy()
            summ_matrix = torch.stack([doc[x] for x in selected_indexs]).cpu().data.numpy()
            summ_len_arr = torch.stack([doc_len[x] for x in selected_indexs]).cpu().data.numpy()
            #best_sent = " ".join([data.itow[x] for x in doc_matrix[best_index]][:doc_len_arr[best_index]])
            doc_ = ""
            for i in range(np.shape(doc_matrix)[0]):
                doc_ += str(i) + ": " + " ".join([data.itow[x] for x in doc_matrix[i]][:doc_len_arr[i]]) + "\n\n"

            golden_summ_ = ""
            for i in range(np.shape(golden_summ_matrix)[0]):
                golden_summ_ += str(i) + ": " + " ".join([data.itow[x] for x in golden_summ_matrix[i]][:golden_summ_len_arr[i]]) + "\n\n"
            
            summ_ = ""
            for i in range(np.shape(summ_matrix)[0]):
                summ_ += str(i) + ": " + " ".join([data.itow[x] for x in summ_matrix[i]][:summ_len_arr[i]]) + "\n\n"

            logging.info("\nsample case %d:\n\ndocument:\n\n%s\n\ngolden summary:\n\n%s\n\nmy summary:\n\n%s\n\n"%(valid_count, doc_, golden_summ_, summ_))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--SE-type', default='GRU', choices=['GRU', 'BiGRU', 'AVG'])
    parser.add_argument('--neg-case', default = 'max', choices=['max', 'random'])
    parser.add_argument('--neg-sample', default = 'mix', choices=['mix', 'delete', 'replace'])
    parser.add_argument('--word-dim', type=int, default=300, help='dimension of word embeddings')
    parser.add_argument('--hidden-dim', type=int, default=300, help='dimension of hidden units per layer')
    parser.add_argument('--num-layers', type=int, default=1, help='number of layers in LSTM/BiLSTM')
    parser.add_argument('--kernel-num', type=int, default=64, help='kernel num/ output dim in CNN')
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--margin', type=float, default=1, help='margin of hinge loss, must >= 0')

    parser.add_argument('--clip', type=float, default=.5, help='clip to prevent the too large grad')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='weight decay rate per batch')
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


