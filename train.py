import torch
import time
import argparse
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
logFormatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
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
from utils import recursive_to_device
#from IPython import embed


def trainChannelModel(args):
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
    print('Initializing optimizer and summary writer......')
    params = [p for p in sentenceEncoder.parameters() if p.requires_grad] +\
            [p for p in channelModel.parameters() if p.requires_grad]
    optimizer_class = {
            'adam': optim.Adam,
            'sgd': optim.SGD,
            }[args.optimizer]
    optimizer = optimizer_class(params=params, lr=args.lr, weight_decay=args.weight_decay)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='max', factor=0.5, patience=20, verbose=True)
    train_writer = SummaryWriter(os.path.join(args.save_dir, 'log', 'train'))
    tic = time.time()
    iter_count = 0
    loss_arr = []
    print('Start training......')
    for epoch_num in range(args.max_epoch):
        if args.anneal:
            channelModel.temperature = 1 - epoch_num * 0.99 / (args.max_epoch-1) # from 1 to 0.01 as the epoch_num increases
        for batch_iter, train_batch in enumerate(data.gen_train_minibatch()):
            sentenceEncoder.train(); channelModel.train()
            progress = epoch_num + batch_iter / data.train_size
            iter_count += 1
            doc, sums, doc_len, sums_len = recursive_to_device(device, *train_batch)
            if len(sums) == 2: # only one sentence in summary
                continue
            D = sentenceEncoder(doc, doc_len)
            S_bads = []
            S_good = sentenceEncoder(sums[0], sums_len[0])
            S_bads = [sentenceEncoder(s, s_l) for s, s_l in zip(sums[1:], sums_len[1:])] # TODO so many repetitions
            good_prob = channelModel(D, S_good)
            bad_probs = [channelModel(D, S_bad) for S_bad in S_bads]
            ########### hinge loss ############
            bad_index = np.argmax([p.item() for p in bad_probs])
            loss = bad_probs[bad_index] - 1.1 * good_prob
            #loss = bad_probs[bad_index]
            #print(bad_probs, good_prob)
            #loss = - good_prob
            bad_prob_value = bad_probs[bad_index].item()
            good_prob_value = good_prob.item()
            loss_value = loss.item()
            if loss_value > -args.margin:
                optimizer.zero_grad()
                loss.backward()
                #print(param.grad)
                nn.utils.clip_grad_norm_(parameters=params, max_norm=args.clip)
                optimizer.step()
            ###################################
            # summary
            train_writer.add_scalar('loss', loss, iter_count)
            if(iter_count % 1000 == 0):
                for name, param in list(sentenceEncoder.named_parameters()) + list(channelModel.named_parameters()):
                    #print(param.grad)
                    if param.requires_grad and name not in ['word_embedding.weight']:
                        train_writer.add_histogram(name, param.clone().cpu().data.numpy(), iter_count)
                        train_writer.add_histogram(name+'/grad', param.grad.clone().cpu().data.numpy(), iter_count)
            #scheduler.step(valid_accuracy)
            # if (batch_iter+1) % (data.train_size / 100) == 0:
            logging.info('Epoch %.2f, loss: %.4f, bad_prob: %.4f, good_prob: %.4f' % (progress, loss_value, bad_prob_value, good_prob_value))
    torch.save(sentenceEncoder.state_dict(), os.path.join(args.save_dir, 'se.pkl'))
    torch.save(channelModel.state_dict(), os.path.join(args.save_dir, 'channel.pkl'))
    [rootLogger.removeHandler(h) for h in rootLogger.handlers if isinstance(h, logging.FileHandler)]


            



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--SE-type', default='GRU', choices=['GRU', 'BiGRU', 'AVG'])
    parser.add_argument('--word-dim', type=int, default=300, help='dimension of word embeddings')
    parser.add_argument('--hidden-dim', type=int, default=300, help='dimension of hidden units per layer')
    parser.add_argument('--num-layers', type=int, default=1, help='number of layers in LSTM/BiLSTM')
    parser.add_argument('--kernel-num', type=int, default=64, help='kernel num/ output dim in CNN')
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--margin', type=float, default=123, help='margin of hinge loss, must >= 0')
    
    parser.add_argument('--clip', type=float, default=0.5, help='clip to prevent the too large grad')
    parser.add_argument('--lr', type=float, default=.00001, help='initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='weight decay rate per batch')
    parser.add_argument('--max-epoch', type=int, default=5)
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--optimizer', default='adam', choices=['adam', 'sgd', 'adadelta'])
    parser.add_argument('--batch-size', type=int, default=1, help='batch size for training, not used now')
    parser.add_argument('--tune-word-embedding', action='store_true', help='specified to fine tune glove vectors')
    parser.add_argument('--anneal', action='store_true')
    parser.add_argument('--display', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--seed', type=int, default=666, help='random seed')
    parser.add_argument('--alpha', type=float, default=0.1, help='weight of regularization term')

    parser.add_argument('--data-path', required=True, help='pickle file obtained by dataset dump or datadir for torchtext')
    parser.add_argument('--save-dir', type=str, required=True, help='path to save checkpoints and logs')
    args = parser.parse_args()
    return args


def prepare():
    # dir preparation
    args = parse_args()
    if os.path.isdir(args.save_dir):
        shutil.rmtree(args.save_dir)
    os.mkdir(args.save_dir)
    # seed setting
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(args.seed)
    # make logging.info display into both shell and file
    rootLogger = logging.getLogger()
    fileHandler = logging.FileHandler(os.path.join(args.save_dir, 'stdout.log'))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    # args display
    for k, v in vars(args).items():
        logging.info(k+':'+str(v))
    return args

def main():
    args = prepare()
    trainChannelModel(args)


if __name__ == '__main__':
    main()

