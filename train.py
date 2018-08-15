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
import heapq
from model.noisyChannel import ChannelModel
from model.sentence import SentenceEmbedding
from dataset.data import Dataset
from torch import nn
from torch import nn, optim
from torch.autograd import Variable
from rouge import Rouge
from pyrouge import Rouge155
import numpy as np
from tensorboardX import SummaryWriter
from utils import recursive_to_device, visualize_tensor, genPowerSet
#from IPython import embed

Rouge155_obj = Rouge155(stem=True, tmp='./tmp1')

def rouge_atten_matrix(doc, summ):
    doc_len = len(doc)
    summ_len = len(summ)
    temp_mat = np.zeros([doc_len, summ_len])
    for i in range(doc_len):
        for j in range(summ_len):
            #temp_mat[i, j] = Rouge155_obj.score([doc[i]], {'A':[summ[j]]})['rouge_1_f_score']
            temp_mat[i, j] = Rouge().get_scores(doc[i], summ[j])[0]['rouge-1']['f']
            #                 + Rouge().get_scores(doc[i], summ[j])[0]['rouge-2']['f']\
            #                 + Rouge().get_scores(doc[i], summ[j])[0]['rouge-l']['f']
    return temp_mat

def pyrouge_atten_matrix(doc, summ):
    doc_len = len(doc)
    summ_len = len(summ)
    temp_mat = np.zeros([doc_len, summ_len])
    global Rouge155_obj
    for i in range(doc_len):
        for j in range(summ_len):
            temp_mat[i, j] = Rouge155_obj.score([doc[i]], {'A':[summ[j]]})['rouge_1_f_score']
    return temp_mat

def trainChannelModel(args):
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
    optimizer_class = {
            'adam': optim.Adam,
            'sgd': optim.SGD,
            'adadelta': optim.Adadelta,
            }[args.optimizer]
    optimizer = optimizer_class(params=params, lr=args.lr, weight_decay=args.weight_decay)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=[5,10,20,30],gamma = 0.5)
    train_writer = SummaryWriter(os.path.join(args.save_dir, 'log', 'train'))
    tic = time.time()
    iter_count = 0
    loss_arr = []
    valid_loss = 0
    valid_all_loss = 0
    valid_acc = 0
    valid_all_acc = 0
    print('Start training......')
    #print(data.itow[123])
    if(args.load_previous_model):
        sentenceEncoder.load_state_dict(torch.load(os.path.join(args.save_dir, 'se.pkl')))
        channelModel.load_state_dict(torch.load(os.path.join(args.save_dir, 'channel.pkl')))

    if(args.validation):
        validate(data, sentenceEncoder, channelModel, device, args)
        return 0
    try:
        os.mkdir(os.path.join(args.save_dir, "checkpoints"))
    except:
        pass

    for epoch_num in range(args.max_epoch):
        scheduler.step()
        if args.anneal:
            channelModel.temperature = 1 - epoch_num * 0.99 / (args.max_epoch-1) # from 1 to 0.01 as the epoch_num increases

        if(epoch_num % 100 == 90):
            valid_loss, valid_all_loss, valid_acc, valid_all_acc, rouge_score = validate(data, sentenceEncoder, channelModel, device, args)
            train_writer.add_scalar('validation/loss', valid_loss, epoch_num)
            train_writer.add_scalar('validation/all_loss', valid_all_loss, epoch_num)
            train_writer.add_scalar('validation/acc', valid_acc, epoch_num)
            train_writer.add_scalar('validation/all_acc', valid_all_acc, epoch_num)
            train_writer.add_scalar('validation/rouge', rouge_score, epoch_num)
        eq = 0
        for batch_iter, train_batch in enumerate(data.gen_train_minibatch()):
            t1 = time.time()
            sentenceEncoder.train(); channelModel.train()
            progress = epoch_num + batch_iter / data.train_size
            iter_count += 1
            doc, sums, doc_len, sums_len = recursive_to_device(device, *train_batch)
            num_sent_of_sum = sums[0].size(0)
            if num_sent_of_sum == 1: # if delete, summary should have more than one sentence
                continue
            D = sentenceEncoder(doc, doc_len)
            S_good = sentenceEncoder(sums[0], sums_len[0])
            neg_sent_embed = sentenceEncoder(sums[1], sums_len[1])

            l = S_good.size(0)   

            ## train sample selection strategy:
            ## 0. good case: rouge max, bad case: random
            ## 1. good case: origin, bad case: random
            ## 2. good case: rouge max, bad case: one ranking in the second half
            ## 3. good case: origin, bad case: one ranking in the second half

            S_bads = []
            doc_matrix = doc.cpu().data.numpy()
            doc_len_arr = doc_len.cpu().data.numpy()
            summ_matrix = sums[0].cpu().data.numpy()
            summ_len_arr = sums_len[0].cpu().data.numpy()
            doc_ = []
            summ_ = []
            top_k_indexes = []
            pyrouge_scores = []
            for i in range(np.shape(doc_matrix)[0]):
                doc_.append(" ".join([data.itow[x] for x in doc_matrix[i]][:doc_len_arr[i]]))

            index = random.randint(0, l - 1) 
            summ_.append(" ".join([data.itow[x] for x in summ_matrix[index]][:summ_len_arr[index]]))
            
            atten_mat = rouge_atten_matrix(summ_, doc_)
            tmp_atten = atten_mat[0]
            t2 = time.time()
            for _ in range(2):
                best_index = np.argmax(tmp_atten)
                tmp_atten[best_index] = -1.
                top_k_indexes.append(best_index)
                pyrouge_scores.append(pyrouge_atten_matrix(summ_, [doc_[best_index]])[0][0])
                
            best_index = top_k_indexes[np.argmax(pyrouge_scores)]
            t3 = time.time()
            worse_indexes = []
            if args.train_sample / 2 == 0:
                #worse_index = random.randint(0, D.size(0) - 1)
                worse_indexes = random.sample(range(D.size(0)), min(D.size(0), 1))
            else:
                ind = random.randint(1, int(D.size(0) / 2) + 1)
                worse_indexes = [list(map(list(atten_mat[0]).index, heapq.nsmallest(ind, list(atten_mat[0]))))[-1]]

            temp_good = []
            for i in range(l):
                if(not i == index):
                    temp_good.append(S_good[i])
                else:
                    temp_good.append(D[best_index])

            if args.train_sample % 2 == 0:
                S_good = torch.stack(temp_good)

            for worse_index in worse_indexes:
                temp_bad = []
                for i in range(l):
                    if not i == index:
                        temp_bad.append(S_good[i])
                    else:
                        temp_bad.append(D[worse_index])
                S_bads.append(torch.stack(temp_bad))
            # prob calculation
            good_prob, addition = channelModel(D, S_good)
            good_prob_vector, good_attention_weight = addition['prob_vector'], addition['att_weight']
            bad_probs, bad_probs_vector = [], []
            bad_prob = 0.
            if(args.neg_case == 'max'):
                for S_bad in S_bads:
                    bad_prob, addition = channelModel(D, S_bad)
                    bad_probs.append(bad_prob)
                    bad_probs_vector.append(addition['prob_vector'])
                bad_index = np.argmax([p.item() for p in bad_probs])
                bad_prob = bad_probs[bad_index]
            elif(args.neg_case == 'random'):
                bad_index = random.randint(0, len(S_bads) - 1)
                bad_prob, addition = channelModel(D, S_bads[bad_index])
                bad_probs_vector.append(addition['prob_vector'])
            elif(args.neg_case == 'mean'):
                tmp = 0.
                for S_bad in S_bads:
                    bad_prob, addition = channelModel(D, S_bad)
                    tmp = bad_prob + tmp
                bad_prob = tmp / (len(S_bads) + 0.)

            else:
                bad_num = min(8, len(S_bads))
                S_bads_sample = random.sample(S_bads, bad_num)
                for S_bad in S_bads_sample:
                    bad_prob, addition = channelModel(D, S_bad)
                    bad_probs.append(bad_prob)
                    bad_probs_vector.append(addition['prob_vector'])
                bad_index = np.argmax([p.item() for p in bad_probs])
                bad_prob = bad_probs[bad_index]



            ########### loss ############
            loss_prob_term = bad_prob - good_prob
            n, m = good_attention_weight.size()
            regulation_term = torch.norm(torch.mm(good_attention_weight.t(), good_attention_weight) - n/m * torch.eye(m).to(device), 2)
            regulation_weight = 0.
            if(epoch_num < 30):
                regulation_weight = args.alpha * (1 - epoch_num / 30.)
            #loss = loss_prob_term
            loss = loss_prob_term + args.alpha * regulation_term
            #print(good_prob.item(), ', '.join([str(p.item()) for p in bad_probs]))
            if loss_prob_term.item() > -args.margin:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(parameters=params, max_norm=args.clip)
                #nn.utils.clip_grad_value_(parameters=params, clip_value=args.clip)
                optimizer.step()
            ###################################
            # summary
            # train_writer.add_scalar('loss/total', loss, iter_count)
            # train_writer.add_scalar('loss/prob_term', loss_prob_term, iter_count)
            # train_writer.add_scalar('loss/regulation_term', regulation_term, iter_count)
            # train_writer.add_scalar('prob/good_prob', good_prob, iter_count)
            # if(iter_count % 1000 == 0):
            #    train_writer.add_histogram('good_prob_vector', good_prob_vector.clone().cpu().data.numpy(), iter_count)
            #    for name, param in list(sentenceEncoder.named_parameters()) + list(channelModel.named_parameters()):
            #        if param.requires_grad and name not in ['word_embedding.weight']:
            #            train_writer.add_histogram(name, param.clone().cpu().data.numpy(), iter_count)
            #            train_writer.add_histogram(name+'/grad', param.grad.clone().cpu().data.numpy(), iter_count)
            #scheduler.step(valid_acc)
            if iter_count % 100 == 0:
                logging.info('Epoch %.2f, loss_prob: %.4f, bad_prob: %.4f, good_prob: %.4f, regulation_value: %.4f' % (progress, loss_prob_term.item(), bad_prob.item(), good_prob.item(), regulation_term.item()))
            if args.debug and iter_count % 100 == 0:
                print(visualize_tensor(good_prob_vector))
                print('-'*33)
                for i in range(len(bad_probs)):
                    print(visualize_tensor(bad_probs_vector[i]))
                    if i == bad_index:
                        print(' @#@')
                print('-'*33)
                print(visualize_tensor(good_attention_weight))
                #print('-'*33)
                #print(visualize_tensor(torch.norm(S_good, p=2, dim=0)))
                print('='*66)
            t4 = time.time()
            print(t2-t1, t3-t2, t4-t3)
        if(epoch_num % 1 == 0):
            try:
                os.mkdir(os.path.join(args.save_dir, 'checkpoints/'+str(epoch_num)))
            except:
                pass
            torch.save(sentenceEncoder.state_dict(), os.path.join(args.save_dir, 'checkpoints/'+ str(epoch_num) + '/se.pkl'))
            torch.save(channelModel.state_dict(), os.path.join(args.save_dir, 'checkpoints/'+ str(epoch_num) + '/channel.pkl'))
    [rootLogger.removeHandler(h) for h in rootLogger.handlers if isinstance(h, logging.FileHandler)]


def validate(data_, sentenceEncoder_, channelModel_, device_, args):
    neg_count = 0
    valid_iter_count = 0
    all_neg_count = 0
    sent_count = 0
    loss_arr = []
    all_loss_arr = []
    Rouge_list = []

    for batch_iter, valid_batch in enumerate(data_.gen_test_minibatch()):
        if not(batch_iter % 10 == 0):
            continue
        sentenceEncoder_.eval(); channelModel_.eval()
        doc, sums, doc_len, sums_len = recursive_to_device(device_, *valid_batch)
        num_sent_of_sum = sums[0].size(0)
        D = sentenceEncoder_(doc, doc_len)
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
            temp_sent = " ".join([data_.itow[x] for x in doc_matrix[i]][:doc_len_arr[i]])
            doc_ += str(i) + ": " + temp_sent + "\n\n"
            doc_arr.append(temp_sent)

        golden_summ_ = ""
        golden_summ_arr = []
        for i in range(np.shape(golden_summ_matrix)[0]):
            temp_sent = " ".join([data_.itow[x] for x in golden_summ_matrix[i]][:golden_summ_len_arr[i]])
            golden_summ_ += str(i) + ": " + temp_sent + "\n\n"
            golden_summ_arr.append(temp_sent)

        selected_indexs = []
        probs_arr = []

        for _ in range(num_sent_of_sum):
            probs = []
            for i in range(l):
                temp = [D[x] for x in selected_indexs]
                temp.append(D[i])
                temp_prob, addition = channelModel_(D, torch.stack(temp))
                probs.append(temp_prob.item())
            probs_arr.append(probs)
            best_index = np.argmax(probs)
            while(best_index in selected_indexs):
                probs[best_index] = -100000
                best_index = np.argmax(probs)
            selected_indexs.append(best_index)
        summ_matrix = torch.stack([doc[x] for x in selected_indexs]).cpu().data.numpy()
        summ_len_arr = torch.stack([doc_len[x] for x in selected_indexs]).cpu().data.numpy()
        
        summ_ = ""
        summ_arr = []
        for i in range(np.shape(summ_matrix)[0]):
            temp_sent = " ".join([data_.itow[x] for x in summ_matrix[i]][:summ_len_arr[i]])
            summ_ += str(i) + ": " + temp_sent + "\n\n"
            summ_arr.append(temp_sent)
        
        best_rouge_summ_arr = []
        for s in golden_summ_arr:
            temp = []
            for d in doc_arr:
                temp.append(Rouge().get_scores(s, d)[0]['rouge-1']['f'])
            index = np.argmax(temp)
            best_rouge_summ_arr.append(doc_arr[index])

        score_Rouge = Rouge().get_scores(" ".join(summ_arr), " ".join(golden_summ_arr))
        Rouge_list.append(score_Rouge[0]['rouge-1']['f'])
    
    rouge_score = np.mean(Rouge_list)
    print("ROUGE 1/100 sample : ", rouge_score)

    for batch_iter, valid_batch in enumerate(data_.gen_valid_minibatch()):
        if not(batch_iter % 100 == 0):
            continue
        sentenceEncoder_.eval(); channelModel_.eval()
        valid_iter_count += 1
        doc, sums, doc_len, sums_len = recursive_to_device(device_, *valid_batch)
        num_sent_of_sum = sums[0].size(0)
        if num_sent_of_sum == 1: # if delete, summary should have more than one sentence
            continue
        D = sentenceEncoder_(doc, doc_len)
        S_good = sentenceEncoder_(sums[0], sums_len[0])
        neg_sent_embed = sentenceEncoder_(sums[1], sums_len[1])

        l = S_good.size(0)        
        S_bads = []
        
        doc_matrix = doc.cpu().data.numpy()
        doc_len_arr = doc_len.cpu().data.numpy()
        summ_matrix = sums[0].cpu().data.numpy()
        summ_len_arr = sums_len[0].cpu().data.numpy()
        doc_ = []
        summ_ = []
        for i in range(np.shape(doc_matrix)[0]):
            doc_.append(" ".join([data_.itow[x] for x in doc_matrix[i]][:doc_len_arr[i]]))

        index = random.randint(0, l - 1) 
        summ_.append(" ".join([data_.itow[x] for x in summ_matrix[index]][:summ_len_arr[index]]))
         
        atten_mat = rouge_atten_matrix(summ_, doc_)
        best_index = np.argmax(atten_mat[0])
        worst_index= np.argmin(atten_mat[0])
        temp_good = []
        temp_bad = []
        for i in range(l):
            if(not i == index):
                temp_good.append(S_good[i])
                temp_bad.append(S_good[i])
            else:
                temp_good.append(D[best_index])
                temp_bad.append(D[worst_index])
        S_good = torch.stack(temp_good)
        S_bads.append(torch.stack(temp_bad))
        # prob calculation
        good_prob, addition = channelModel_(D, S_good)
        good_prob_vector, good_attention_weight = addition['prob_vector'], addition['att_weight']
        bad_probs, bad_probs_vector = [], []
        for S_bad in S_bads:
            bad_prob, addition = channelModel_(D, S_bad)
            bad_probs.append(bad_prob)
            bad_probs_vector.append(addition['prob_vector'])
        bad_index = np.argmax([p.item() for p in bad_probs])
        bad_prob = bad_probs[bad_index]
        ########### loss ############
        loss_prob_term = bad_prob - good_prob
        loss = loss_prob_term.item()
        loss_arr.append(loss)
        for bad in bad_probs:
            all_loss_arr.append((bad - good_prob).item())
        if(args.visualize and valid_iter_count % 100 == 0):
            doc_matrix = doc.cpu().data.numpy()
            doc_len_arr = doc_len.cpu().data.numpy()
            summ_matrix = sums[0].cpu().data.numpy()
            summ_len_arr = sums_len[0].cpu().data.numpy()
            doc_ = ""
            for i in range(np.shape(doc_matrix)[0]):
                doc_ += str(i) + ": " + " ".join([data_.itow[x] for x in doc_matrix[i]][:doc_len_arr[i]]) + "\n\n"

            summ_ = ""
            for i in range(np.shape(summ_matrix)[0]):
                summ_ += str(i) + ": " + " ".join([data_.itow[x] for x in summ_matrix[i]][:summ_len_arr[i]]) + "\n\n"
            logging.info("\nsample case %d:\n\ndocument:\n\n%s\n\nsummary:\n\n%s\n\nattention matrix:\n\n%s\n\n"%(valid_iter_count, str(doc_), str(summ_), str(good_attention_weight.cpu().data.numpy())))
            
    valid_loss = float(np.mean(loss_arr))
    valid_all_loss = float(np.mean(all_loss_arr))
    valid_acc = (np.sum(np.int32(np.array(loss_arr) < 0)) + 0.) / len(loss_arr)
    valid_all_acc = (np.sum(np.int32(np.array(all_loss_arr) < 0)) + 0.) / len(all_loss_arr)
    logging.info("avg loss: %4f, avg all_loss: %4f, acc: %4f, all_acc: %4f" % (valid_loss, valid_all_loss, valid_acc, valid_all_acc))


    return valid_loss, valid_all_loss, valid_acc, valid_all_acc, rouge_score



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--SE-type', default='GRU', choices=['GRU', 'BiGRU', 'AVG'])
    parser.add_argument('--neg-case', default = 'max', choices=['max', 'mean', 'random', 'randmax'])
    parser.add_argument('--train-sample', type=int, default=0, help='train sample selection strategy')
    parser.add_argument('--word-dim', type=int, default=300, help='dimension of word embeddings')
    parser.add_argument('--hidden-dim', type=int, default=300, help='dimension of hidden units per layer')
    parser.add_argument('--num-layers', type=int, default=1, help='number of layers in LSTM/BiLSTM')
    parser.add_argument('--kernel-num', type=int, default=64, help='kernel num/ output dim in CNN')
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--margin', type=float, default=3, help='margin of hinge loss, must >= 0')
    
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
    parser.add_argument('--alpha', type=float, default=0.01, help='weight of regularization term')

    parser.add_argument('--data-path', required=True, help='pickle file obtained by dataset dump or datadir for torchtext')
    parser.add_argument('--save-dir', type=str, required=True, help='path to save checkpoints and logs')
    parser.add_argument('--load-previous-model', action='store_true')
    parser.add_argument('--validation', action='store_true')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--small', action='store_true')
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
    trainChannelModel(args)


if __name__ == '__main__':
    main()

