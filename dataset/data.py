import pickle
import random
import time
import torch
import numpy as np
random.seed(666)

def wrap_numpy_to_longtensor(*args):
    return map(torch.LongTensor, args)

class Dataset(object):

    def __init__(self, path, batch_size=1):
        self.batch_size = batch_size

        tic = time.time()
        data_file = open(path, 'rb')
        self.train_set, self.valid_set, self.test_set = pickle.load(data_file)
        self.weight = torch.FloatTensor(pickle.load(data_file))
        self.wtoi = pickle.load(data_file)
        self.itow = pickle.load(data_file)
        data_file.close()
        self.train_size, self.valid_size, self.test_size = len(self.train_set), len(self.valid_set), len(self.test_set)
        print('Take %.2f seconds to load data. train/valid/test: %d/%d/%d.' % (time.time()-tic, self.train_size, self.valid_size, self.test_size))

        self.train_num_batch = int(math.ceil(self.train_size / self.batch_size))
        self.train_set_bucket_suffle()
        self.train_ptr = 0

    def train_set_bucket_shuffle(self):
        self.train_set.sort(key=lambda e: len(e[0])) # sort based on length
        shuffle_unit = 200
        for i in range(0, self.train_size, shuffle_unit): # shuffle for every unit
            tmp = self.train_set[i: i+shuffle_unit]
            random.shuffle(tmp)
            self.train_set[i: i+shuffle_unit] = tmp
        self.train_iter_idx = list(range(0, self.train_num_batch))
        random.shuffle(self.train_iter_idx)

    def gen_train_minibatch(self):
        while self.train_ptr < self.train_num_batch:
            i = self.train_iter_idx[self.train_ptr]
            batch_size = min(self.batch_size, self.train_size - i*self.batch_size)
            minibatch = self.train_set[i*self.batch_size : i*self.batch_size + batch_size]
            doc_l = max(list(map(lambda x: len(x[0]), minibatch)))
            sum_l = max(list(map(lambda x: len(x[1]), minibatch)))
            documents = np.zeros((batch_size, doc_l), dtype='int32')
            summaries = np.zeros((batch_size, sum_l), dtype='int32')
            doc_lengthes = np.zeros((batch_size,), dtype='int32')
            sum_lengthes = np.zeros((batch_size,), dtype='int32')
            # directly padding with zero
            for i, (d, s) in enumerate(minibatch):
                documents[i, :len(d)] = d
                summaries[i, :len(s)] = s
                doc_lengthes[i] = len(d)
                sum_lengthes[i] = len(s)
            self.train_ptr += 1
            yield wrap_numpy_to_longtensor(documents, summaries, doc_lengthes, sum_lengthes) # Note the order
        else:
            self.train_ptr = 0
            self.train_set_bucket_shuffle()
            raise StopIteration


    def gen_valid_minibatch(self):
        for d, s in self.valid_set:
            doc_lengthes = np.asarray(d.shape)
            sum_lengthes = np.asarray(s.shape)
            documents = d.reshape(1, -1)
            summaries = d.reshape(1, -1)
            yield wrap_numpy_to_longtensor(documents, summaries, doc_lengthes, sum_lengthes)
        else:
            raise StopIteration
            

    def gen_test_minibatch(self):
        for d, s in self.test_set:
            doc_lengthes = np.asarray(d.shape)
            sum_lengthes = np.asarray(s.shape)
            documents = d.reshape(1, -1)
            summaries = d.reshape(1, -1)
            yield wrap_numpy_to_longtensor(documents, summaries, doc_lengthes, sum_lengthes)
        else:
            raise StopIteration

