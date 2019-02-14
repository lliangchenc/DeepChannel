import pickle
import random
import time
import torch
import numpy as np

def wrap_numpy_to_longtensor(*args):
    return [torch.LongTensor(arg) if not isinstance(arg, list) \
            else wrap_numpy_to_longtensor(*arg) for arg in args]

class Dataset(object):

    def __init__(self, path, batch_size=1, fraction=1):
        self.batch_size = batch_size
        #sys.exit(1)

        tic = time.time()
        data_file = open(path, 'rb')
        self.train_set, self.valid_set, self.test_set = pickle.load(data_file)
        self.train_len, self.valid_len, self.test_len = pickle.load(data_file)
        self.weight = torch.FloatTensor(pickle.load(data_file))
        self.wtoi = pickle.load(data_file)
        self.itow = pickle.load(data_file)
        data_file.close()
        self.train_size, self.valid_size, self.test_size = len(self.train_set), len(self.valid_set), len(self.test_set)
        
        
        ## used for small training set
        self.train_set = self.train_set[:int(self.train_size * fraction)]
        self.train_len = self.train_len[:int(self.train_size * fraction)]
        self.train_size = len(self.train_set)
        self.train_ori_index = list(range(self.train_size)) 
        # self.train_ori_index[i] is the original index of self.train_set[i]


        print('Take %.2f seconds to load data. train/valid/test: %d/%d/%d.' % (time.time()-tic, self.train_size, self.valid_size, self.test_size))
        self.train_ptr = 0

    def gen_train_minibatch(self, shuffle=True):
        # random shuffle both train_set and train_len
        if shuffle:
            combined = list(zip(self.train_set, self.train_len, self.train_ori_index))
            random.shuffle(combined)
            self.train_set[:], self.train_len[:], self.train_ori_index[:] = zip(*combined)
        for d, s, d_len, s_len in map(lambda _: _[0]+_[1], zip(self.train_set, self.train_len)):
            s_batch = [s]
            s_len_batch = [s_len]
             
            # Strategy 1 : Replace a golden summary sentence with a random selected sentence from document
            d_index = random.randint(0, len(d) - 1)
            end_j = min(s.shape[1], d_len[d_index])

            neg_sent = np.int32(np.zeros([1, s.shape[1]]))
            neg_sent[0, :end_j] = d[d_index, :end_j]
            s_batch.append(neg_sent)
            s_len_batch.append(np.int32([end_j]))

            yield wrap_numpy_to_longtensor(d, s_batch, d_len, s_len_batch)

        else:
            raise StopIteration


    def gen_valid_minibatch(self):
        combined = list(zip(self.valid_set, self.valid_len))
        self.valid_set[:], self.valid_len[:] = zip(*combined)
        for d, s, d_len, s_len in map(lambda _: _[0]+_[1], zip(self.valid_set, self.valid_len)):
            s_batch = [s]
            s_len_batch = [s_len]
            d_index = random.randint(0, len(d) - 1)
            end_j = min(s.shape[1], d_len[d_index])
            neg_sent = np.int32(np.zeros([1, s.shape[1]]))
            neg_sent[0, :end_j] = d[d_index, :end_j]
            s_batch.append(neg_sent)
            s_len_batch.append(np.int32([end_j]))
            yield wrap_numpy_to_longtensor(d, s_batch, d_len, s_len_batch)
        else:
            raise StopIteration
            

    def gen_test_minibatch(self):
        combined = list(zip(self.test_set, self.test_len))
        self.test_set[:], self.test_len[:] = zip(*combined)
        for d, s, d_len, s_len in map(lambda _: _[0]+_[1], zip(self.test_set, self.test_len)):
            s_batch = [s]
            s_len_batch = [s_len]
            d_index = random.randint(0, len(d) - 1)
            end_j = min(s.shape[1], d_len[d_index])
            neg_sent = np.int32(np.zeros([1, s.shape[1]]))
            neg_sent[0, :end_j] = d[d_index, :end_j]
            s_batch.append(neg_sent)
            s_len_batch.append(np.int32([end_j]))
            yield wrap_numpy_to_longtensor(d, s_batch, d_len, s_len_batch)
        else:
            raise StopIteration

def main():
    pass

if __name__ == "__main__":
    main()
