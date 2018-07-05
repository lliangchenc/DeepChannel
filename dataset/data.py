import pickle
import random
import time
import torch
import numpy as np

def wrap_numpy_to_longtensor(*args):
    return [torch.LongTensor(arg) if not isinstance(arg, list) \
            else wrap_numpy_to_longtensor(*arg) for arg in args]

class Dataset(object):

    def __init__(self, path, batch_size=1):
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
        print('Take %.2f seconds to load data. train/valid/test: %d/%d/%d.' % (time.time()-tic, self.train_size, self.valid_size, self.test_size))
        self.train_ptr = 0

#        self.train_num_batch = int(math.ceil(self.train_size / self.batch_size))
#        self.train_set_bucket_suffle()
#
#    def train_set_bucket_shuffle(self):
#        self.train_set.sort(key=lambda e: len(e[0])) # sort based on length
#        shuffle_unit = 200
#        for i in range(0, self.train_size, shuffle_unit): # shuffle for every unit
#            tmp = self.train_set[i: i+shuffle_unit]
#            random.shuffle(tmp)
#            self.train_set[i: i+shuffle_unit] = tmp
#        self.train_iter_idx = list(range(0, self.train_num_batch))
#        random.shuffle(self.train_iter_idx)
#
#    def gen_train_minibatch(self):
#        while self.train_ptr < self.train_num_batch:
#            i = self.train_iter_idx[self.train_ptr]
#            batch_size = min(self.batch_size, self.train_size - i*self.batch_size)
#            minibatch = self.train_set[i*self.batch_size : i*self.batch_size + batch_size]
#            doc_l = max(list(map(lambda x: len(x[0]), minibatch)))
#            sum_l = max(list(map(lambda x: len(x[1]), minibatch)))
#            documents = np.zeros((batch_size, doc_l), dtype='int32')
#            summaries = np.zeros((batch_size, sum_l), dtype='int32')
#            doc_lengthes = np.zeros((batch_size,), dtype='int32')
#            sum_lengthes = np.zeros((batch_size,), dtype='int32')
#            # directly padding with zero
#            for i, (d, s) in enumerate(minibatch):
#                documents[i, :len(d)] = d
#                summaries[i, :len(s)] = s
#                doc_lengthes[i] = len(d)
#                sum_lengthes[i] = len(s)
#            self.train_ptr += 1
#            yield wrap_numpy_to_longtensor(documents, summaries, doc_lengthes, sum_lengthes) # Note the order
#        else:
#            self.train_ptr = 0
#            self.train_set_bucket_shuffle()
#            raise StopIteration

    def gen_train_minibatch(self):
        # random shuffle both train_set and train_len
        combined = list(zip(self.train_set, self.train_len))
        random.shuffle(combined)
        self.train_set[:], self.train_len[:] = zip(*combined)
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
            #print(s_batch, neg_sent, end_j, s_len_batch)

            #for i in range(len(s)):
            #    new_s, new_s_len = np.copy(s), np.copy(s_len)
            #    new_s[i, :end_j] = d[d_index, :end_j]
            #    new_s[i, end_j:] = 0
            #    new_s_len[i] = end_j
            #    s_batch.append(new_s)
            #    s_len_batch.append(new_s_len)
            
            #index_max = np.max(d) 
            #s_batch.extend([np.append(np.delete(s, x, 0), [np.random.randint(index_max + 1, size=[np.shape(s)[1], ])], axis=0) for x in range(len(s))])
            #s_len_batch.extend([np.append(np.delete(s_len, x, 0), [np.shape(s)[1]], axis=0) for x in range(len(s_len))])
  
            # Strategy 2 : Delete a golden summary sentence
            #s_batch.extend([np.delete(s, x, 0) for x in range(len(s))])
            #s_len_batch.extend([np.delete(s_len, x, 0) for x in range(len(s))])
             
            yield wrap_numpy_to_longtensor(d, s_batch, d_len, s_len_batch)
            #print(s, s_batch[1])
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
#        for d, s, d_len, s_len in map(lambda _: _[0]+_[1], zip(self.test_set, self.test_len)):
#            yield wrap_numpy_to_longtensor(d, s, d_len, s_len)
#        else:
#            raise StopIteration
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
    print("haha")
    dataset = Dataset("/data/sjx/Summarization-Exp/cnn.pickle")
    #print(dataset.train_size)
    #print(dataset.gen_train_minibatch())
    #print(dataset.gen_train_minibatch)
    #print("here",dataset.itow(123))

if __name__ == "__main__":
    main()
