import pickle
import numpy as np
from tqdm import tqdm

def main():
    glove300 = '/data/sjx/glove.840B.300d.py36.pkl'
    path100d = '/data/c-liang/data/cnndaily_5w_100d.pkl'
    path300d = '/data/c-liang/data/cnndaily_5w_300d.pkl'

    with open(path100d, 'rb') as f:
        data1 = pickle.load(f)
        data2 = pickle.load(f)
        weight = pickle.load(f)
        data4 = pickle.load(f)
        itow = pickle.load(f)

    glove = pickle.load(open(glove300, 'rb'))
    word_dim = 300
    new_weight = []
    for i in tqdm(range(weight.shape[0])):
        w = itow[i]
        new_weight.append(glove.get(w, np.zeros((word_dim, ))))
    weight = np.vstack(new_weight)
    print(weight.shape)

    with open(path300d, 'wb') as f:
        pickle.dump(data1, f)
        pickle.dump(data2, f)
        pickle.dump(weight, f)
        pickle.dump(data4, f)
        pickle.dump(itow, f)

if __name__ == '__main__':
    main()
