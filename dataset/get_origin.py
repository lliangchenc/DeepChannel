import os
import tqdm
import hashlib

def hashhex(s):
    h = hashlib.sha1()
    h.update(s)
    return h.hexdigest()

def key2File(data_dir, key2file):
    for k in os.listdir(data_dir):
        f = os.path.join(data_dir, k)
        l = k[:-6]
        key2file[k] = f

def query(index):
    keys = {}
    data_dir = ["/data/share/cnn_stories/stories", "/data/share/dailymail_stories/stories"]
    for i in range(2):
        key2File(data_dir[i], keys)
    url_file = "./cnndaily_url_splits/all_test.txt"
    lines = open(url_file).readlines()
    line = lines[index]
    k = hashhex(line.strip().encode())
    f = keys[k+".story"]
    return f

if __name__ == "__main__":
    file_name = query(9740)
    content = open(file_name).read()
    open("9740.txt", "w").write(content)
