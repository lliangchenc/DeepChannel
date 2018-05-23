import os
import argparse
import re
import numpy as np
import pandas as pd


def read_dict(path):
	f = open(path).readlines()
	dic = {}
	for line in f:
		temp = line.split(" ")
		dic[temp[0]] = [float(x) for x in temp[1:]]
	return dic
	
def dump(wordlist, dic):
	temp = []
	for word in wordlist:
		if(re.compile(r'[0-9]+').match(word)):
			temp.append(dic["zero"])
		elif(dic.has_key(word)):
			temp.append(dic[word])
		else:
			temp.append(dic["the"])
	return np.array(temp)
	
def pad(embed_arr, len):
	if(np.shape(embed_arr)[0] >= len):
		return embed_arr
	else:
		new = np.array([len, np.shape(embed_arr)[1]])
		new[:np.shape(embed_arr)[0], :] = embed_arr
		return new

def read_data(path):
	data = pd.read_csv(path)
	return list(data['content']), list(data['summary'])

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--dict_path', default="glove.6B/glove.6B.100d.txt", type=str, help='path of the "glove.6B.100d.txt"')
	print "HERE"
	#print len(read_data("cnn_processed.csv")[0])
