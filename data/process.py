import os
import argparse
import re
import numpy as np
import pandas as pd
from nltk.tokenize.stanford import StanfordTokenizer



def process_cnn_data():
	parser = argparse.ArgumentParser()
	parser.add_argument('--dict_path', default="glove.6B/glove.6B.100d.txt", type=str, help='path of the "glove.6B.100d.txt"')
	parser.add_argument('--base_dir', default="./cnn_stories/cnn/stories/ ./dailymail_stories/dailymail/stories/", \
						type=str, help='paths of the source folders')
	parser.add_argument('--saved_names', default="cnn dailymail", \
						type=str, help='saved names')
	args = parser.parse_args()
	print args

	j = 0


	#, "./dailymail_stories/dailymail/stories/"

	#dic = read_dict(args.dict_path)
	base_dirs = args.base_dir.split(" ")
	names = args.saved_names.split(" ")

	#print dic['zero'], dump(['the', 'red', '123'], dic)

	for index in range(len(base_dirs)):
		content = []
		summary = []
		content_embed = []
		summary_embed = []
		dir = os.listdir(base_dirs[index])
		print len(dir)
		
		for i in range(2):
			
			f = open(base_dirs[index] + dir[i]).read()
			f = f.replace("\n", "")
			print i
			temp = f.split("@highlight")
			temp_content = StanfordTokenizer().tokenize(temp[0])
			s1 = []
			for word in temp_content:
				if(re.compile(r'[0-9]+').match(word)):
					s1.append("0")
				else:
					s1.append(word)
			
			content.append(" ".join(s1))
			#content_embed.append(dump(s1, dic))

			temp_content = StanfordTokenizer().tokenize(" ".join(temp[1:]))
			s1 = []
			for word in temp_content:
				if(re.compile(r'[0-9]+').match(word)):
					s1.append("0")
				else:
					s1.append(word)

			summary.append(" ".join(s1))
			#summary_embed.append(dump(s1, dic))

		j += 1

		print np.shape(content_embed)
		save = pd.DataFrame({'content' : content, 'summary' : summary})
		save.to_csv(names[index] + "_processed.csv")

if __name__ == "__main__":
	process_cnn_data()