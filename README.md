# DeepChannel
This repository is the pytorch implementation of paper

[DeepChannel: Salience Estimation by Contrastive Learning for Extractive Document Summarization](https://arxiv.org/abs/1811.02394).
> Jiaxin Shi*, Chen Liang*, Lei Hou, Juanzi Li, [Zhiyuan Liu](http://nlp.csai.tsinghua.edu.cn/~lzy/index.html), [Hanwang Zhang](http://www.ntu.edu.sg/home/hanwangzhang/#aboutme).

In this paper, we propose DeepChannel, a robust, data-efficient, and interpretable neural model for extractive document summarization. Given any document-summary pair, we estimate a salience score, which is modeled using an attention-based deep neural network, to represent the salience degree of the summary for yielding the document. We devise a contrastive training strategy to learn the salience estimation network, and then use the learned salience score as a guide and iteratively extract the most salient sentences from the document as our generated summary.

If you find this code useful in your research, please cite
``` tex
@Inproceedings{Shi2018DeepChannel,
title={DeepChannel: Salience Estimation by Contrastive Learning for Extractive Document Summarization},
author={Jiaxin Shi, Chen Liang, Lei Hou, Juanzi Li, Zhiyuan Liu, Hanwang Zhang},
booktitle = {AAAI},
year = {2019}
}
```

## Requirements
- python==3.6
- pytorch==1.0.0
- spacy
- nltk
- pyrouge & rouge

## Preprocessing

Before training the model, please follow the instructions below to prepare all the data needed for the experiments.

#### GLOVE
Please download the [GloVe 300d pretrained vector](http://nlp.stanford.edu/data/glove.840B.300d.zip), which is used for word embeddding initialization in all experiments.


#### CNN-Daily
1. Download the [CNN-Daily story corpus](https://cs.nyu.edu/~kcho/DMQA/).
2. Preprocess the original CNN-Daily story corpus and generate the data file:
``` shell
python dataset/process.py --glove </path/to/the/pickle/file> --data cnn+dailymail --data-dir </path/to/the/corpus> --save-path </path/to/the/output/file> --max-word-num MAX_WORD_NUM
```
The output file will be used in the data loader when training or testing. To reproduce state-of-the-art result, please use the 300d glove file and use the default max-word-num

#### DUC2007
1. Download the [DUC2007 corpus](https://duc.nist.gov/duc2007/tasks.html).
2. Preprocess the original DUC2007 corpus and generate the data file:
``` shell
python dataset/process.py --glove </path/to/the/pickle/file> --data duc2007 --data-dir </path/to/the/corpus> --save-path </path/to/the/output/file> 
```
The output file will be used in testing.

#### pyrouge
We modified the original python wrapper of ROUGE-1.5.5, fixing some errors and rewriting its interfaces in a more friendly way. To accelerate the training process and alleviate the freqrent IO operation of ROUGE-1.5.5, we pre-calculate the rouge attention matrix of every document-summary pair. Please use the following command to accomplish this step:
``` shell
python offline_pyrouge.py --data-path </path/to/the/processed/data> --save-path </path/to/the/output/file> 
```

## Train
You can simply use the command below to train the DeepChannel model with the default hyperparameters:
```shell
python train.py --data-path </path/to/the/processed/data> --save-dir </path/to/save/the/model> --offline-pyrouge-index-json </path/to/the/offline/pyrouge/file>
```

You can also try training the model with different hyperparameters:
```shell
python train.py --data-path </path/to/the/processed/data> --save-dir </path/to/save/the/model> --offline-pyrouge-index-json </path/to/the/offline/pyrouge/file> --SE-type SE_TYPE --word-dim WORD_DIM --hidden-dim HIDDEN_DIM --dropout DROPOUT --margin MARGIN --lr LR --optimizer OPT ...
```

For detailed information about all the hyperparameters, please run the command:
```shell
python train.py --help
```

We implement three sentence embedding strategies: GRU, Bi-GRU and average, which can be specified by the argument `--SE-type`. If you want to train the model with the reduced dataset, please specify the `--fraction` argument.

## Test
You can run `summarize.py` to apply the greedy extraction procedure on test set as well as evaluating the performance on it. Please ensure the hyperparameters in test step is consistent with the ones in traing step. For comparision, we implement some different extracting strategies, which can be specified by the argument `--method`. Typically, you can directly run the following command for a basic evaluation.
``` shell
python summarize.py --data-path </path/to/the/processed/data> --save-dir </path/to/the/saved/model> 
```

## Acknowledgement
We refer to some codes of these repos:

- [Preprocessing of the CNN-Daily dataset](https://github.com/abisee/cnn-dailymail)
- [Tensorflow implementation of Pointer-Generator](https://github.com/abisee/pointer-generator)
- [Pytorch implementation of Pointer-Generator](https://github.com/atulkum/pointer_summarizer)
- [Pytorch implementation of SummaRuNNer](https://github.com/hpzhao/SummaRuNNer)
- [Tensorflow implementation of Refresh](https://github.com/EdinburghNLP/Refresh)
- [pyrouge](https://github.com/bheinzerling/pyrouge)
- [rouge](https://github.com/pltrdy/rouge)

Appreciate for their great contributions!