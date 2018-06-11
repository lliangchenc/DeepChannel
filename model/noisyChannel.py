import torch
import math
import random
from torch import nn
from torch.nn import init, Parameter, functional
import numpy as np

class ChannelModel(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.se_dim = kwargs['se_dim'] # dim of sentence embedding
        self.W = Parameter(torch.FloatTensor(self.se_dim, self.se_dim))
        self.temperature = 1 # for annealing
        self.prob_arr = []
        self.prob_matrix = []
        self.attention = []
        self.reset_parameters()

    def reset_parameters(self):
        init.normal_(self.W, mean=0, std=0.01)

    def forward(self, D, S): # TODO batch with torch.bmm
        '''
        To calculate P(D|S) given the sentence embeddings of D and S. 
        D: [n, Ds], where n is # sentences of document
        S: [m, Ds], where m is # sentences of summary
        '''
        n, m = D.size(0), S.size(0)
        S_T = torch.transpose(S, 0, 1).contiguous() # [Ds, m]
#        att_weight = functional.softmax(
#                torch.mm(D, S_T) / self.temperature,
#                #torch.mm(D, S_T) / math.sqrt(self.se_dim) / self.temperature,
#                dim=1
#                ) # [n, m]
#        att_S = torch.mm(att_weight, S) # [n, Ds]
#        # logits = torch.diag(torch.mm(torch.mm(D, self.W), torch.t(att_S))) # TODO: which is faster?
#        logit_arr = [torch.dot(D[i], torch.mv(self.W, att_S[i])) for i in range(n)]
#        logits = torch.stack(logit_arr)
#        log_prob = torch.mean(functional.logsigmoid(logits)) # P(D|S) on the independent assumption
#        #print(logits)
#        print(att_weight)
#        print(logits)
#        self.prob_arr = [x.item() for x in logit_arr]
#        self.attention = att_weight.clone().cpu().data.numpy()
#        return log_prob

        # Choose the max prob
        self.prob_matrix = torch.mm(torch.mm(D, self.W), S_T) # [n, m]
        #rand_matrix = torch.randn_like(prob_matrix) * (torch.max(prob_matrix)-torch.min(prob_matrix))
        #select_matrix = torch.add(prob_matrix, rand_matrix)

        #if(random.random() < 1e-3):
        #    logits = torch.stack([prob_matrix[i, random.randint(0, m - 1)] for i in range(n)])
        #else:
        #    logits = torch.max(prob_matrix, dim=1)[0]
        #logits = torch.stack(temp)
        logits, index = torch.max(self.prob_matrix, dim=1) # [n, ]
        self.attention = self.prob_matrix.clone().cpu().data.numpy()
        #print(self.prob_matrix)
        #print(torch.equal(logits, logit))
        #print(functional.sigmoid(logits), logits)
        log_prob = torch.mean(functional.sigmoid(logits))
        #log_prob = torch.mean(torch.log(0.5 + functional.sigmoid(logits))) # TODO: mean or sum???
        return log_prob

