import torch
import math
from torch import nn
from torch.nn import init, functional
import time
import numpy as np

class ChannelModel(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.se_dim = kwargs['se_dim'] # dim of sentence embedding
        self.p_producer = nn.Sequential(
                nn.Dropout(kwargs['dropout']),
                #nn.Linear(self.se_dim * 2, 1024),
                nn.Linear(self.se_dim * 2+1, 1024),
                nn.ReLU(),
                #nn.Dropout(kwargs['dropout']),
                #nn.Linear(512, 512),
                #nn.ReLU(),
                #nn.Dropout(kwargs['dropout']),
                nn.Linear(1024, 256),
                nn.ReLU(),
                #nn.Dropout(kwargs['dropout']),
                nn.Linear(256, 1),
                nn.Sigmoid()
                )
        self.trans_matrix = torch.rand([self.se_dim, self.se_dim], requires_grad=True)
        self.temperature = 0.1 # for annealing
        self.reset_parameters()
    def reset_parameters(self):
        for name, param in list(self.p_producer.named_parameters()):
            if 'bias' in name:
                init.constant_(param, 0.0)
            elif 'weight' in name:
                init.kaiming_normal_(param)


    def forward(self, D, S): # TODO batch with torch.bmm
        '''
        To calculate P(D|S) given the sentence embeddings of D and S. 
        D: [n, Ds], where n is # sentences of document
        S: [m, Ds], where m is # sentences of summary
        '''
        t1 = time.time()
        n, m = D.size(0), S.size(0)
        S_T = torch.transpose(S, 0, 1).contiguous() # [Ds, m]
        att_weight = functional.softmax(
                torch.mm(D, S_T) / self.temperature,
                #torch.mm(D, S_T) / math.sqrt(self.se_dim) / self.temperature,
                dim=1
                ) # [n, m]
        att_S = torch.mm(att_weight, S) # [n, Ds]
        l1 = torch.unsqueeze(D[0], 0)
        tmp = torch.mm(l1, self.trans_matrix)
        cross = torch.unsqueeze(torch.diag(torch.mm(torch.mm(D, self.trans_matrix), torch.transpose(att_S, 0, 1).contiguous()), 0), 1)
        ## --------------------------------------------------------------- ##
        d_s_feat = torch.cat([D, att_S, cross], dim=1)
        #d_s_feat = torch.cat([D, att_S], dim=1)
        prob_vector = self.p_producer(d_s_feat)# P(d|S) in each row
        ## --------------------------------------------------------------- ##

        # P(D|S) on the independent assumption. TODO: 0.5 trick not elegant;
        log_prob = torch.sum(torch.log(0.5 + prob_vector))
        addition = {
                'prob_vector': prob_vector,
                'att_weight': att_weight,
                'att_S': att_S
                }
        return log_prob, addition 
