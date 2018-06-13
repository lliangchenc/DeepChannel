import torch
import math
from torch import nn
from torch.nn import init, functional
import numpy as np

class ChannelModel(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.se_dim = kwargs['se_dim'] # dim of sentence embedding
        self.mid_dim = 512
        self.d_transformer = nn.Sequential(
                nn.Linear(self.se_dim, self.mid_dim),
                nn.Tanh(),
                )
        self.s_transformer = nn.Sequential(
                nn.Linear(self.se_dim, self.mid_dim),
                nn.Tanh(),
                )
        self.p_producer = nn.Sequential(
                nn.Linear(self.mid_dim * 2, 1),
                nn.Sigmoid()
                )
        self.temperature = 1 # for annealing
        self.reset_parameters()

    def reset_parameters(self):
        for name, param in list(self.d_transformer.named_parameters()) + \
                list(self.s_transformer.named_parameters()) + list(self.p_producer.named_parameters()):
            if 'bias' in name:
                init.constant_(param, 0.0)
            elif 'weight' in name:
                init.normal_(param, mean=0, std=0.01)


    def forward(self, D, S): # TODO batch with torch.bmm
        '''
        To calculate P(D|S) given the sentence embeddings of D and S. 
        D: [n, Ds], where n is # sentences of document
        S: [m, Ds], where m is # sentences of summary
        '''
        n, m = D.size(0), S.size(0)
        S_T = torch.transpose(S, 0, 1).contiguous() # [Ds, m]
        att_weight = functional.softmax(
                torch.mm(D, S_T) / self.temperature,
                #torch.mm(D, S_T) / math.sqrt(self.se_dim) / self.temperature,
                dim=1
                ) # [n, m]
        att_S = torch.mm(att_weight, S) # [n, Ds]

        ## --------------------------------------------------------------- ##
        d_feat = self.d_transformer(D)
        s_feat = self.s_transformer(att_S)
        d_s_feat = torch.cat([d_feat, s_feat], dim=1)
        prob_vector = self.p_producer(d_s_feat) # P(d|S) in each row
        ## --------------------------------------------------------------- ##

        # P(D|S) on the independent assumption. TODO: 0.5 trick not elegant;
        log_prob = torch.sum(torch.log(prob_vector + 0.5))
        return log_prob, prob_vector, att_weight

