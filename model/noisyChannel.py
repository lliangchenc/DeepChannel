import torch
import math
from torch import nn
from torch.nn import init, Parameter, functional

class ChannelModel(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.se_dim = kwargs['se_dim'] # dim of sentence embedding
        self.W = Parameter(torch.FloatTensor(self.se_dim, self.se_dim))
        self.temperature = 1 # for annealing
        self.prob_arr = []
        self.attention = []
        self.reset_parameters()

    def reset_parameters(self):
        init.normal(self.W, mean=0, std=0.01)

    def forward(self, D, S): # TODO batch with torch.bmm
        '''
        To calculate P(D|S) given the sentence embeddings of D and S. 
        D: [n, Ds], where n is # sentences of document
        S: [m, Ds], where m is # sentences of summary
        '''
        n, m = D.size(0), S.size(0)
        S_T = torch.transpose(S, 0, 1).contiguous() # [Ds, m]
        att_weight = functional.softmax(
                torch.mm(D, S_T) / math.sqrt(self.se_dim) / self.temperature,
                dim=1
                ) # [n, m]
        att_S = torch.mm(att_weight, S) # [n, Ds]
        # logits = torch.diag(torch.mm(torch.mm(D, self.W), torch.t(att_S))) # TODO: which is faster?
        logit_arr = [torch.dot(D[i], torch.mv(self.W, att_S[i])) for i in range(n)]
        logits = torch.stack(logit_arr)
        log_prob = torch.sum(functional.logsigmoid(logits)) # P(D|S) on the independent assumption
        self.prob_arr = [x.item() for x in logit_arr]
        self.attention = att_weight.clone().cpu().data.numpy()
        return log_prob

