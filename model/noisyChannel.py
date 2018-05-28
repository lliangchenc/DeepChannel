import torch
import math
from torch.nn import init, Parameter, functional

class ChannelModel(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.Ds = kwargs['Ds'] # dim of sentence embedding
        self.W = Parameter(torch.FloatTensor(self.Ds, self.Ds))
        self.temperature = 1 # for annealing
        self.reset_parameters()

    def reset_parameters(self):
        init.normal(self.W.data, mean=0, std=0.01)

    def forward(self, D, S): # TODO batch with torch.bmm
        '''
        To calculate P(D|S) given the sentence embeddings of D and S. 
        D: [n, Ds], where n is # sentences of document
        S: [m, Ds], where m is # sentences of summary
        '''
        n, m = D.size(0), S.size(0)
        S_T = torch.transpose(S, 0, 1).contiguous() # [Ds, m]
        att_weight = functional.Softmax(
                torch.mm(D, S_T) / math.sqrt(self.Ds) / self.temperature,
                dim=1
                ) # [n, m]
        att_S = torch.mm(att_weight, S) # [n, Ds]
        # logits = torch.diag(torch.mm(torch.mm(D, self.W), torch.t(att_S))) # TODO: which is faster?
        logits = torch.stack([torch.dot(D[i], torch.mv(self.W, att_S[i])) for i in range(n)])
        log_prob = torch.sum(functional.logsigmoid(logits)) # P(D|S) on the independent assumption
        return log_prob

