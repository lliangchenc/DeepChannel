import torch
from torch import nn
from torch.nn import init
from ..utils import reverse_padded_sequence, unwrap_scalar_variable

class BiGRU(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.forward_encoder = nn.GRU(input_size=kwargs['word_dim'],
                hidden_size=kwargs['hidden_dim'],
                num_layers=kwargs['num_layers'],
                dropout=kwargs['dropout'])
        self.backward_encoder = nn.GRU(input_size=kwargs['word_dim'],
                hidden_size=kwargs['hidden_dim'],
                num_layers=kwargs['num_layers'],
                dropout=kwargs['dropout'])
        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.forward_encoder.named_parameters():
            if 'bias' in name:
                init.constant(param.data, 0.0)
            elif 'weight' in name:
                init.normal(param.data, mean=0, std=0.01)
        for name, param in self.backward_encoder.named_parameters():
            if 'bias' in name:
                init.constant(param.data, 0.0)
            elif 'weight' in name:
                init.normal(param.data, mean=0, std=0.01)

    def forward(self, input, length):
        # input: [bsz, len, w_dim]
        # length: [bsz, ]
        input = torch.transpose(input, 0, 1).contiguous() # [len, bsz, w_dim]
        reversed_input = reverse_padded_sequence(input, length, batch_first=False)
        forward_output = self.forward_encoder(input)[0]
        backward_output = self.backward_encoder(reversed_input)[0] # [len, bsz, hid_dim]
        backward_output = reverse_padded_sequence(backward_output, length, batch_first=False)
        output = torch.cat([forward_output, backward_output], dim=2) # [len, bsz, 2*hid_dim]
        return output


class BiGRU_wrapper(BiGRU):

    def __init__(self, **kwargs):
        super().__init__(kwargs)

    def forward(self, input, length):
        all_h = super().forward(input, length)
        bsz, dim2 = all_h.size(1), all_h.size(2)
        output = torch.stack([
                torch.cat([
                    all_h[unwrap_scalar_variable(length[i])-1, i, :dim2/2], # forward
                    all_h[0, i, dim2/2:] # backward
                ]) # concat the forward embedding and the backward embedding
            for i in range(bsz)])
        return output # [bsz, 2*h_dim]

        


class GRU_wrapper(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = nn.GRU(input_size=kwargs['word_dim'],
                hidden_size=kwargs['hidden_dim'],
                num_layers=kwargs['num_layers'],
                dropout=kwargs['dropout'])
        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.encoder.named_parameters():
            if 'bias' in name:
                init.constant(param.data, 0.0)
            elif 'weight' in name:
                init.normal(param.data, mean=0, std=0.01)

    def forward(self, input, length):
        # input: [bsz, len, w_dim]
        # length: [bsz, ]
        input = torch.transpose(input, 0, 1).contiguous() # [len, bsz, w_dim]
        all_h = self.encoder(input)[0]
        bsz = length.size(0)
        output = torch.stack(
                [all_h[unwrap_scalar_variable(length[i])-1, i] for i in range(bsz)]
                )
        return output # [bsz, h_dim]





class SentenceEmbedding(nn.Module):

    def __init__(self, **kwargs):
        super(SentenceEmbedding, self).__init__()
        self.drop = nn.Dropout(kwargs['dropout'])
        self.word_embedding = nn.Embedding(num_embeddings=kwargs['num_words'],
                                            embedding_dim=kwargs['word_dim'])
        self.SE_type = kwargs['SE_type']
        if self.SE_type == 'GRU':
            self.encoder = GRU_wrapper(**kwargs)
            self.dim = kwargs['hidden_dim']
        elif self.SE_type == 'BiGRU':
            self.encoder = BiGRU_wrapper(**kwargs)
            self.dim = 2 * kwargs['hidden_dim']
        elif self.SE_type == 'AVG':
            pass

    def getDim(self):
        return self.dim

    def forward(self, input, length):
        # input: [bsz, len]
        # length: [bsz, ]
        input = self.drop(self.word_embedding(input)) # [bsz, len, word_dim]
        output = self.encoder(input, length)
        return output # [bsz, self.dim]

