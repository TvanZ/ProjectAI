import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class NarmPlus(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_size, output_dim, num_layers, num_items, activation_fn=nn.RReLU()):
        
        super(NarmPlus, self).__init__()
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.hidden_state_dim = 0 # calculation needed
        self.ItemEmbedding = nn.Embedding(num_items, embedding_dim)
        self.ActivationFn = activation_fn
        self.UserProfile = nn.Linear(embedding_dim, self.hidden_state_dim)
        self.Local = nn.GRU()
        self.Global = nn.GRU()
        self.Softmax = nn.Softmax()
        self.Decoder = nn.Bilinear(embedding_dim, output_dim)
                                   
        self.loss = self.top1
        
    def forward(self, x):

        history, current = x
        embeds_h = self.ItemEmbedding(history)
        profile = self.UserProfile(embeds_h)
        c = torch.cat(c_global, c_local, dim=2)

        out = self.Decoder()
        output = self.Softmax(out)

        return output
    
    def top1(self, yhat):
        ''' Top1 loss, yhat is vector with softmax probabilities '''
        # Not sure if you can just call backward to this, but I think it should work. Code from: 
        # https://github.com/mquad/hgru4rec/blob/master/src/hgru4rec.py
        yhatT = torch.transpose(yhat, 0, 1)
        loss = torch.mean(torch.mean(nn.sigmoid( - torch.diag(yhat) + yhatT) + nn.sigmoid(yhat ** 2), dim = 0) - nn.sigmoid(T.diag(yhat ** 2)))
        return loss

