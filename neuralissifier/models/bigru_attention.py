import torch
import torch.nn as nn


class BiGRUAttention(nn.Module):

    def __init__(self, 
                 seqlen, 
                 dim, 
                 hidden_size=128, 
                 bidirectional=True, 
                 drp=0.1, 
                 num_layers=3,                 
                 do_layernorm=False,
                 do_batchnorm=True,
                 activation='relu',
                 activation_alpha=0.0):
        """
        Seqlen - length of the sequence
        Dim - dimension of the input sequence
        Hidden_size - hidden_size of the feed forward component
        """
        super(BiGRUAttention, self).__init__()        
        self.do_batchnorm = do_batchnorm
        self.do_layernorm = do_layernorm
        mul = 2 if bidirectional else 1
        self.hidden_size = hidden_size
        self.lstm = nn.GRU(dim, self.hidden_size, num_layers=num_layers,
                            bidirectional=bidirectional, batch_first=True)
        self.layernorm_lstm = nn.LayerNorm((seqlen,self.hidden_size*mul))
        self.batchnorm = nn.BatchNorm1d(seqlen)
        self.batchnorm_fc = nn.BatchNorm1d(self.hidden_size)
        self.attention = nn.Linear(self.hidden_size*mul, 1)
        self.linear = nn.Linear(hidden_size*mul, hidden_size)
        self.layernorm_fc = nn.LayerNorm(hidden_size)
        if activation == 'tanh':
            self.act = nn.Tanh()
        elif activation == 'lrelu':
            self.act = nn.LeakyReLU(negative_slope=activation_alpha)
        else:
            self.act = nn.ReLU()
        self.dropout = nn.Dropout(drp)
        self.outlayer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h_lstm, _ = self.lstm(x)
        if self.do_layernorm:
            h_lstm = self.layernorm_lstm(h_lstm)
        if self.do_batchnorm:
            h_lstm = self.batchnorm(h_lstm)
        #h_flat = torch.flatten(h_lstm, 1)
        attention_weights = torch.softmax(self.attention(h_lstm).squeeze(-1), dim=-1)
        context_vector = torch.sum(h_lstm * attention_weights.unsqueeze(-1), dim=1)
        out = self.linear(context_vector)
        out = self.act(out)
        if self.do_layernorm:
            out = self.layernorm_fc(out)
        if self.do_batchnorm:
            out = self.batchnorm_fc(out)
        out = self.dropout(out)
        out = self.outlayer(out)
        return out

if __name__=="__main__":
    X = torch.randn(32, 100, 1378)
    model = BiGRUAttention(100, 1378, hidden_size=128, bidirectional=False)
    yh = model(X)
    print(yh.shape)
    def count_parameters(mmodel):
        return sum(p.numel() for p in mmodel.parameters() if p.requires_grad)
    print(count_parameters(model))