import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, 
                 input_dim, 
                 output_dim, 
                 hidden_activation=nn.ReLU, 
                 output_activation=nn.Softmax, 
                 hidden_neurons=[], 
                 flatten=True, 
                 bias=True):
        """Standard MLP with
        """
        super().__init__()
        self.layers = nn.ModuleList()
        if flatten:
            self.layers.append(nn.Flatten(1))
        hidden_neurons.append(output_dim)
        in0 = input_dim
        for i, h0 in enumerate(hidden_neurons):
            layer = nn.Linear(in0, h0, bias=bias)
            self.layers.append(layer)
            if i != len(hidden_neurons) - 1:
                self.layers.append(hidden_activation())
            else:
                self.layers.append(output_activation())
            in0 = h0
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
if __name__ == "__main__":
    import tqdm
    X1 = torch.randn(256, 53)*1 + 1
    X2 = torch.rand(256, 53)*1 - 1   
    Y1 = torch.zeros((256,))
    Y2 = torch.ones((256,))
    Xs = torch.cat([X1, X2], 0)
    Ys = torch.cat([Y1, Y2], 0)
    model = MLP(53, 2, hidden_neurons=[4096*2, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4])
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    pbar = tqdm.tqdm(range(100))
    for epoch in pbar:
        idx = torch.randperm(64)
        X = Xs[idx, ...]
        Y = Ys[idx]
        opt.zero_grad()
        Yh = model(X)
        loss = nn.CrossEntropyLoss()(Yh, Y.long())
        loss.backward()
        opt.step()
        pbar.set_description("Loss=%.20f" % loss.item())