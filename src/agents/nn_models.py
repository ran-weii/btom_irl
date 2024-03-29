import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_hidden, activation):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_hidden = num_hidden
        self.activation = activation
        
        if activation == "relu":
            act = nn.ReLU()
        elif activation == "silu":
            act = nn.SiLU()
        else:
            raise NotImplementedError

        layers = [nn.Linear(input_dim, hidden_dim)]
        for _ in range(num_hidden):
            layers.append(act)
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(act)
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.layers = nn.ModuleList(layers)

    def __repr__(self):
        s = "{}(input_dim={}, output_dim={}, hidden_dim={}, num_hidden={}, activation={})".format(
            self.__class__.__name__, self.input_dim, self.output_dim, 
            self.hidden_dim, self.num_hidden, self.activation
        )
        return s

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class EnsembleLinear(nn.Module):
    """ Ensemble version of nn.Linear """
    def __init__(self, input_dim, output_dim, ensemble_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.ensemble_dim = ensemble_dim

        self.weight = nn.Parameter(torch.zeros(ensemble_dim, input_dim, output_dim))
        self.bias = nn.Parameter(torch.zeros(ensemble_dim, output_dim))
        nn.init.trunc_normal_(self.weight, std=1/(2*input_dim**0.5))
    
    def forward(self, x):
        """ Output size=[..., ensemble_dim, output_dim] """
        out = torch.einsum("kio, ...ki -> ...ko", self.weight, x) + self.bias
        return out
    
    
class EnsembleMLP(nn.Module):
    def __init__(self, input_dim, output_dim, ensemble_dim, hidden_dim, num_hidden, activation):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.ensemble_dim = ensemble_dim
        self.hidden_dim = hidden_dim
        self.num_hidden = num_hidden
        self.activation = activation
        
        if activation == "relu":
            act = nn.ReLU()
        elif activation == "silu":
            act = nn.SiLU()
        else:
            raise NotImplementedError

        layers = [EnsembleLinear(input_dim, hidden_dim, ensemble_dim)]
        for _ in range(num_hidden):
            layers.append(act)
            layers.append(EnsembleLinear(hidden_dim, hidden_dim, ensemble_dim))
        layers.append(act)
        layers.append(EnsembleLinear(hidden_dim, output_dim, ensemble_dim))
        self.layers = nn.ModuleList(layers)

    def __repr__(self):
        s = "{}(input_dim={}, output_dim={}, ensemble_dim={}, hidden_dim={}, num_hidden={}, activation={})".format(
            self.__class__.__name__, self.input_dim, self.output_dim, self.ensemble_dim,
            self.hidden_dim, self.num_hidden, self.activation
        )
        return s

    def forward(self, x):
        """
        Args:
            x (torch.tensor): input batch. size=[batch_size, input_dim]

        Outputs:
            x (torch.tensor): output batch. size=[batch_size, k, output_dim]
        """
        x = x.unsqueeze(-2).repeat_interleave(self.ensemble_dim, dim=-2)
        for layer in self.layers:
            x = layer(x)
        return x
    
    def forward_separete(self, x):
        """ Forward each ensemble member separately 
        
        Args:
            x (torch.tensor): size=[..., ensemble_dim, input_dim]

        Returns:
            out (torch.tensor): size=[..., ensemble_dim, output_dim] 
        """
        for layer in self.layers:
            x = layer(x)
        return x