import math
import torch
import torch.nn as nn

class EnsembleLinear(nn.Module):
    """ Ensemble version of nn.Linear """
    def __init__(self, input_dim, output_dim, ensemble_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.ensemble_dim = ensemble_dim

        self.weight = nn.Parameter(torch.randn(ensemble_dim, output_dim, input_dim))
        self.bias = nn.Parameter(torch.randn(ensemble_dim, output_dim))

        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        """ Output size=[..., ensemble_dim, output_dim] """
        out = torch.einsum("koi, ...ki -> ...ko", self.weight, x) + self.bias
        return out


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


class DoubleQNetwork(nn.Module):
    """ Double Q network for continuous actions """
    def __init__(self, obs_dim, act_dim, hidden_dim, num_hidden, activation="silu"):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.q1 = MLP(
            input_dim=obs_dim + act_dim,
            output_dim=1,
            hidden_dim=hidden_dim,
            num_hidden=num_hidden,
            activation=activation,
        )
        self.q2 = MLP(
            input_dim=obs_dim + act_dim,
            output_dim=1,
            hidden_dim=hidden_dim,
            num_hidden=num_hidden,
            activation=activation,
        )
    
    def __repr__(self):
        s = "{}(input_dim={}, hidden_dim={}, num_hidden={}, activation={})".format(
            self.__class__.__name__, self.obs_dim + self.act_dim, self.q1.hidden_dim, 
            self.q1.num_hidden, self.q1.activation
        )
        return s

    def forward(self, o, a):
        """ Compute q1 and q2 values
        
        Args:
            o (torch.tensor): observation. size=[batch_size, obs_dim]
            a (torch.tensor): action. size=[batch_size, act_dim]

        Returns:
            q1 (torch.tensor): q1 value. size=[batch_size, 1]
            q2 (torch.tensor): q2 value. size=[batch_size, 1]
        """
        oa = torch.cat([o, a], dim=-1)
        q1 = self.q1(oa)
        q2 = self.q2(oa)
        return q1, q2

if __name__ == "__main__":
    torch.manual_seed(0)
    
    input_dim = 10
    output_dim = 2
    ensemble_dim = 5
    
    # synthetic input
    batch_size = 32
    x = torch.randn(batch_size, input_dim)
    
    # test ensemble linear
    ensemble_lin = EnsembleLinear(input_dim, output_dim, ensemble_dim)
    out = ensemble_lin(x.unsqueeze(-2).repeat_interleave(ensemble_dim, dim=-2))
    assert list(out.shape) == [batch_size, ensemble_dim, output_dim]
    print("EnsembleLinear passed")
    
    # test ensemble mlp
    hidden_dim = 64
    num_hidden = 2
    activation = "relu"

    ensemble_mlp = EnsembleMLP(
        input_dim, output_dim, ensemble_dim, hidden_dim, num_hidden, activation
    )
    out = ensemble_mlp(x)
    assert list(out.shape) == [batch_size, ensemble_dim, output_dim]
    print("EnsembleMLP passed")

    