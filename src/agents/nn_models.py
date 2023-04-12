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
    hidden_dim = 64
    num_hidden = 2
    activation = "relu"
    
    # synthetic input
    batch_size = 32
    x = torch.randn(batch_size, input_dim)
    x_separete = torch.randn(batch_size, ensemble_dim, input_dim)
    y_separate = torch.randn(batch_size, ensemble_dim, output_dim)
    
    def test_ensemble_linear():
        ensemble_lin = EnsembleLinear(input_dim, output_dim, ensemble_dim)

        # test shapes
        out = ensemble_lin(x.unsqueeze(-2).repeat_interleave(ensemble_dim, dim=-2))
        out_separete = ensemble_lin(x_separete)

        assert list(out.shape) == [batch_size, ensemble_dim, output_dim]
        assert list(out_separete.shape) == [batch_size, ensemble_dim, output_dim]
        
        out_separate_true = torch.cat([
            (x_separete[:, i].matmul(ensemble_lin.weight[i]) + ensemble_lin.bias[i]).unsqueeze(-2) 
            for i in range(ensemble_dim)
        ], dim=-2)
        assert torch.all(torch.isclose(out_separete, out_separate_true, atol=1e-5))

        # test gradients: drop one member and check no gradients
        loss = torch.pow(out_separete[:, :-1] - y_separate[:, :-1], 2).sum()
        loss.backward()
        
        assert torch.all(ensemble_lin.weight.grad[:-1] != 0)
        assert torch.all(ensemble_lin.weight.grad[-1] == 0)
        assert torch.all(ensemble_lin.bias.grad[:-1] != 0)
        assert torch.all(ensemble_lin.bias.grad[-1] == 0)

        # compare with separate gradients
        weight = nn.Parameter(ensemble_lin.weight.data)
        bias = nn.Parameter(ensemble_lin.bias.data)
        
        for i in range(ensemble_dim - 1):
            loss_i = torch.pow(x_separete[:, i].matmul(weight[i]) + bias[i] - y_separate[:, i], 2).sum()
            loss_i.backward()
            
            assert torch.isclose(weight.grad[i], ensemble_lin.weight.grad[i], atol=1e-5).all()

        print("EnsembleLinear passed")
    
    def test_ensemble_mlp():
        ensemble_mlp = EnsembleMLP(
            input_dim, output_dim, ensemble_dim, hidden_dim, num_hidden, activation
        )

        # test shapes
        out = ensemble_mlp(x)
        out_separete = ensemble_mlp.forward_separete(x_separete)

        assert list(out.shape) == [batch_size, ensemble_dim, output_dim]
        assert list(out_separete.shape) == [batch_size, ensemble_dim, output_dim]

        # test gradients: drop one member and check no gradients
        loss = torch.pow(out_separete[:, :4] - y_separate[:, :4], 2).mean()
        loss.backward()
        
        for n, p in ensemble_mlp.named_parameters():
            if "weight" or "bias" in n:
                assert torch.all(p.grad[-1] == 0)

        print("EnsembleMLP passed")
    
    test_ensemble_linear()
    test_ensemble_mlp()

    