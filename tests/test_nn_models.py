import torch
import torch.nn as nn
from src.agents.nn_models import EnsembleLinear, EnsembleMLP

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

    test_ensemble_linear()
    print("EnsembleLinear passed")
    
    test_ensemble_mlp()
    print("EnsembleMLP passed")

    