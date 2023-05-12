import numpy as np
import torch
from src.agents.dynamics import soft_clamp
from src.agents.dynamics import EnsembleDynamics
from src.agents.dynamics import set_ensemble_params, parse_ensemble_params
from src.agents.dynamics import get_random_index

def test_soft_clamp():
    _min = torch.Tensor([-1.])
    _max = torch.Tensor([1.])
    x1 = torch.Tensor([0.])
    x2 = torch.Tensor([1.5])
    x3 = torch.Tensor([-1.5])
    assert _min < soft_clamp(x1, _min, _max) < _max
    assert _min < soft_clamp(x2, _min, _max) < _max
    assert _min < soft_clamp(x3, _min, _max) < _max

def test_ensemble(residual=False, termination_fn=None, pred_rwd=False):
    dynamics = EnsembleDynamics(
        obs_dim,
        act_dim, 
        pred_rwd,
        ensemble_dim,
        topk,
        hidden_dim,
        num_hidden,
        activation,
        decay=decay,
        residual=residual,
        termination_fn=termination_fn,
        min_std=min_std,
        max_std=max_std
    )
    
    # synthetic data
    inputs = torch.cat([obs, act], dim=-1)
    targets = torch.cat([next_obs, rwd], dim=-1) if pred_rwd else next_obs
    inputs_sep = torch.cat([obs_sep, act_sep], dim=-1)
    targets_sep = torch.cat([next_obs_sep, rwd_sep], dim=-1) if pred_rwd else next_obs_sep

    # test internal variable shapes
    assert list(dynamics.topk_dist.shape) == [ensemble_dim]
    assert list(dynamics.min_lv.shape) == [obs_dim + pred_rwd]
    assert list(dynamics.max_lv.shape) == [obs_dim + pred_rwd]
    assert torch.isclose(dynamics.min_lv.exp() ** 0.5, min_std * torch.ones(1), atol=1e-5).all()
    assert torch.isclose(dynamics.max_lv.exp() ** 0.5, max_std * torch.ones(1), atol=1e-5).all()
    
    # test method output shapes on regular batch
    pred, _ = dynamics.compute_stats(inputs)
    dist = dynamics.compute_dists(obs, act)
    logp = dynamics.compute_log_prob(obs, act, next_obs, rwd)
    mix_logp = dynamics.compute_mixture_log_prob(obs, act, next_obs, rwd)
    sample = dynamics.sample_dist(obs, act)
    next_obs_step, rwd_step, done = dynamics.step(obs, act)
    
    assert list(pred.shape) == [batch_size, ensemble_dim, obs_dim + pred_rwd]
    assert list(dist.mean.shape) == [batch_size, ensemble_dim, obs_dim + pred_rwd]
    assert list(dist.variance.shape) == [batch_size, ensemble_dim, obs_dim + pred_rwd]
    assert list(logp.shape) == [batch_size, ensemble_dim, 1]
    assert list(mix_logp.shape) == [batch_size, 1]
    assert list(sample.shape) == [batch_size, obs_dim + pred_rwd]
    assert list(next_obs_step.shape) == [batch_size, obs_dim]
    assert list(rwd_step.shape) == [batch_size, 1]
    assert list(done.shape) == [batch_size, 1]
    if pred_rwd == False:
        assert torch.all(rwd_step == 0)
    
    # test method output shapes on separate batch
    pred_sep, _ = dynamics.compute_stats_separate(inputs_sep)
    assert list(pred_sep.shape) == [batch_size, ensemble_dim, obs_dim + pred_rwd]

    # test eval
    stats = dynamics.evaluate(inputs, targets)
    dynamics.update_topk_dist(stats)
    assert sum(dynamics.topk_dist == 0) == (dynamics.ensemble_dim - dynamics.topk)
    
    # test gradients: drop one member and check no gradients
    pred_sep, _ = dynamics.compute_stats_separate(inputs_sep)
    loss = torch.mean(pred_sep[:, :-1] - targets_sep[:, :-1])
    loss.backward()
    
    head_weight = dict(dynamics.named_parameters())["mlp.layers.6.weight"]
    head_bias = dict(dynamics.named_parameters())["mlp.layers.6.bias"]

    assert torch.all(head_weight.grad[-1] == 0)
    assert torch.all(head_bias.grad[-1] == 0)

    for n, p in dynamics.named_parameters():
        if p.grad is not None:
            p.grad.zero_()

def test_ensemble_parsing():
    def ensemble_equal(m1, m2):
        """ Check whether two ensembles have equal parameters """
        are_equal = True
        for ((n1, p1), (n2, p2)) in zip(m1.named_parameters(), m2.named_parameters()):
            if ("weight" in n1 or "bias" in n1) and "weight" in n2 or "bias" in n2:
                if torch.all(p1 != p2):
                    are_equal = False
                    break
        return are_equal
    
    pred_rwd = True
    ensemble_1 = EnsembleDynamics(
        obs_dim,
        act_dim, 
        pred_rwd,
        ensemble_dim,
        topk,
        hidden_dim,
        num_hidden,
        activation,
    )
    ensemble_2 = EnsembleDynamics(
        obs_dim,
        act_dim, 
        pred_rwd,
        ensemble_dim,
        topk,
        hidden_dim,
        num_hidden,
        activation,
    )
    assert not ensemble_equal(ensemble_1, ensemble_2)

    params_list = parse_ensemble_params(ensemble_1)
    set_ensemble_params(ensemble_2, params_list)
    assert ensemble_equal(ensemble_1, ensemble_2)

def test_get_random_index():
    idx = get_random_index(batch_size, ensemble_dim, bootstrap=True)
    obs_batch = obs[idx]

    assert all([len(np.unique(idx[:, i])) == batch_size for i in range(ensemble_dim)])
    assert np.any(idx[:, 0] != idx[:, 1]) # at least one is not the same

    for i in range(ensemble_dim):
        assert torch.isclose(obs_batch[:, i], obs[idx[:, i]]).all()

if __name__ == "__main__":
    from src.env.gym_wrapper import get_termination_fn
    np.random.seed(0)
    torch.manual_seed(0)

    obs_dim = 11
    act_dim = 3
    ensemble_dim = 7
    topk = 5
    hidden_dim = 200
    num_hidden = 2
    activation = "silu"
    decay = [0.000025, 0.00005, 0.000075, 0.0001]
    min_std = 1e-5
    max_std = 100
    termination_fn = get_termination_fn("Hopper-v4")

    # synthetic data
    batch_size = 1000
    obs = torch.randn(batch_size, obs_dim)
    act = torch.randn(batch_size, act_dim)
    rwd = torch.randn(batch_size, 1)
    next_obs = torch.randn(batch_size, obs_dim)

    obs_sep = torch.randn(batch_size, ensemble_dim, obs_dim)
    act_sep = torch.randn(batch_size, ensemble_dim, act_dim)
    rwd_sep = torch.randn(batch_size, ensemble_dim, 1)
    next_obs_sep = torch.randn(batch_size, ensemble_dim, obs_dim)

    test_soft_clamp()
    print("soft_clamp passed")
    
    test_ensemble(
        residual=False, 
        termination_fn=None,
        pred_rwd=False
    )
    
    test_ensemble(
        residual=False, 
        termination_fn=termination_fn,
        pred_rwd=True
    )

    test_ensemble(
        residual=True, 
        termination_fn=termination_fn,
        pred_rwd=True
    )
    print("test_ensemble passed")

    test_get_random_index()
    print("test_get_random_index passed")

    test_ensemble_parsing()
    print("test_ensemble_parsing passed")