import torch

def value_iteration(transition, reward, gamma, alpha, max_iter=1000, tol=1e-5):
    """ Infinite horizon discounted soft value iteration

    Args:
        transition (torch.tensor): transition matrix. size=[act_dim, state_dim, state_dim]
        reward (torch.tensor): reward vector. size=[state_dim, act_dim]
        gamma (float): discount factor.
        alpha (float): softmax temperature.
        max_iter (int): max iterations. Default=1000
        tol (float): stopping tolerance. Default=1e-5

    Returns:
        q (torch.tensor): final Q function. size=[state_dim, act_dim]
        error (float): stopping bellman error
    """
    assert torch.all(torch.isclose(transition.sum(-1), torch.ones(1)))
    assert len(reward.shape) == 2
    state_dim = transition.shape[-1]
    act_dim = transition.shape[0]
    
    q = [torch.zeros(state_dim, act_dim)] + [torch.empty(0)] * (max_iter)
    for i in range(max_iter):
        v = torch.logsumexp(alpha * q[i], dim=-1) / alpha
        ev = torch.einsum("kij, j -> ik", transition, v)
        q[i+1] = reward + gamma * ev
        
        error = torch.norm(q[i+1] - q[i])
        if error < tol:
            q = q[:i+1]
            break
    return q[-1], error.data.item()

if __name__ == "__main__":
    torch.manual_seed(0)
    state_dim = 10
    act_dim = 3
    gamma = 0.9
    alpha = 1.
    
    transition = torch.softmax(torch.randn(act_dim, state_dim, state_dim), dim=-1)
    reward = torch.rand(state_dim, act_dim)

    q, error = value_iteration(transition, reward, gamma, alpha)

    assert list(q.shape) == [state_dim, act_dim]
    print("value iteration passed, tol={:.6f}".format(error))
