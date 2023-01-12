import torch

def finite_horizon_value_iteration(transition, reward, gamma, alpha, horizon):
    """ Finite horizon discounted soft value iteration

    Args:
        transition (torch.tensor): transition matrix. size=[act_dim, state_dim, state_dim]
        reward (torch.tensor): reward vector. size=[state_dim, act_dim]
        gamma (float): discount factor
        alpha (float): softmax temperature
        horizon (int): planning horizon

    Returns:
        q (torch.tensor): Q functions. size=[horizon + 1, state_dim, act_dim]
    """
    assert torch.all(torch.isclose(transition.sum(-1), torch.ones(1)))
    assert len(reward.shape) == 2
    
    q = [reward] + [torch.empty(0)] * (horizon)
    for i in range(horizon):
        v = torch.logsumexp(alpha * q[i], dim=-1) / alpha
        ev = torch.einsum("kij, j -> ik", transition, v)
        q[i+1] = reward + gamma * ev
    q = torch.stack(q)
    return q

def infinite_horizon_value_iteration(transition, reward, gamma, alpha, max_iter=1000, tol=1e-5):
    """ Infinite horizon discounted soft value iteration

    Args:
        transition (torch.tensor): transition matrix. size=[act_dim, state_dim, state_dim]
        reward (torch.tensor): reward vector. size=[state_dim, act_dim]
        gamma (float): discount factor
        alpha (float): softmax temperature
        max_iter (int): max iterations. Default=1000
        tol (float): stopping tolerance. Default=1e-5

    Returns:
        q (torch.tensor): final Q function. size=[state_dim, act_dim]
        error (float): stopping bellman error
    """
    assert torch.all(torch.isclose(transition.sum(-1), torch.ones(1)))
    assert len(reward.shape) == 2
    
    q = [reward] + [torch.empty(0)] * (max_iter)
    for i in range(max_iter):
        v = torch.logsumexp(alpha * q[i], dim=-1) / alpha
        ev = torch.einsum("kij, j -> ik", transition, v)
        q[i+1] = reward + gamma * ev
        
        error = torch.norm(q[i+1] - q[i])
        if error < tol:
            q = q[:i+1]
            break
    q = torch.stack(q)
    return q[-1], error.data.item()

def finite_horizon_riccati_equation(A, B, I, Q, R, gamma, alpha, horizon):
    """ Finite horizon riccati equation 
    
    Args:
        A (torch.tensor): transition matrix A. size=[state_dim, state_dim]
        B (torch.tensor): control matrix B. size=[state_dim, act_dim]
        I (torch.tensor): noise covariance matrix I. size=[state_dim, state_dim]
        Q (torch.tensor): state cost matrix Q. size=[state_dim, state_dim]
        R (torch.tensor): control cost matrix R. size=[act_dim, act_dim]
        gamma (float): discount factor
        alpha (float): softmax temperature
        horizon (int): planning horizon
    
    Returns:
        Q_t (torch.tensor): value quadratic matrices. size=[horizon + 1, state_dim, state_dim]
        c_t (torch.tensor): value constants. size=[horizon + 1, 1]
    """
    d = A.shape[-1]
    d2_log_2pi = d/2*torch.log(2*torch.pi*torch.ones(1))

    Q_t = [Q] + [torch.empty(0)] * horizon
    c_t = [torch.zeros(1)] + [torch.empty(0)] * horizon
    for t in range(horizon):
        aqa = A.T.matmul(Q_t[t]).matmul(A)
        aqb = A.T.matmul(Q_t[t]).matmul(B)
        bqa = B.T.matmul(Q_t[t]).matmul(A)

        rbqb = R + gamma * B.T.matmul(Q_t[t]).matmul(B)
        rbqb_inv = torch.linalg.inv(rbqb)
        rbqb_det = torch.linalg.det(rbqb)

        qi_tr = torch.trace(gamma * Q_t[t].matmul(I))

        Q_t[t+1] = Q + gamma * aqa - gamma ** 2 * aqb.matmul(rbqb_inv).matmul(bqa)
        c_t[t+1] = 0.5 * alpha * torch.log(rbqb_det) + 0.5 * qi_tr + gamma * c_t[t] - alpha * d2_log_2pi
    
    Q_t = torch.stack(Q_t)
    c_t = torch.stack(c_t)
    return Q_t, c_t

def infinite_horizon_riccati_equation(A, B, I, Q, R, gamma, alpha, max_iter=1000, tol=1e-5):
    """ Infinite horizon riccati equation 
    
    Args:
        A (torch.tensor): transition matrix A. size=[state_dim, state_dim]
        B (torch.tensor): control matrix B. size=[state_dim, act_dim]
        I (torch.tensor): noise covariance matrix I. size=[state_dim, state_dim]
        Q (torch.tensor): state cost matrix Q. size=[state_dim, state_dim]
        R (torch.tensor): control cost matrix R. size=[act_dim, act_dim]
        gamma (float): discount factor
        alpha (float): softmax temperature
        max_iter (int): max iterations. Default=1000
        tol (float): stopping tolerance. Default=1e-5
    
    Returns:
        Q_t (torch.tensor): value quadratic matrices. size=[state_dim, state_dim]
        c_t (torch.tensor): value constants. size=[1]
        error (float): stopping bellman error
    """
    d = A.shape[-1]
    d2_log_2pi = d/2*torch.log(2*torch.pi*torch.ones(1))

    Q_t = [Q] + [torch.empty(0)] * max_iter
    c_t = [torch.zeros(1)] + [torch.empty(0)] * max_iter
    for t in range(max_iter):
        aqa = A.T.matmul(Q_t[t]).matmul(A)
        aqb = A.T.matmul(Q_t[t]).matmul(B)
        bqa = B.T.matmul(Q_t[t]).matmul(A)

        rbqb = R + gamma * B.T.matmul(Q_t[t]).matmul(B)
        rbqb_inv = torch.linalg.inv(rbqb)
        rbqb_det = torch.linalg.det(rbqb)

        qi_tr = torch.trace(gamma * Q_t[t].matmul(I))

        Q_t[t+1] = Q + gamma * aqa - gamma ** 2 * aqb.matmul(rbqb_inv).matmul(bqa)
        c_t[t+1] = 0.5 * alpha * torch.log(rbqb_det) + 0.5 * qi_tr + gamma * c_t[t] - alpha * d2_log_2pi

        error = torch.norm(c_t[t+1] - c_t[t])
        if error < tol:
            Q_t = Q_t[:t+1]
            c_t = c_t[:t+1]
            break

    Q_t = torch.stack(Q_t)
    c_t = torch.stack(c_t)
    return Q_t[-1], c_t[-1], error

if __name__ == "__main__":
    torch.manual_seed(0)

    # test value iteration
    state_dim = 10
    act_dim = 3
    gamma = 0.9
    alpha = 1.
    horizon = 10
    
    transition = torch.softmax(torch.randn(act_dim, state_dim, state_dim), dim=-1)
    reward = torch.rand(state_dim, act_dim)
    
    q_finite = finite_horizon_value_iteration(transition, reward, gamma, alpha, horizon)
    q_infinite, error = infinite_horizon_value_iteration(transition, reward, gamma, alpha)

    assert list(q_finite.shape) == [horizon + 1, state_dim, act_dim]
    assert list(q_infinite.shape) == [state_dim, act_dim]
    print("finite and infinite horizon value iteration passed, tol={:.6f}".format(error))

    # test riccati equation
    dt = 0.1
    A = torch.tensor([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    B = torch.tensor([
        [dt**2, 0],
        [0, dt**2],
        [1, 0],
        [0, 1]
    ])
    I = torch.diag(0.1 * torch.tensor([1, 1, 1, 1]))
    Q = torch.diag(torch.tensor([10., 10., 0., 0.]))
    R = torch.diag(torch.tensor([0., 0.]))
    gamma = 0.7
    alpha = 1.
    horizon = 10
    state_dim = 4
    act_dim = 2
    
    Q_t_finite, c_t_finite = finite_horizon_riccati_equation(A, B, I, Q, R, gamma, alpha, horizon)
    Q_t_infinite, c_t_infinite, error = infinite_horizon_riccati_equation(A, B, I, Q, R, gamma, alpha)
    
    assert list(Q_t_finite.shape) == [horizon + 1, state_dim, state_dim]
    assert list(c_t_finite.shape) == [horizon + 1, 1]
    assert list(Q_t_infinite.shape) == [state_dim, state_dim]
    assert list(c_t_infinite.shape) == [1]
    print("finite and infinite horizon riccati equation passed, tol={:.6f}".format(error))
