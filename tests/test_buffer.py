import numpy as np
from src.agents.buffer import ReplayBuffer

def test_replay_buffer():
    buffer = ReplayBuffer(obs_dim, act_dim, max_size, momentum=0.)
    buffer.push_batch(obs, act, rwd, next_obs, done)
    
    assert np.isclose(buffer.obs, obs, atol=tol).all()
    assert np.isclose(buffer.act, act, atol=tol).all()
    assert np.isclose(buffer.rwd, rwd, atol=tol).all()
    assert np.isclose(buffer.next_obs, next_obs, atol=tol).all()
    assert np.isclose(buffer.done, done, atol=tol).all()

    assert buffer.size == len(obs)
    assert np.isclose(buffer.obs_mean, obs_mean, atol=tol).all()
    assert np.isclose(buffer.obs_variance, obs_var, atol=tol).all()
    assert np.isclose(buffer.rwd_mean, rwd_mean, atol=tol).all()
    assert np.isclose(buffer.rwd_variance, rwd_var, atol=tol).all()

    # test limit exceed
    buffer.push_batch(obs_new, act_new, rwd_new, next_obs_new, done_new)
    
    obs_stack = np.vstack([obs, obs_new])[-buffer.max_size:]
    act_stack = np.vstack([act, act_new])[-buffer.max_size:]
    rwd_stack = np.vstack([rwd, rwd_new])[-buffer.max_size:]
    next_obs_stack = np.vstack([next_obs, next_obs_new])[-buffer.max_size:]
    done_stack = np.vstack([done, done_new])[-buffer.max_size:]
    
    assert buffer.size == buffer.max_size
    assert np.isclose(buffer.obs, obs_stack).all()
    assert np.isclose(buffer.act, act_stack).all()
    assert np.isclose(buffer.rwd, rwd_stack).all()
    assert np.isclose(buffer.next_obs, next_obs_stack).all()
    assert np.isclose(buffer.done, done_stack).all()

if __name__ == "__main__":
    np.random.seed(0)
    
    tol = 1e-5
    obs_dim = 11
    act_dim = 3
    max_size = 500
    
    batch_size = 256
    obs = np.random.normal(size=(batch_size, obs_dim))
    act = np.random.normal(size=(batch_size, act_dim))
    rwd = np.random.normal(size=(batch_size, 1))
    next_obs = np.random.normal(size=(batch_size, obs_dim))
    done = np.random.randint(0, 2, size=(batch_size, 1))

    obs_mean = obs.mean(0)
    obs_var = obs.var(0)
    rwd_mean = rwd.mean(0)
    rwd_var = rwd.var(0)

    obs_new = np.random.normal(size=(batch_size, obs_dim))
    act_new = np.random.normal(size=(batch_size, act_dim))
    rwd_new = np.random.normal(size=(batch_size, 1))
    next_obs_new = np.random.normal(size=(batch_size, obs_dim))
    done_new = np.random.randint(0, 2, size=(batch_size, 1))

    test_replay_buffer()
    print("test_replay_buffer")



