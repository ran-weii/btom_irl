import numpy as np

def rollout_parallel(env, agent, max_steps):
    s = env.reset()
    data = {"s": [s], "a": [], "r": []}
    for t in range(max_steps):
        a = agent.choose_action(s)
        s, r, done, info = env.step(a)
        
        if any(done) or (t > max_steps):
            break

        data["s"].append(s)
        data["a"].append(a)
        data["r"].append(r)

    data["s"] = np.stack(data["s"])
    data["a"] = np.stack(data["a"])
    data["r"] = np.stack(data["r"])
    return data