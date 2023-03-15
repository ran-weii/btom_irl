import numpy as np
import torch

def rollout_parallel(env, agent, max_steps):
    s = env.reset()
    data = {"s": [s], "a": [], "r": []}
    for t in range(max_steps):
        with torch.no_grad():
            a = agent.choose_action(
                torch.from_numpy(s).to(torch.float32)
            ).numpy()
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