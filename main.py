from env import *
from dqn import *
from per import *


def main(*game_names):

    import cv2
    import gym
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from gym.spaces import Box
    from IPython.display import clear_output
    from torch import nn, optim

    np.random.seed(1)
    torch.manual_seed(1)

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    cv2.ocl.setUseOpenCL(False)
    T = 700_000

    for game_name in game_names:
        env = Env(game_name)
        env.seed(1)

        # double DQN, primary network qnet
        qnet = DQN(env.action_space.n).to(DEVICE)
        # the target network
        tnet = DQN(env.action_space.n).to(DEVICE)
        optimizer = optim.Adam(qnet.parameters(), lr=0.0001)

        per = PER()
        tnet.load_state_dict(qnet.state_dict())  # Synchronize policies

        epsilon, k, gamma = 0.01, 32, 0.99
        rewards, reward, plot_rewards = list(), 0, list()

        s = env.reset()
        for t in range(1, T + 1):
            a = qnet.action_selection(s, epsilon)
            s_prime, r, done, _ = env.step(a)
            per.store(s, a, r, s_prime, done)
            s = s_prime
            reward += r
            if done:
                s = env.reset()
                rewards.append(reward)
                reward = 0
            if len(per.rep_memory) > (1e4):
                beta = min(1.0, 0.4 + t * 0.6 / (1e5) )  # Linear annealing
                # TD Loss
                S, A, R, S_prime, Done, W, idx = per.replay(
                    k, beta)

                S = torch.tensor(S, dtype=torch.float32,
                                 requires_grad=True, device=DEVICE)
                S_prime = torch.tensor(
                    S_prime, dtype=torch.float32, requires_grad=True, device=DEVICE)
                A = torch.tensor(A, dtype=torch.long, device=DEVICE)
                R = torch.tensor(R, dtype=torch.long, device=DEVICE)
                Done = torch.tensor(Done, dtype=torch.long, device=DEVICE)
                W = torch.tensor(W, dtype=torch.long, device=DEVICE)
                Q_qnet = qnet(S).gather(1, A.unsqueeze(1)).squeeze(1)
                loss = (Q_qnet - (R + gamma * tnet(S_prime).max(1)
                                  [0] * (1 - Done)).detach()) ** 2 * W
                priorities = loss + 1e-5
                loss = loss.mean()
                optimizer.zero_grad()
                loss.backward()
                per.priorities[idx] = priorities.detach().cpu().numpy()
                optimizer.step()

            if t % (T / 200) == 0:
                plot_rewards.append(np.mean(rewards[-10:]))
                clear_output(True)
                plt.figure(figsize=(12,8))
                plt.plot(plot_rewards, 'g', linewidth=2, markersize=12)
                plt.title(game_name)
                plt.title(game_name, fontsize=14)
                plt.xlabel(f"Training step / {T // 200}", fontsize=12)
                plt.ylabel("reward", fontsize=12)
                plt.savefig(game_name + ".png")
                plt.show()

            if t % (T / 10) == 0:
                tnet.load_state_dict(qnet.state_dict())  # Synchronize policies


if __name__ == "__main__":
    main("Pitfall")