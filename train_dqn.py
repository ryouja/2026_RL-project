import os
import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from chem_env import MoleculeEnv, TARGET_LOW, TARGET_HIGH

BATCH_SIZE = 64
LR = 0.0005
GAMMA = 0.99
MEMORY_SIZE = 10000
MIN_REPLAY_SIZE = 500
EPISODES = 500
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.995
TARGET_UPDATE_FREQ = 10


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, s, a, r, ns, done):
        self.buffer.append((s, a, r, ns, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)

        return (
            np.array(s, dtype=np.float32),
            np.array(a, dtype=np.int64),
            np.array(r, dtype=np.float32),
            np.array(ns, dtype=np.float32),
            np.array(d, dtype=np.bool_),
        )

    def __len__(self):
        return len(self.buffer)


def save_plots(rewards, target_hits):
    os.makedirs("results", exist_ok=True)
    window = max(1, len(rewards) // 20)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    reward_avg = np.convolve(rewards, np.ones(window) / window, mode="valid")
    hit_avg = np.convolve(target_hits, np.ones(window) / window, mode="valid")

    axes[0].plot(rewards, alpha=0.3)
    axes[0].plot(reward_avg, label=f"avg {window}")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Reward")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(target_hits, alpha=0.3)
    axes[1].plot(hit_avg, label=f"avg {window}")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel(f"Steps in [{TARGET_LOW}~{TARGET_HIGH}] Debye")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/training_metrics.png", dpi=150)
    plt.close()


def main():
    env = MoleculeEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    q_net = QNetwork(state_dim, action_dim).to(device)
    target_net = QNetwork(state_dim, action_dim).to(device)
    target_net.load_state_dict(q_net.state_dict())

    optimizer = optim.Adam(q_net.parameters(), lr=LR)
    buffer = ReplayBuffer(MEMORY_SIZE)
    epsilon = EPS_START

    episode_rewards = []
    episode_target_hits = []

    print("=" * 70)
    print(f"{'Ep':<8}{'Reward':<14}{'Target Hits':<14}{'Dipole':<14}{'Eps':<8}")
    print("=" * 70)

    for ep in range(1, EPISODES + 1):
        state, info = env.reset()
        total_reward = 0.0
        target_hits = 0

        while True:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    st = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                    action = q_net(st).argmax().item()

            next_state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if info["in_target"]:
                target_hits += 1

            buffer.push(state, action, reward, next_state, truncated)
            state = next_state

            if len(buffer) >= MIN_REPLAY_SIZE:
                s, a, r, ns, d = buffer.sample(BATCH_SIZE)

                s = torch.tensor(s).to(device)
                a = torch.tensor(a).unsqueeze(1).to(device)
                r = torch.tensor(r).to(device)
                ns = torch.tensor(ns).to(device)
                d = torch.tensor(d).to(device)

                current_q = q_net(s).gather(1, a).squeeze(1)

                with torch.no_grad():
                    next_q = target_net(ns).max(1)[0]
                    target_q = r + GAMMA * next_q * (~d)

                loss = nn.MSELoss()(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if terminated or truncated:
                break

        epsilon = max(EPS_END, epsilon * EPS_DECAY)

        if ep % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(q_net.state_dict())

        episode_rewards.append(total_reward)
        episode_target_hits.append(target_hits)

        if ep % 10 == 0 or ep == 1:
            print(f"{ep:<8}{total_reward:<14.4f}{target_hits:<14}{info['current_dipole']:<14.4f}{epsilon:<8.3f}")

    print("=" * 70)

    os.makedirs("models", exist_ok=True)
    torch.save(q_net.state_dict(), "models/dqn_molecule_model.pth")

    os.makedirs("results", exist_ok=True)
    np.save("results/dqn_rewards.npy", np.array(episode_rewards))
    np.save("results/dqn_target_hits.npy", np.array(episode_target_hits))

    save_plots(episode_rewards, episode_target_hits)


if __name__ == "__main__":
    main()
