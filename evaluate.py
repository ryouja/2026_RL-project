"""
Evaluation: Compare Random / Greedy / DQN
Metric: Number of steps landing in target dipole range [2.0, 3.0] Debye
Also shows DQN trajectory (path) from start to target molecules
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from chem_env import MoleculeEnv, TARGET_LOW, TARGET_HIGH, TARGET_CENTER


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 128),       nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)


# ── Search strategies ──────────────────────────────────────

def run_random(env, n_episodes):
    results = []
    for _ in range(n_episodes):
        _, _ = env.reset()
        hits = 0
        while True:
            _, _, _, truncated, info = env.step(env.action_space.sample())
            if info["in_target"]: hits += 1
            if truncated: break
        results.append(hits)
    return results


def run_greedy(env, n_episodes):
    results = []
    for _ in range(n_episodes):
        _, _ = env.reset()
        hits = 0
        while True:
            neighbors = env.neighbor_map[env.current_idx]
            action = int(np.argmin([abs(env.dipoles[idx] - TARGET_CENTER) for idx in neighbors]))
            _, _, _, truncated, info = env.step(action)
            if info["in_target"]: hits += 1
            if truncated: break
        results.append(hits)
    return results


def run_dqn(env, q_net, device, n_episodes):
    q_net.eval()
    results = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        hits = 0
        while True:
            with torch.no_grad():
                st = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                action = q_net(st).argmax().item()
            state, _, _, truncated, info = env.step(action)
            if info["in_target"]: hits += 1
            if truncated: break
        results.append(hits)
    return results


# ── Step-tracking ──────────────────────────────────────────

def run_tracked(env, policy_fn, n_episodes):
    all_curves = []
    for _ in range(n_episodes):
        state, info = env.reset()
        curve = [1 if info["in_target"] else 0]
        while True:
            action = policy_fn(env, state)
            state, _, _, truncated, info = env.step(action)
            curve.append(1 if info["in_target"] else 0)
            if truncated: break
        all_curves.append(curve)
    min_len = min(len(c) for c in all_curves)
    return np.cumsum(np.mean([c[:min_len] for c in all_curves], axis=0))


# ── DQN Trajectory ─────────────────────────────────────────

def show_dqn_trajectory(env, q_net, device, smiles_list, n_episodes=3):
    """Show DQN trajectory: starting molecule and steps that hit the target range."""
    q_net.eval()
    print("\n" + "=" * 70)
    print(f"  DQN Trajectory  (target: [{TARGET_LOW}, {TARGET_HIGH}] Debye)")
    print(f"  Only steps landing in target range are shown.")
    print("=" * 70)

    for ep in range(n_episodes):
        state, info = env.reset()
        start_smiles = smiles_list[info["current_idx"]]
        start_dipole = info["current_dipole"]
        hits = []

        while True:
            with torch.no_grad():
                st = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                action = q_net(st).argmax().item()
            state, _, _, truncated, info = env.step(action)

            if info["in_target"]:
                hits.append({
                    "step": info["step"],
                    "smiles": smiles_list[info["current_idx"]],
                    "dipole": info["current_dipole"],
                })
            if truncated: break

        # Print episode summary
        start_short = start_smiles[:45] + "..." if len(start_smiles) > 45 else start_smiles
        print(f"\n[Episode {ep+1}]")
        print(f"  Start : {start_short}")
        print(f"          dipole = {start_dipole:.4f} Debye  ({'in target' if TARGET_LOW <= start_dipole <= TARGET_HIGH else 'out of target'})")
        print(f"  Total hits: {len(hits)} / 50 steps")
        print()

        if hits:
            print(f"  {'Step':<6} {'Dipole':>8}   SMILES")
            print("  " + "-" * 60)
            for h in hits:
                smiles_short = h["smiles"][:45] + "..." if len(h["smiles"]) > 45 else h["smiles"]
                print(f"  {h['step']:<6} {h['dipole']:>8.4f}   {smiles_short}")
        else:
            print("  (No steps landed in target range)")

    print("\n" + "=" * 70)


# ── Plotting ───────────────────────────────────────────────

def plot_results(random_r, greedy_r, dqn_r,
                 random_t, greedy_t, dqn_t):
    os.makedirs("results", exist_ok=True)
    window = 10

    fig, ax = plt.subplots(figsize=(9, 5))
    def smooth(arr): return np.convolve(arr, np.ones(window)/window, mode="valid")
    ax.plot(smooth(random_r), label="Random", color="gray",      linestyle="--")
    ax.plot(smooth(greedy_r), label="Greedy", color="orange")
    ax.plot(smooth(dqn_r),    label="DQN",    color="steelblue", linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Target Hits per Episode [{TARGET_LOW}~{TARGET_HIGH} D]")
    ax.set_title("Search Performance Comparison (per Episode)")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/comparison_episodes.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(random_t, label="Random", color="gray",      linestyle="--")
    ax.plot(greedy_t, label="Greedy", color="orange")
    ax.plot(dqn_t,    label="DQN",    color="steelblue", linewidth=2)
    ax.set_xlabel("Search Step")
    ax.set_ylabel(f"Cumulative Target Hits [{TARGET_LOW}~{TARGET_HIGH} D]")
    ax.set_title("Target Hit Progression per Step")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/comparison_steps.png", dpi=150)
    plt.close()

    print("[INFO] Plots saved: results/")


# ── Main ───────────────────────────────────────────────────

def main(model_path="models/dqn_molecule_model.pth", n_episodes=100, n_track=20):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    env = MoleculeEnv()
    smiles_list = pd.read_csv("data/smiles_list.csv")["smiles"].tolist()

    q_net = QNetwork(env.observation_space.shape[0], env.action_space.n).to(device)
    q_net.load_state_dict(torch.load(model_path, map_location=device))
    q_net.eval()

    # Episode-level evaluation
    print("[INFO] Episode-level evaluation...")
    random_r = run_random(env, n_episodes)
    greedy_r = run_greedy(env, n_episodes)
    dqn_r    = run_dqn(env, q_net, device, n_episodes)

    # Step-level tracking
    print("[INFO] Step-level tracking...")
    def random_policy(env, state): return env.action_space.sample()
    def greedy_policy(env, state):
        neighbors = env.neighbor_map[env.current_idx]
        return int(np.argmin([abs(env.dipoles[idx] - TARGET_CENTER) for idx in neighbors]))
    def dqn_policy(env, state):
        with torch.no_grad():
            st = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            return q_net(st).argmax().item()

    random_t = run_tracked(env, random_policy, n_track)
    greedy_t = run_tracked(env, greedy_policy, n_track)
    dqn_t    = run_tracked(env, dqn_policy,    n_track)

    print(f"\n===== Results (avg target hits / episode) =====")
    print(f"  Random : {np.mean(random_r):.2f} ± {np.std(random_r):.2f}")
    print(f"  Greedy : {np.mean(greedy_r):.2f} ± {np.std(greedy_r):.2f}")
    print(f"  DQN    : {np.mean(dqn_r):.2f} ± {np.std(dqn_r):.2f}")
    print(f"  Target : [{TARGET_LOW}, {TARGET_HIGH}] Debye")

    np.save("results/random_results.npy", np.array(random_r))
    np.save("results/greedy_results.npy", np.array(greedy_r))
    np.save("results/dqn_results.npy",    np.array(dqn_r))

    plot_results(random_r, greedy_r, dqn_r, random_t, greedy_t, dqn_t)

    # DQN trajectory
    show_dqn_trajectory(env, q_net, device, smiles_list, n_episodes=3)


if __name__ == "__main__":
    main()
