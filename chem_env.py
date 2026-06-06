"""
Molecular Search Environment (Gymnasium)
Goal: Navigate chemical space to find molecules with dipole moment in [TARGET_LOW, TARGET_HIGH]
State : Morgan Fingerprint (2048-dim)
Action: Select one of N structurally similar neighbor molecules
Reward: Proximity to target dipole value
"""

import numpy as np
import joblib
import gymnasium as gym
from gymnasium import spaces

TARGET_LOW   = 2.0
TARGET_HIGH  = 3.0
TARGET_CENTER = (TARGET_LOW + TARGET_HIGH) / 2.0  # 2.5


class MoleculeEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        fps_path: str = "data/fps.npy",
        surrogate_path: str = "models/surrogate_rf.pkl",
        n_neighbors: int = 10,
        max_steps: int = 50,
        max_subset_size: int = 20000,
        sample_size: int = 300,
    ):
        super().__init__()

        # Load and slice fingerprint data
        raw_fps = np.load(fps_path).astype(np.float32)
        self.fps = raw_fps[:max_subset_size]
        self.n_molecules = len(self.fps)
        self.n_neighbors = n_neighbors
        self.max_steps = max_steps
        self.sample_size = sample_size

        # Predict dipole moments using surrogate model
        print(f"[INFO] Dataset: {self.n_molecules} molecules")
        print(f"[INFO] Target range: [{TARGET_LOW}, {TARGET_HIGH}] Debye")
        surrogate = joblib.load(surrogate_path)
        self.dipoles = surrogate.predict(self.fps).astype(np.float32)

        # Precompute neighbor map using Tanimoto similarity sampling
        print("[INFO] Precomputing neighbor map...")
        self._precompute_neighbors()
        print("[INFO] Ready.")

        self.action_space = spaces.Discrete(n_neighbors)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.fps.shape[1],), dtype=np.float32)

        self.current_idx = None
        self.current_dipole = None
        self.step_count = None

    def _precompute_neighbors(self):
        np.random.seed(42)
        fps_sum = np.sum(self.fps, axis=1)
        self.neighbor_map = {}

        for i in range(self.n_molecules):
            if i % 5000 == 0:
                print(f"  -> {i}/{self.n_molecules}")

            candidates = np.random.choice(self.n_molecules, size=self.sample_size, replace=False)
            candidates = candidates[candidates != i]

            dot = self.fps[candidates] @ self.fps[i]
            sim = dot / (fps_sum[candidates] + fps_sum[i] - dot + 1e-8)

            top = np.argsort(sim)[-self.n_neighbors:]
            self.neighbor_map[i] = list(candidates[top])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_idx = int(self.np_random.integers(0, self.n_molecules))
        self.current_dipole = float(self.dipoles[self.current_idx])
        self.step_count = 0

        return self.fps[self.current_idx], self._info()

    def step(self, action: int):
        next_idx = self.neighbor_map[self.current_idx][action]
        next_dipole = float(self.dipoles[next_idx])

        # Reward: closer to TARGET_CENTER = higher reward
        in_target = TARGET_LOW <= next_dipole <= TARGET_HIGH
        if in_target:
            reward = 1.0 - abs(next_dipole - TARGET_CENTER) / 0.5
        else:
            reward = -abs(next_dipole - TARGET_CENTER) / 5.0

        self.current_idx = next_idx
        self.current_dipole = next_dipole
        self.step_count += 1

        return self.fps[self.current_idx], reward, False, self.step_count >= self.max_steps, self._info()

    def _info(self):
        return {
            "current_idx": self.current_idx,
            "current_dipole": self.current_dipole,
            "in_target": TARGET_LOW <= self.current_dipole <= TARGET_HIGH,
            "step": self.step_count,
        }

    def render(self):
        print(f"Step {self.step_count:3d} | Dipole: {self.current_dipole:.4f} | In Target: {TARGET_LOW <= self.current_dipole <= TARGET_HIGH}")
