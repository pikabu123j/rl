import numpy as np
class SDWANLatencyEnv:
    def __init__(self):
        self.num_paths = 4
        self.state_dim = self.num_paths * 3  # Each path has latency, jitter, loss
        self.action_dim = self.num_paths
        self.reset()

    def _simulate_metrics(self):
        latency = np.random.uniform(10, 200, self.num_paths)
        jitter = np.random.uniform(0, 20, self.num_paths)
        loss = np.random.uniform(0, 0.1, self.num_paths)
        return latency, jitter, loss

    def _build_state(self):
        latency, jitter, loss = self._simulate_metrics()
        self.last_latency = latency
        return np.concatenate([latency, jitter, loss])

    def reset(self):
        self.step_count = 0
        self.max_steps = 50
        return self._build_state()

    def step(self, action):
        latency = self.last_latency[action]
        reward = -latency / 100.0  # Lower latency = higher reward
        done = self.step_count >= self.max_steps
        next_state = self._build_state()
        self.step_count += 1
        return next_state, reward, done, {}
