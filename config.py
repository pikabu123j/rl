CONFIG = {
    "state_dim" : 12,
    "action_dim" : 4,
    "hidden_dim": 128,      # Hidden units in the neural network
    "learning_rate": 1e-3,
    "gamma": 0.99,          # Discount factor
    "epsilon_start": 1.0,   # Initial exploration
    "epsilon_end": 0.01,    # Minimum exploration
    "epsilon_decay": 0.995, # Decay rate of epsilon per episode
    "memory_size": 10000,
    "batch_size": 64,
    "target_update_freq": 10,
    "num_episodes": 500,
    "save_path": "./checkpoints/",
    "log_path": "./logs/",
}
