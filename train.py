# train.py

from agent import DQNAgent
from env import SDWANLatencyEnv
from utils import save_checkpoint
from config import CONFIG
from torch.utils.tensorboard import SummaryWriter

def train():
    env = SDWANLatencyEnv()
    agent = DQNAgent(CONFIG['state_dim'], CONFIG['action_dim'])
    writer = SummaryWriter(CONFIG["log_path"])

    for episode in range(CONFIG['num_episodes']):
        state = env.reset()
        total_reward = 0
        total_latency = 0
        done = False
        steps = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            # Log latency of selected action
            latency = env.last_latency[action]
            total_latency += latency
            steps += 1

            agent.store_transition(state, action, reward, next_state, done)
            agent.learn()
            state = next_state
            total_reward += reward

        avg_latency = total_latency / steps if steps > 0 else 0

        # TensorBoard logs
        writer.add_scalar("Reward/Total", total_reward, episode)
        writer.add_scalar("Latency/Average", avg_latency, episode)
        writer.add_scalar("Exploration/Epsilon", agent.epsilon, episode)

        if episode % CONFIG['target_update_freq'] == 0:
            agent.update_target_network()
            save_checkpoint(agent.q_network, agent.optimizer, episode, CONFIG['save_path'])

        print(f"Episode {episode}, Reward: {total_reward:.2f}, Avg Latency: {avg_latency:.2f} ms, Epsilon: {agent.epsilon:.2f}")

    writer.close()

if __name__ == "__main__":
    train()

