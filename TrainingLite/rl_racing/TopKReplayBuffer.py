import heapq
import pickle
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
class TopKTrajectoryBuffer:
    def __init__(self, k: int = 20):
        self.k = k
        self.buffer = []  # List of (score, episode_id, episode_data)
        self.injected_ids = set()
        self.next_episode_id = 0

    def add_episode(self, episode_data, score):
        ep_id = self.next_episode_id
        self.next_episode_id += 1

        if len(self.buffer) < self.k:
            self.buffer.append((score, ep_id, episode_data))
            print(f"[TopKBuffer] Buffer size: {len(self.buffer)} | Added episode with score {score:.2f}")
        else:
            # Find the worst score in the current buffer
            min_score = min(self.buffer, key=lambda x: x[0])[0]
            if score > min_score:
                # Replace the worst episode
                self.buffer.append((score, ep_id, episode_data))
                old_ids = {ep_id for _, ep_id, _ in self.buffer}
                self.buffer = heapq.nlargest(self.k, self.buffer, key=lambda x: x[0])
                new_ids = {ep_id for _, ep_id, _ in self.buffer}
                removed_ids = old_ids - new_ids

                for removed_ep_id in removed_ids:
                    print(f"[TopKBuffer] Discarded episode {removed_ep_id}")

                print(f"[TopKBuffer] Added new episode with score {score:.2f}")
            else:
                pass
                # print(f"[TopKBuffer] Ignored episode with score {score:.2f} (below top-K)")



    def get_uninjected_episodes(self):
        return [(ep_id, episode) for (score, ep_id, episode) in self.buffer if ep_id not in self.injected_ids]

    def mark_injected(self, ep_id):
        self.injected_ids.add(ep_id)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.buffer, f)

    def load(self, path):
        with open(path, 'rb') as f:
            self.buffer = pickle.load(f)

class TopKTrajectoryCallback(BaseCallback):
    def __init__(self, topk_buffer: TopKTrajectoryBuffer, inject_every: int = 10000, verbose=0):
        super().__init__(verbose)
        self.topk_buffer = topk_buffer
        self.episode_buffers = []
        self.total_rewards = []
        self.episode_counter = 0
        self.last_injection_step = 0
        self.inject_every = inject_every

    def _on_training_start(self):
        n_envs = self.training_env.num_envs
        self.episode_buffers = [[] for _ in range(n_envs)]
        self.total_rewards = [0.0 for _ in range(n_envs)]

    def _on_step(self) -> bool:
        if self.locals is None or 'infos' not in self.locals:
            return True

        infos = self.locals['infos']
        dones = self.locals['dones']
        observations = self.locals['new_obs']
        actions = self.locals['actions']
        rewards = self.locals['rewards']

        for i in range(len(dones)):
            transition = (
                observations[i].copy(),
                actions[i].copy(),
                rewards[i].item(),
                observations[i].copy(),  # Placeholder for next_obs if not explicitly available
                dones[i].item()
            )
            self.episode_buffers[i].append(transition)
            self.total_rewards[i] += rewards[i].item()

            if dones[i]:
                self.episode_counter += 1
                # print(f"[TopKCallback] Episode {self.episode_counter} done. Total reward: {self.total_rewards[i]:.2f}")
                self.topk_buffer.add_episode(self.episode_buffers[i], self.total_rewards[i])
                self.episode_buffers[i] = []
                self.total_rewards[i] = 0.0

        current_step = self.num_timesteps
        if current_step - self.last_injection_step >= self.inject_every:
            # print("[TopKCallback] Re-injecting top-K episodes into replay buffer...")
            
            self.last_injection_step = current_step

            batch_size = self.model.n_envs  # Usually 16 with SubprocVecEnv
            batch_obs, batch_next_obs, batch_actions = [], [], []
            batch_rewards, batch_dones, batch_infos = [], [], []

            for ep_id, episode in self.topk_buffer.get_uninjected_episodes():
                for obs, action, reward, next_obs, done in episode:
                    batch_obs.append(np.array(obs, dtype=np.float32))
                    batch_next_obs.append(np.array(next_obs, dtype=np.float32))
                    batch_actions.append(np.array(action, dtype=np.float32))
                    batch_rewards.append([reward])
                    batch_dones.append([done])
                    batch_infos.append({})
                    self.topk_buffer.mark_injected(ep_id)
                    # Once we fill a batch, inject it
                    if len(batch_obs) == batch_size:
                        try:
                            self.model.replay_buffer.add(
                                np.stack(batch_obs),
                                np.stack(batch_next_obs),
                                np.stack(batch_actions),
                                np.array(batch_rewards, dtype=np.float32).squeeze(-1),  # (n_envs,)
                                np.array(batch_dones, dtype=bool).squeeze(-1),          # (n_envs,)

                                infos=batch_infos
                            )
                            # print(f"[TopKCallback] Injected batch of {batch_size} transitions into replay buffer.")
                        except Exception as e:
                            print(f"[TopKCallback] Injection failed: {e}")

                        # Reset batch
                        batch_obs, batch_next_obs, batch_actions = [], [], []
                        batch_rewards, batch_dones, batch_infos = [], [], []

        return True
