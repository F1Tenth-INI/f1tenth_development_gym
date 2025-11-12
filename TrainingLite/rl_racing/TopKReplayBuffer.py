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
    def __init__(
        self,
        topk_buffer: TopKTrajectoryBuffer,
        inject_every: int = 10000,
        injection_chunk: int = 256,   # decouple from n_envs
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.topk_buffer = topk_buffer
        self.inject_every = inject_every
        self.injection_chunk = injection_chunk

        self.episode_buffers = []
        self.total_rewards = []
        self.episode_counter = 0
        self.last_injection_step = 0
        self.last_obs = None  # shape: (n_envs, obs_dim)

    def _on_training_start(self):
        n_envs = self.training_env.num_envs
        self.episode_buffers = [[] for _ in range(n_envs)]
        self.total_rewards = [0.0 for _ in range(n_envs)]
        self.last_obs = None  # will be set on the first step when we see new_obs

    def _on_step(self) -> bool:
        if self.locals is None or 'infos' not in self.locals:
            return True

        infos = self.locals['infos']
        dones = self.locals['dones']
        observations = self.locals['new_obs']     # this is obs_{t+1}
        actions = self.locals['actions']          # a_t
        rewards = self.locals['rewards']          # r_{t+1}

        # Initialize last_obs on the very first step we see
        if self.last_obs is None:
            # We don't have obs_t for the first transition yet; just set baseline
            self.last_obs = observations.copy()
            return True

        # Build correct transitions (obs_t, a_t, r_{t+1}, obs_{t+1}, done_{t+1})
        for i in range(len(dones)):
            # If episode ended by time-limit, SB3 puts the true terminal obs here
            next_obs_i = infos[i].get("terminal_observation", observations[i])

            transition = (
                self.last_obs[i].copy(),        # obs_t
                actions[i].copy(),              # a_t
                float(rewards[i]),              # r_{t+1}
                next_obs_i.copy(),              # obs_{t+1}
                bool(dones[i])                  # done_{t+1}
            )
            self.episode_buffers[i].append(transition)
            self.total_rewards[i] += float(rewards[i])

            if dones[i]:
                # Episode ended (crash/goal/time-limit/etc.)
                self.episode_counter += 1
                self.topk_buffer.add_episode(self.episode_buffers[i], self.total_rewards[i])
                self.episode_buffers[i] = []
                self.total_rewards[i] = 0.0

        # Advance last_obs to current obs_{t+1} for next step
        self.last_obs = observations.copy()

        # Periodic injection of top-K transitions
        current_step = self.num_timesteps
        if current_step - self.last_injection_step >= self.inject_every:
            self.last_injection_step = current_step

            n_envs = self.model.n_envs  # must add in exact multiples of this
            batch_obs, batch_next_obs, batch_actions = [], [], []
            batch_rewards, batch_dones, batch_infos = [], [], []

            for ep_id, episode in self.topk_buffer.get_uninjected_episodes():
                for obs, action, reward, next_obs, done in episode:
                    batch_obs.append(np.asarray(obs, dtype=np.float32))
                    batch_next_obs.append(np.asarray(next_obs, dtype=np.float32))
                    batch_actions.append(np.asarray(action, dtype=np.float32))
                    batch_rewards.append(float(reward))
                    batch_dones.append(bool(done))
                    batch_infos.append({})

                    # When we have n_envs transitions, push one chunk
                    if len(batch_obs) == n_envs:
                        try:
                            self.model.replay_buffer.add(
                                np.stack(batch_obs),                   # (n_envs, obs_dim)
                                np.stack(batch_next_obs),              # (n_envs, obs_dim)
                                np.stack(batch_actions),               # (n_envs, act_dim)
                                np.asarray(batch_rewards, np.float32), # (n_envs,)
                                np.asarray(batch_dones,   bool),       # (n_envs,)
                                infos=batch_infos                      # list of length n_envs
                            )
                        except Exception as e:
                            print(f"[TopKCallback] Injection failed: {e}")
                        # reset the small chunk
                        batch_obs, batch_next_obs, batch_actions = [], [], []
                        batch_rewards, batch_dones, batch_infos = [], [], []

                # mark after scheduling this episode (whether or not fully injected yet)
                self.topk_buffer.mark_injected(ep_id)

            # If anything remains (< n_envs), either discard or pad; here we discard to be strict:
            if batch_obs:
                print(f"[TopKCallback] Skipping leftover {len(batch_obs)} (< n_envs) transitions to satisfy SB3 add() shape.")


        return True
