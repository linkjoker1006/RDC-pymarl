from collections.abc import Iterable
import warnings

import gymnasium as gym
from gymnasium.spaces import flatdim
from gymnasium.wrappers import TimeLimit
import numpy as np

from envs.multiagentenv import MultiAgentEnv
from envs.wrappers import FlattenObservation
import envs.mpe.pretrained as pretrained  # noqa

try:
    from envs.pz_wrapper import PettingZooWrapper  # noqa
except ImportError:
    warnings.warn(
        "PettingZoo is not installed, so these environments will not be available! To install, run `pip install pettingzoo`"
    )

try:
    from .vmas_wrapper import VMASWrapper  # noqa
except ImportError:
    warnings.warn(
        "VMAS is not installed, so these environments will not be available! To install, run `pip install 'vmas[gymnasium]'`"
    )


class GymmaEnvWrapper(MultiAgentEnv):
    def __init__(
        self,
        key,
        time_limit,
        pretrained_wrapper,
        seed,
        common_reward,
        reward_scalarisation,
        **kwargs,
    ):
        self.key = key
        self._env = gym.make(f"{key}", **kwargs)
        self._env = TimeLimit(self._env, max_episode_steps=time_limit)
        self._env = FlattenObservation(self._env)
        
        if pretrained_wrapper:
            self._env = getattr(pretrained, pretrained_wrapper)(self._env)

        self.n_agents = self._env.unwrapped.n_agents
        self.episode_limit = time_limit
        self._obs = None
        self.delay_obs = None
        self._info = None

        self.longest_action_space = max(self._env.action_space, key=lambda x: x.n)
        self.longest_observation_space = max(
            self._env.observation_space, key=lambda x: x.shape
        )

        self._seed = seed
        try:
            self._env.unwrapped.seed(self._seed)
        except:
            self._env.reset(seed=self._seed)

        self.common_reward = common_reward
        if self.common_reward:
            if reward_scalarisation == "sum":
                self.reward_agg_fn = lambda rewards: sum(rewards)
            elif reward_scalarisation == "mean":
                self.reward_agg_fn = lambda rewards: sum(rewards) / len(rewards)
            else:
                raise ValueError(
                    f"Invalid reward_scalarisation: {reward_scalarisation} (only support 'sum' or 'mean')"
                )

    def _pad_observation(self, obs):
        return [
            np.pad(
                o,
                (0, self.longest_observation_space.shape[0] - len(o)),
                "constant",
                constant_values=0,
            )
            for o in obs
        ]

    def step(self, actions):
        """Returns obss, reward, terminated, truncated, info"""
        actions = [int(a) for a in actions]
        obs, reward, done, truncated, self._info = self._env.step(actions)
        self._obs = self._pad_observation(obs)
        self._obs = self.reshape_obs()
        if self.common_reward and isinstance(reward, Iterable):
            reward = float(self.reward_agg_fn(reward))
        elif not self.common_reward and not isinstance(reward, Iterable):
            warnings.warn(
                "common_reward is False but received scalar reward from the environment, returning reward as is"
            )

        if isinstance(done, Iterable):
            done = all(done)
        return self._obs.copy(), reward, done, truncated, self._info

    def get_obs(self):
        """Returns all agent observations in a list"""
        return self._obs.copy()

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id"""
        return self._obs[agent_id].copy()

    def get_obs_size(self):
        """Returns the shape of the observation"""
        return flatdim(self.longest_observation_space)

    def get_state(self):
        return np.concatenate(self._obs, axis=0).astype(np.float32)

    def get_state_size(self):
        """Returns the shape of the state"""
        if hasattr(self._env.unwrapped, "state_size"):
            return self._env.unwrapped.state_size
        return self.n_agents * flatdim(self.longest_observation_space)

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id"""
        valid = flatdim(self._env.action_space[agent_id]) * [1]
        invalid = [0] * (self.longest_action_space.n - len(valid))
        return valid + invalid

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take"""
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        return flatdim(self.longest_action_space)

    def reset(self, seed=None, options=None):
        """Returns initial observations and info"""
        obs, info = self._env.reset(seed=seed, options=options)
        self._obs = self._pad_observation(obs)
        self._obs = self.reshape_obs()
        return self._obs.copy(), info

    def render(self):
        self._env.render()

    def close(self):
        self._env.close()

    def seed(self, seed=None):
        return self._env.unwrapped.seed(seed)

    def save_replay(self):
        pass

    def get_stats(self):
        return {}

    def get_env_info(self, delay_aware):
        # obs_own_feats_size并不代表自身属性的观测，只是为了和sc2保持一致
        env_info = {
            "state_shape": self.get_state_size(),
            "obs_shape": self.get_obs_size(),
            "n_actions": self.get_total_actions(),
            "n_agents": self.n_agents,
            "episode_limit": self.episode_limit,
            "unit_type_bits": 0}
        if self.key == "pz-mpe-simple-reference-v3":
            env_info["n_enemies"] = 3
            env_info["n_allies"] = 1
            env_info["obs_move_feats_size"] = 2
            env_info["obs_enemy_feats_size"] = 2
            env_info["obs_ally_feats_size"] = 10
            env_info["obs_own_feats_size"] = 3
            env_info["obs_component"] = (2, (3, 2), (1, 10), 3)
            env_info["map_type"] = "reference"
        elif self.key == "pz-mpe-simple-speaker-listener-v4":
            env_info["n_enemies"] = 3
            env_info["n_allies"] = 1
            env_info["obs_move_feats_size"] = 2
            env_info["obs_enemy_feats_size"] = 2
            env_info["obs_ally_feats_size"] = 3
            env_info["obs_own_feats_size"] = 0
            env_info["obs_component"] = (2, (3, 2), (1, 3), 0)
            env_info["map_type"] = "speaker-listener"
        elif self.key == "pz-mpe-simple-spread-v3":
            env_info["n_enemies"] = 3
            env_info["n_allies"] = 2
            env_info["obs_move_feats_size"] = 4
            env_info["obs_enemy_feats_size"] = 2
            env_info["obs_ally_feats_size"] = 4
            env_info["obs_own_feats_size"] = 0
            env_info["obs_component"] = (4, (3, 2), (2, 4), 0)
            env_info["map_type"] = "spread"
        elif self.key == "pz-mpe-simple-tag-v3":
            env_info["n_enemies"] = 4  # 两个障碍物 一个猎物 额外将猎物速度也看做独立发生延迟
            env_info["n_allies"] = 2
            env_info["obs_move_feats_size"] = 4
            env_info["obs_enemy_feats_size"] = 2
            env_info["obs_ally_feats_size"] = 2
            env_info["obs_own_feats_size"] = 0
            env_info["obs_component"] = (4, (4, 2), (2, 2), 0)
            env_info["map_type"] = "tag"
        else:
            assert False, f"Unsupported key: {self.key}"      

        print(env_info)
        return env_info

    def reshape_obs(self):
        if self.key == "pz-mpe-simple-reference-v3":
            for i in range(len(self._obs)):
                color_info = self._obs[i][-13:-10].copy()
                ally_feats = self._obs[i][-10:].copy()
                self._obs[i][-13:-3] = ally_feats
                self._obs[i][-3:] = color_info
        elif self.key == "pz-mpe-simple-spread-v3":
            for i in range(len(self._obs)):
                first_ally_feats = self._obs[i][-4:-2].copy()
                second_ally_feats = self._obs[i][-6:-4].copy()
                self._obs[i][-6:-4] = first_ally_feats
                self._obs[i][-4:-2] = second_ally_feats
        elif self.key == "pz-mpe-simple-tag-v3":
            for i in range(len(self._obs)):
                target_feats = self._obs[i][-4:].copy()
                allies_feats = self._obs[i][-8:-4].copy()
                self._obs[i][-8:-4] = target_feats
                self._obs[i][-4:] = allies_feats
        else:
            assert False, f"Unsupported key: {self.key}"
        return self._obs

    def get_obs_delayed(self, delay_type, delay_value, delay_scope):
        assert delay_scope <= delay_value
        if self.key == "pz-mpe-simple-reference-v3":
            move_feats = 2
            n_enemies = 3
            enemy_feats = 2
            n_allies = 1
            ally_feats = 10  # 友方通信信息
            own_feats = 3  # 自身存储友方目标信息
        elif self.key == "pz-mpe-simple-speaker-listener-v4":
            # speaker的观测不存在延迟，只考虑listener的观测延迟
            move_feats = 2
            n_enemies = 3
            enemy_feats = 2
            n_allies = 1
            ally_feats = 3  # 友方通信信息
            own_feats = 0
        elif self.key == "pz-mpe-simple-spread-v3":
            move_feats = 4
            n_enemies = 3
            enemy_feats = 2
            n_allies = 2
            ally_feats = 4  # 友方位置+通信信息
            own_feats = 0  # 通信信息
        elif self.key == "pz-mpe-simple-tag-v3":
            move_feats = 4
            n_enemies = 4
            enemy_feats = 2
            n_allies = 2
            ally_feats = 2
            own_feats = 0
        else:
            assert False, f"Unsupported key: {self.key}"
        steps = self._env.unwrapped._env.unwrapped.steps
        if steps == 0:  
            # 记录每个智能体每步的真实观测
            self.origin_agent_obs = {i: [self.get_obs_agent(i)] for i in range(self.n_agents)}
            # 记录每个智能体每步观测中延迟后的实际时间步的历史记录，只有uf有用
            self.enemy_delay_values_history = [[] for _ in range(self.n_agents)]
            self.ally_delay_values_history = [[] for _ in range(self.n_agents)]
        # 记录每个智能体每步观测中延迟后的实际时间步
        # 注意这里enemy_delay_values和ally_delay_values存的是每块观测对应的时间步，也就是当前时间步-延迟值，后面enemy_delay和ally_delay表示的是延迟值
        enemy_delay_values = [[] for _ in range(self.n_agents)]
        ally_delay_values = [[] for _ in range(self.n_agents)]

        obs = []
        for i in range(self.n_agents):
            # (move_feats + n_enemies * enemy_feats + n_allies * ally_feats + own_feats, )
            agent_obs = self.get_obs_agent(i)
            if steps > 0:
                self.origin_agent_obs[i].append(agent_obs.copy())
            
            if delay_type == "none":
                # 不进行延迟处理，直接输出无延迟观测
                delay_idx = steps
                enemy_delay_values[i] = [delay_idx] * n_enemies
                ally_delay_values[i] = [delay_idx] * n_allies

            elif delay_type == "f":
                # 固定延迟，全部敌军、友军观测延迟相同，其它观测不延迟，fixed
                delay_idx = max(0, steps - delay_value)
                agent_obs[move_feats: move_feats + (n_enemies * enemy_feats) + (n_allies * ally_feats)] = \
                    self.origin_agent_obs[i][delay_idx][move_feats: move_feats + (n_enemies * enemy_feats) + (n_allies * ally_feats)]
                enemy_delay_values[i] = [delay_idx] * n_enemies
                ally_delay_values[i] = [delay_idx] * n_allies

            elif delay_type == "pf":
                # TODO: 部分随机延迟（延迟数值固定，观测中有部分发生延迟，partially fixed） 
                enemy_idx = np.random.choice(n_enemies, np.random.randint(0, n_enemies + 1), replace=False)
                ally_idx = np.random.choice(n_allies, np.random.randint(0, n_allies + 1), replace=False)

                for j in range(n_enemies):
                    if j in enemy_idx:
                        # 当前延迟最多是上一次延迟+1
                        enemy_delay = min(delay_value, steps - self.enemy_delay_values_history[i][j - n_enemies]) if len(self.enemy_delay_values_history[i]) > 0 else 0
                        enemy_delay = max(0, steps - enemy_delay)
                        agent_obs[move_feats + (j * enemy_feats): move_feats + ((j + 1) * enemy_feats)] = self.origin_agent_obs[i][enemy_delay][move_feats + (j * enemy_feats): move_feats + ((j + 1) * enemy_feats)]
                        enemy_delay_values[i].append(enemy_delay)
                    else:
                        enemy_delay_values[i].append(steps)
                for j in range(n_allies):
                    if j in ally_idx:
                        # 当前延迟最多是上一次延迟+1
                        ally_delay = min(delay_value, steps - self.ally_delay_values_history[i][j - n_allies]) if len(self.ally_delay_values_history[i]) > 0 else 0
                        ally_delay = max(0, steps - ally_delay)
                        agent_obs[move_feats + (n_enemies * enemy_feats) + (j * ally_feats): move_feats + (n_enemies * enemy_feats) + ((j + 1) * ally_feats)] = self.origin_agent_obs[i][ally_delay][move_feats + (n_enemies * enemy_feats) + (j * ally_feats): move_feats + (n_enemies * enemy_feats) + ((j + 1) * ally_feats)]
                        ally_delay_values[i].append(ally_delay)
                    else:
                        ally_delay_values[i].append(steps)
            
            elif delay_type == "uf":
                # 完全随机延迟（友军延迟数值不固定，观测中有部分发生延迟，敌军延迟数值不确定，全部观测发生延迟，unfixed）
                for j in range(n_enemies):
                    # 当前延迟最多是上一次延迟+1
                    enemy_delay = np.random.randint(max(0, delay_value - delay_scope), delay_value + delay_scope + 1)
                    enemy_delay = min(enemy_delay, steps - self.enemy_delay_values_history[i][j - n_enemies]) if len(self.enemy_delay_values_history[i]) > 0 else 0
                    enemy_delay = max(0, steps - enemy_delay)
                    agent_obs[move_feats + (j * enemy_feats): move_feats + ((j + 1) * enemy_feats)] = self.origin_agent_obs[i][enemy_delay][move_feats + (j * enemy_feats): move_feats + ((j + 1) * enemy_feats)]
                    enemy_delay_values[i].append(enemy_delay)
                for j in range(n_allies):
                    # 当前延迟最多是上一次延迟+1
                    ally_delay = np.random.randint(max(0, delay_value - delay_scope), delay_value + delay_scope + 1)
                    ally_delay = min(ally_delay, steps - self.ally_delay_values_history[i][j - n_allies]) if len(self.ally_delay_values_history[i]) > 0 else 0
                    ally_delay = max(0, steps - ally_delay)
                    agent_obs[move_feats + (n_enemies * enemy_feats) + (j * ally_feats): move_feats + (n_enemies * enemy_feats) + ((j + 1) * ally_feats)] = self.origin_agent_obs[i][ally_delay][move_feats + (n_enemies * enemy_feats) + (j * ally_feats): move_feats + (n_enemies * enemy_feats) + ((j + 1) * ally_feats)]
                    ally_delay_values[i].append(ally_delay)
            else:
                assert False, f"Unsupported delay_type: {delay_type}"
            
            # 维护列表
            self.enemy_delay_values_history[i] = self.enemy_delay_values_history[i] + enemy_delay_values[i]
            self.ally_delay_values_history[i] = self.ally_delay_values_history[i] + ally_delay_values[i]
            obs.append(agent_obs.copy())
        self.delay_obs = obs

        return obs, enemy_delay_values, ally_delay_values

    def get_state_delayed(self, delay_type, delay_value, delay_scope):
        return np.concatenate(self.delay_obs, axis=0).astype(np.float32)
