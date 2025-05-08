#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：API-Network 
@File    ：StarCraft2EnvWrapper.py
@Author  ：Hao Xiaotian
@Date    ：2022/6/13 16:26 
'''

import copy
import numpy as np
from .official.starcraft2 import StarCraft2Env
from envs.multiagentenv import MultiAgentEnv

class StarCraft2EnvWrapper(MultiAgentEnv):
    def __init__(self, map_name, seed, **kwargs):
        self.env = StarCraft2Env(map_name=map_name, seed=seed, **kwargs)
        self.episode_limit = self.env.episode_limit
    
    def step(self, actions):
        """Returns obss, reward, terminated, truncated, info"""
        rews, terminated, info = self.env.step(actions)
        obss = self.get_obs()
        truncated = False
        return obss, rews, terminated, truncated, info

    def get_obs(self):
        """Returns all agent observations in a list"""
        return self.env.get_obs()

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id"""
        return self.env.get_obs_agent(agent_id)

    def get_obs_size(self):
        """Returns the shape of the observation"""
        return self.env.get_obs_size()

    def get_state(self):
        return self.env.get_state()

    def get_state_size(self):
        """Returns the shape of the state"""
        return self.env.get_state_size()

    def get_avail_actions(self):
        return self.env.get_avail_actions()

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id"""
        return self.env.get_avail_agent_actions(agent_id)

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take"""
        return self.env.get_total_actions()

    def reset(self, seed=None, options=None):
        """Returns initial observations and info"""
        if seed is not None:
            self.env.seed(seed)
        obss, _ = self.env.reset()
        return obss, {}

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def seed(self, seed=None):
        self.env.seed(seed)

    def save_replay(self):
        self.env.save_replay()

    def get_stats(self):
        return self.env.get_stats()

    def get_state_delayed(self, delay_type, delay_value, delay_scope):
        delay_obs, enemy_delay_values, ally_delay_values = self.get_obs_delayed(delay_type, delay_value, delay_scope)
        return np.concatenate(delay_obs, axis=0)

    # Add new functions to support permutation operation
    def get_obs_component(self):
        move_feats_dim = self.env.get_obs_move_feats_size()
        enemy_feats_dim = self.env.get_obs_enemy_feats_size()
        ally_feats_dim = self.env.get_obs_ally_feats_size()
        own_feats_dim = self.env.get_obs_own_feats_size()
        obs_component = [move_feats_dim, enemy_feats_dim, ally_feats_dim, own_feats_dim]
        return obs_component

    def get_state_component(self):
        if self.env.obs_instead_of_state:
            return [self.env.get_obs_size()] * self.env.n_agents

        nf_al = 4 + self.env.shield_bits_ally + self.env.unit_type_bits
        nf_en = 3 + self.env.shield_bits_enemy + self.env.unit_type_bits

        enemy_state = self.env.n_enemies * nf_en
        ally_state = self.env.n_agents * nf_al

        size = [ally_state, enemy_state]

        if self.env.state_last_action:
            size.append(self.env.n_agents * self.env.n_actions)
        if self.env.state_timestep_number:
            size.append(1)
        return size

    def get_env_info(self, delay_aware):
        env_info = {
            "state_shape": self.env.get_state_size(),
            "obs_shape": self.env.get_obs_size(),
            "n_actions": self.env.get_total_actions(),
            "n_agents": self.env.n_agents,
            "episode_limit": self.env.episode_limit,

            "n_normal_actions": self.env.n_actions_no_attack,
            "n_enemies": self.env.n_enemies,
            "n_allies": self.env.n_agents - 1,
            "state_ally_feats_size": self.env.get_ally_num_attributes(),  # 4 + self.shield_bits_ally + self.unit_type_bits,
            "state_enemy_feats_size": self.env.get_enemy_num_attributes(),  # 3 + self.shield_bits_enemy + self.unit_type_bits,
            "obs_move_feats_size": self.env.get_obs_move_feats_size(),
            "obs_own_feats_size": self.env.get_obs_own_feats_size(),
            "obs_ally_feats_size": self.env.get_obs_ally_feats_size(),
            "obs_enemy_feats_size": self.env.get_obs_enemy_feats_size(),
            "unit_type_bits": self.env.unit_type_bits,
            "obs_component": self.get_obs_component(),
            "state_component": self.get_state_component(),
            "map_type": self.env.map_type,
        }
        if delay_aware == False:
            env_info["state_shape"] = env_info["obs_shape"] * env_info["n_agents"]
        print(env_info)
        return env_info

    def _get_medivac_ids(self):
        medivac_ids = []
        for al_id, al_unit in self.env.agents.items():
            if self.env.map_type == "MMM" and al_unit.unit_type == self.env.medivac_id:
                medivac_ids.append(al_id)
        print(medivac_ids)  # [9]
        return medivac_ids

# 延迟设计思路：观测延迟也就是观察敌方状态的延迟+通信延迟也就是观察友方状态的延迟
    def get_obs_delayed(self, delay_type, delay_value, delay_scope):
        # TODO：有一个问题，友军延迟是否应该让环境明确延迟观测的时间戳，而敌军延迟观测则不包含这一信息？
        assert delay_scope <= delay_value

        move_feats = self.env.get_obs_move_feats_size()
        n_enemies, enemy_feats = self.env.get_obs_enemy_feats_size()
        n_allies, ally_feats = self.env.get_obs_ally_feats_size()
        own_feats = self.env.get_obs_own_feats_size()
        
        steps = self.env._episode_steps
        if steps == 0:
            # 记录每个智能体每步的真实观测
            self.origin_agent_obs = {i: [self.get_obs_agent(i)] for i in range(self.env.n_agents)}
            # 记录每个智能体每步观测中延迟后的实际时间步的历史记录，只有uf有用
            self.enemy_delay_values_history = [[] for _ in range(self.env.n_agents)]
            self.ally_delay_values_history = [[] for _ in range(self.env.n_agents)]
        
        # 记录每个智能体每步观测中延迟后的实际时间步
        # 注意这里enemy_delay_values和ally_delay_values存的是每块观测对应的时间步，也就是当前时间步-延迟值，后面enemy_delay和ally_delay表示的是延迟值
        enemy_delay_values = [[] for _ in range(self.env.n_agents)]
        ally_delay_values = [[] for _ in range(self.env.n_agents)]

        obs = []
        for i in range(self.env.n_agents):
            # (move_feats + n_enemies * enemy_feats + n_allies * ally_feats + own_feats, )
            agent_obs = self.env.get_obs_agent(i)
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
                # 部分随机延迟（延迟数值固定，观测中有部分发生延迟，partially fixed） 
                delay_idx = max(0, steps - delay_value)
                enemy_idx = np.random.choice(n_enemies, np.random.randint(0, n_enemies + 1), replace=False)
                ally_idx = np.random.choice(n_allies, np.random.randint(0, n_allies + 1), replace=False)
                
                for j in range(n_enemies):
                    if j in enemy_idx:
                        agent_obs[move_feats + (j * enemy_feats): move_feats + ((j + 1) * enemy_feats)] = self.origin_agent_obs[i][delay_idx][move_feats + (j * enemy_feats): move_feats + ((j + 1) * enemy_feats)]
                        enemy_delay_values[i].append(delay_idx)
                    else:
                        enemy_delay_values[i].append(steps)
                for j in range(n_allies):
                    if j in ally_idx:
                        agent_obs[move_feats + (n_enemies * enemy_feats) + (j * ally_feats): move_feats + (n_enemies * enemy_feats) + ((j + 1) * ally_feats)] = self.origin_agent_obs[i][delay_idx][move_feats + (n_enemies * enemy_feats) + (j * ally_feats): move_feats + (n_enemies * enemy_feats) + ((j + 1) * ally_feats)]
                        ally_delay_values[i].append(delay_idx)
                    else:
                        ally_delay_values[i].append(steps)

            elif delay_type == "uf":
                # 完全随机延迟（友军延迟数值不固定，观测中有部分发生延迟，敌军延迟数值不确定，观测中有部分发生延迟，unfixed）
                for j in range(n_enemies):
                    # 当前延迟最多是上一次延迟+1
                    enemy_delay = np.random.randint(max(0, delay_value - delay_scope), delay_value + delay_scope + 1) if delay_value != 0 else 0
                    enemy_delay = min(enemy_delay, steps - self.enemy_delay_values_history[i][j - n_enemies]) if len(self.enemy_delay_values_history[i]) > 0 else 0
                    delay_idx = max(0, steps - enemy_delay)
                    agent_obs[move_feats + (j * enemy_feats): move_feats + ((j + 1) * enemy_feats)] = self.origin_agent_obs[i][delay_idx][move_feats + (j * enemy_feats): move_feats + ((j + 1) * enemy_feats)]
                    enemy_delay_values[i].append(delay_idx)
                for j in range(n_allies):
                    # 当前延迟最多是上一次延迟+1
                    ally_delay = np.random.randint(max(0, delay_value - delay_scope), delay_value + delay_scope + 1) if delay_value != 0 else 0
                    ally_delay = min(ally_delay, steps - self.ally_delay_values_history[i][j - n_allies]) if len(self.ally_delay_values_history[i]) > 0 else 0
                    delay_idx = max(0, steps - ally_delay)
                    agent_obs[move_feats + (n_enemies * enemy_feats) + (j * ally_feats): move_feats + (n_enemies * enemy_feats) + ((j + 1) * ally_feats)] = self.origin_agent_obs[i][delay_idx][move_feats + (n_enemies * enemy_feats) + (j * ally_feats): move_feats + (n_enemies * enemy_feats) + ((j + 1) * ally_feats)]
                    ally_delay_values[i].append(delay_idx)
            else:
                assert False, "This type of delay(" + delay_type + ") is currently not supported"

            # 维护列表
            self.enemy_delay_values_history[i] = self.enemy_delay_values_history[i] + enemy_delay_values[i]
            self.ally_delay_values_history[i] = self.ally_delay_values_history[i] + ally_delay_values[i]
            obs.append(agent_obs.copy())

        return obs, enemy_delay_values, ally_delay_values
