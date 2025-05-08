# code heavily adapted from https://github.com/AnujMahajanOxf/MAVEN
import copy
import time
import numpy as np
import torch as th
from torch.optim import Adam

from components.episode_buffer import EpisodeBatch
from utils.rl_utils import RunningMeanStd
from modules.critics import REGISTRY as critic_resigtry
from components.epsilon_schedules import FlatThenDecayThenFlatSchedule


class PDPPOLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.logger = logger
        self.device = th.device('cuda' if args.use_cuda else 'cpu')

        # actor
        self.mac = mac
        self.old_mac = copy.deepcopy(mac)
        self.agent_params = list(mac.parameters())
        self.agent_optimiser = Adam(params=self.agent_params, lr=args.lr)

        # critic
        self.critic = critic_resigtry[args.critic_type](scheme, args)
        self.target_critic = copy.deepcopy(self.critic)
        self.critic_params = list(self.critic.parameters())
        self.critic_optimiser = Adam(params=self.critic_params, lr=args.lr)

        self.last_target_update_step = 0
        self.critic_training_steps = 0
        self.log_stats_t = -self.args.learner_log_interval - 1
        self.train_t = 0
        self.avg_time = 0
        # 预测器
        if self.args.name in ["pd_mappo"]:
            input_shape, output_shape = self._get_predict_shape(scheme, args)
            if self.args.env in ["sc2", "sc2_v2"]:
                if self.args.predictor_model == "mlp":
                    from components.delay_predictor_mlp4sc2 import Predictor_Controller
                elif self.args.predictor_model == "gru":
                    from components.delay_predictor_gru4sc2 import Predictor_Controller
                elif self.args.predictor_model == "tf":
                    from components.delay_predictor_tf4sc2 import Predictor_Controller
            else:
                if self.args.predictor_model == "mlp":
                    from components.delay_predictor_mlp4mpe import Predictor_Controller
                elif self.args.predictor_model == "gru":
                    from components.delay_predictor_gru4mpe import Predictor_Controller
                elif self.args.predictor_model == "tf":
                    from components.delay_predictor_tf4mpe import Predictor_Controller
            self.predictor = Predictor_Controller(input_shape, output_shape, args)
            self.mac.set_predictor(self.predictor)
            self.old_mac.set_predictor(self.predictor)
        # 价值标准化
        if self.args.standardise_returns:
            self.ret_ms = RunningMeanStd(shape=(self.n_agents,), device=self.device)
        # 奖励标准化
        if self.args.standardise_rewards:
            rew_shape = (1,) if self.args.common_reward else (self.n_agents,)
            self.rew_ms = RunningMeanStd(shape=rew_shape, device=self.device)
        self.ppo_lr = args.lr
        self.entropy_coef = args.entropy_coef
        # curriculum learning
        self.cheating_schedule = FlatThenDecayThenFlatSchedule(args.cheating_start_value, args.cheating_end_value, args.cheating_start_time, args.cheating_end_time, decay="linear")
        self.cheating_prob = self.cheating_schedule.eval(0)
        # teacher student
        self.teacher_schedule = FlatThenDecayThenFlatSchedule(args.teacher_student_start_value, args.teacher_student_end_value, args.teacher_student_start_time, args.teacher_student_end_time, decay="linear")
        self.alpha = self.teacher_schedule.eval(0)
        # teacher model
        if self.args.teacher_model_dir != "":
            groups = {"agents": args.n_agents}
            self.teacher_mac = mac_REGISTRY[args.teacher_mac](scheme, groups, args)
            self.teacher_mac.load_models(self.args.teacher_model_dir)
            if self.args.use_cuda:
                self.teacher_mac.cuda()
            self.teacher_mac.set_evaluation_mode()
            self.teacher_critic = copy.deepcopy(self.critic)
            self.teacher_critic.load_state_dict(th.load("{}/critic.th".format(self.args.teacher_model_dir), map_location=lambda storage, loc: storage))
            if self.args.use_cuda:
                self.teacher_critic.cuda()
            self.teacher_critic.eval()

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        start_time = time.time()
        if self.args.use_cuda and str(self.mac.get_device()) == "cpu":
            self.mac.cuda()

        self.cheating_prob = self.cheating_schedule.eval(t_env)
        self.mac.cheating_prob = self.cheating_prob
        self.old_mac.cheating_prob = self.cheating_prob
        self.alpha = self.teacher_schedule.eval(t_env)

        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        actions = actions[:, :-1]
        if self.args.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)

        # 训练预测器
        self.logger.log_stat("eval/pd_reg_loss_avg", self.predictor.eval_reg_loss_avg.item(), t_env)
        self.logger.log_stat("eval/pd_cla_loss_avg", self.predictor.eval_cla_loss_avg.item(), t_env)
        self.logger.log_stat("eval/pd_obs_loss_avg", self.predictor.eval_obs_loss_avg.item(), t_env)
        self.predictor.unfreeze()
        self.predictor.set_teacher_forcing(t_env)
        predict_obs, pd_reg_loss_avg, pd_cla_loss_avg, pd_obs_loss_avg = self.predictor.do_predict(batch)
        self.logger.log_stat("train/pd_reg_loss_avg", pd_reg_loss_avg.item(), t_env)
        self.logger.log_stat("train/pd_cla_loss_avg", pd_cla_loss_avg.item(), t_env)
        self.logger.log_stat("train/pd_obs_loss_avg", pd_obs_loss_avg.item(), t_env)
        self.predictor.freeze() # 预测器训练完毕
        # 作弊机制：使用真实观测代替预测结果
        cheating = np.random.random() < self.cheating_prob
        if cheating:
            predict_obs = batch["real_obs"]
            self.mac.cheating = True
            self.old_mac.cheating = True
        
        mask = mask.repeat(1, 1, self.n_agents)
        critic_mask = mask.clone()

        self.old_mac.set_evaluation_mode()
        old_mac_out = []
        self.old_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length - 1):
            agent_outs = self.old_mac.forward(batch, t, predict_obs)
            old_mac_out.append(agent_outs)
        old_mac_out = th.stack(old_mac_out, dim=1)  # Concat over time
        old_pi = old_mac_out
        old_pi[mask == 0] = 1.0
        old_pi_taken = th.gather(old_pi, dim=3, index=actions).squeeze(3)
        old_log_pi_taken = th.log(old_pi_taken + 1e-10)
        avg_entropy = 0

        for k in range(self.args.epochs):
            self.mac.set_train_mode()
            mac_out = []
            self.mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length - 1):
                agent_outs = self.mac.forward(batch, t, predict_obs)
                mac_out.append(agent_outs)
            mac_out = th.stack(mac_out, dim=1)  # Concat over time

            pi = mac_out
            advantages, critic_train_stats = self.train_critic_sequential(
                self.critic, self.target_critic, batch, rewards, critic_mask
            )
            advantages = advantages.detach()
            
            # 优势函数标准化
            if self.args.standardise_advantages:
                # 创建一个掩码，标记有效的优势值，只计算有效值的均值和标准差
                valid_mask = mask.bool()
                valid_advantages = advantages[valid_mask]
                mean_advantages = valid_advantages.mean()
                std_advantages = valid_advantages.std()
                advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

            # Calculate policy grad with mask
            pi[mask == 0] = 1.0
            pi_taken = th.gather(pi, dim=3, index=actions).squeeze(3)
            log_pi_taken = th.log(pi_taken + 1e-10)

            ratios = th.exp(log_pi_taken - old_log_pi_taken.detach())
            surr1 = ratios * advantages
            surr2 = (
                th.clamp(ratios, 1 - self.args.eps_clip, 1 + self.args.eps_clip)
                * advantages
            )

            entropy = -th.sum(pi * th.log(pi + 1e-10), dim=-1)
            avg_entropy += entropy
            pg_loss = -((th.min(surr1, surr2) + self.entropy_coef * entropy) * mask).sum() / mask.sum()

            # Optimise agents
            self.agent_optimiser.zero_grad()
            pg_loss.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(
                self.agent_params, self.args.grad_norm_clip
            )
            self.agent_optimiser.step()

        avg_entropy = avg_entropy.mean() / self.args.epochs
        self.old_mac.load_state(self.mac)

        self.critic_training_steps += 1
        if (
            self.args.target_update_interval_or_tau > 1
            and (self.critic_training_steps - self.last_target_update_step)
            / self.args.target_update_interval_or_tau
            >= 1.0
        ):
            self._update_targets_hard()
            self.last_target_update_step = self.critic_training_steps
        elif self.args.target_update_interval_or_tau <= 1.0:
            self._update_targets_soft(self.args.target_update_interval_or_tau)
        self.mac.set_evaluation_mode()  # RL训练完毕

        self.train_t += 1
        self.avg_time += (time.time() - start_time - self.avg_time) / self.train_t
        print("Avg cost {} seconds".format(self.avg_time))
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            ts_logged = len(critic_train_stats["critic_loss"])
            for key in [
                "critic_loss",
                "critic_grad_norm",
                "td_error_abs",
                "q_taken_mean",
                "target_mean",
            ]:
                self.logger.log_stat(
                    "train/" + key, sum(critic_train_stats[key]) / ts_logged, t_env
                )

            self.logger.log_stat(
                "train/advantage_mean",
                (advantages * mask).sum().item() / mask.sum().item(),
                t_env,
            )
            self.logger.log_stat("train/pg_loss", pg_loss.item(), t_env)
            self.logger.log_stat("train/agent_grad_norm", grad_norm.item(), t_env)
            self.logger.log_stat(
                "train/pi_max",
                (pi.max(dim=-1)[0] * mask).sum().item() / mask.sum().item(),
                t_env,
            )
            self.logger.log_stat("train/cheating_prob", self.cheating_prob, t_env)
            self.logger.log_stat("train/pd_obs_loss_avg", pd_obs_loss_avg, t_env)
            self.logger.log_stat("train/ppo_lr", self.ppo_lr, t_env)
            self.logger.log_stat("train/entropy", avg_entropy, t_env)
            self.logger.log_stat("train/entropy_coef", self.entropy_coef, t_env)
            self.logger.log_stat("train/cuda_memory", th.cuda.max_memory_allocated() / 1024**2, t_env)
            self.log_stats_t = t_env

    def train_critic_sequential(self, critic, target_critic, batch, rewards, mask):
        # Optimise critic
        with th.no_grad():
            target_vals = target_critic(batch)
            target_vals = target_vals.squeeze(3)

        if self.args.standardise_returns:
            target_vals = target_vals * th.sqrt(self.ret_ms.var) + self.ret_ms.mean

        target_returns = self.nstep_returns(
            rewards, mask, target_vals, self.args.q_nstep
        )
        if self.args.standardise_returns:
            self.ret_ms.update(target_returns)
            target_returns = (target_returns - self.ret_ms.mean) / th.sqrt(
                self.ret_ms.var
            )

        running_log = {
            "critic_loss": [],
            "critic_grad_norm": [],
            "td_error_abs": [],
            "target_mean": [],
            "q_taken_mean": [],
        }

        v = critic(batch)[:, :-1].squeeze(3)
        td_error = target_returns.detach() - v
        masked_td_error = td_error * mask
        loss = (masked_td_error**2).sum() / mask.sum()

        self.critic_optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(
            self.critic_params, self.args.grad_norm_clip
        )
        self.critic_optimiser.step()

        running_log["critic_loss"].append(loss.item())
        running_log["critic_grad_norm"].append(grad_norm.item())
        mask_elems = mask.sum().item()
        running_log["td_error_abs"].append(
            (masked_td_error.abs().sum().item() / mask_elems)
        )
        running_log["q_taken_mean"].append((v * mask).sum().item() / mask_elems)
        running_log["target_mean"].append(
            (target_returns * mask).sum().item() / mask_elems
        )

        return masked_td_error, running_log

    def nstep_returns(self, rewards, mask, values, nsteps):
        nstep_values = th.zeros_like(values[:, :-1])
        for t_start in range(rewards.size(1)):
            nstep_return_t = th.zeros_like(values[:, 0])
            for step in range(nsteps + 1):
                t = t_start + step
                if t >= rewards.size(1):
                    break
                elif step == nsteps:
                    nstep_return_t += (
                        self.args.gamma ** (step) * values[:, t] * mask[:, t]
                    )
                elif t == rewards.size(1) - 1 and self.args.add_value_last_step:
                    nstep_return_t += (
                        self.args.gamma ** (step) * rewards[:, t] * mask[:, t]
                    )
                    nstep_return_t += self.args.gamma ** (step + 1) * values[:, t + 1]
                else:
                    nstep_return_t += (
                        self.args.gamma ** (step) * rewards[:, t] * mask[:, t]
                    )
            nstep_values[:, t_start, :] = nstep_return_t
        return nstep_values

    def _update_targets(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    def _update_targets_hard(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    def _update_targets_soft(self, tau):
        for target_param, param in zip(
            self.target_critic.parameters(), self.critic.parameters()
        ):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def cuda(self):
        self.predictor.cuda()
        self.old_mac.cuda()
        self.mac.cuda()
        self.critic.cuda()
        self.target_critic.cuda()

    def save_models(self, path):
        self.predictor.save_models(path)
        self.mac.save_models(path)
        th.save(self.critic.state_dict(), "{}/critic.th".format(path))
        th.save(self.agent_optimiser.state_dict(), "{}/agent_opt.th".format(path))
        th.save(self.critic_optimiser.state_dict(), "{}/critic_opt.th".format(path))

    def load_models(self, path):
        self.predictor.load_models(path)
        self.mac.load_models(path)
        self.critic.load_state_dict(
            th.load(
                "{}/critic.th".format(path), map_location=lambda storage, loc: storage
            )
        )
        # Not quite right but I don't want to save target networks
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.agent_optimiser.load_state_dict(
            th.load(
                "{}/agent_opt.th".format(path),
                map_location=lambda storage, loc: storage,
            )
        )
        self.critic_optimiser.load_state_dict(
            th.load(
                "{}/critic_opt.th".format(path),
                map_location=lambda storage, loc: storage,
            )
        )

    def _get_predict_shape(self, scheme, args):
        # 输入是延迟观测+上一步动作+延迟值
        if self.args.one_hot_delay: 
            if self.args.predictor_model == "mlp":
                input_shape = (scheme["obs"]["vshape"] + scheme["actions_onehot"]["vshape"][0] + (args.n_enemies + args.n_allies) * (self.args.n_expand_action + 1)) * self.args.n_expand_action
            else:
                input_shape = scheme["obs"]["vshape"] + scheme["actions_onehot"]["vshape"][0] + (args.n_enemies + args.n_allies) * (self.args.n_expand_action + 1)
        else:
            if self.args.predictor_model == "mlp":
                input_shape = (scheme["obs"]["vshape"] + scheme["actions_onehot"]["vshape"][0] + (args.n_enemies + args.n_allies)) * self.args.n_expand_action
            else:
                input_shape = scheme["obs"]["vshape"] + scheme["actions_onehot"]["vshape"][0] + (args.n_enemies + args.n_allies)
        
        if self.args.env in ["sc2", "sc2_v2"]:
            # 敌方距离、相对坐标血量、护盾+友方距离、相对坐标血量、护盾
            output_shape_regression = (args.n_enemies * (args.obs_enemy_feats_size - args.unit_type_bits - 1)) + (args.n_allies * (args.obs_ally_feats_size - args.unit_type_bits - 1))
            # 目标是否可观测/是否可攻击
            output_shape_classification = args.n_enemies + args.n_allies
            assert output_shape_regression + output_shape_classification + args.obs_move_feats_size + (args.n_enemies + args.n_allies) * args.unit_type_bits + args.obs_own_feats_size == scheme["obs"]["vshape"], "预测器构造输入出错！"
        elif self.args.env == "gymma":
            if self.args.env_args["key"] == "pz-mpe-simple-reference-v3":
                output_shape_regression = args.n_enemies * args.obs_enemy_feats_size
                output_shape_classification = args.n_allies * args.obs_ally_feats_size
            elif self.args.env_args["key"] == "pz-mpe-simple-spread-v3" or self.args.env_args["key"] == "pz-mpe-simple-tag-v3":
                # 全部目标和友方观测
                output_shape_regression = (args.n_enemies * args.obs_enemy_feats_size) + (args.n_allies * args.obs_ally_feats_size)
                # 不需要分类预测
                output_shape_classification = 0
            assert output_shape_regression + output_shape_classification + args.obs_move_feats_size + args.obs_own_feats_size == scheme["obs"]["vshape"], "预测器构造输入出错！"
            
        print("Predict Shape:")
        print(input_shape, (output_shape_regression, output_shape_classification, scheme["obs"]["vshape"]))
        return input_shape, (output_shape_regression, output_shape_classification, scheme["obs"]["vshape"])