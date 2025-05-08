import copy
import time
import numpy as np
import torch as th
from torch.optim import RMSprop, Adam

from components.episode_buffer import EpisodeBatch
from components.epsilon_schedules import FlatThenDecayThenFlatSchedule
from modules.mixers.nmix import Mixer
from modules.mixers.qatten import QattenMixer
from modules.mixers.vdn import VDNMixer
from utils.rl_utils import build_td_lambda_targets, build_q_lambda_targets
from utils.th_utils import get_parameters_num
from utils.rl_utils import RunningMeanStd
from controllers import REGISTRY as mac_REGISTRY


class PDNQLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.last_target_update_episode = 0
        self.device = th.device('cuda' if args.use_cuda else 'cpu')
        self.params = list(mac.agent.parameters())
        # 混合网络
        if args.mixer == "qatten":
            self.mixer = QattenMixer(args)
        elif args.mixer == "vdn":
            self.mixer = VDNMixer()
        elif args.mixer == "qmix":  # 31.521K
            self.mixer = Mixer(args)
        else:
            raise "mixer error"
        self.target_mixer = copy.deepcopy(self.mixer)
        self.params += list(self.mixer.parameters())
        print('Mixer Size: ')
        print(get_parameters_num(self.mixer.parameters()))
        # 目标网络
        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)
        self.log_stats_t = -self.args.learner_log_interval - 1
        self.train_t = 0
        self.avg_time = 0
        # 预测器
        if self.args.name in ["pd_qmix", "pd_vdn"]:
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
            self.target_mac.set_predictor(self.predictor)
        # 优化器
        if self.args.optimizer == 'adam':
            self.optimiser = Adam(params=self.params, lr=args.lr, weight_decay=getattr(args, "weight_decay", 0))
        else:
            self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        # 奖励标准化
        if self.args.standardise_rewards:
            rew_shape = (1,) if self.args.common_reward else (self.n_agents,)
            self.rew_ms = RunningMeanStd(shape=rew_shape, device=self.device)
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
            self.teacher_mac.set_predictor(self.predictor)
            if self.args.use_cuda:
                self.teacher_mac.cuda()
            self.teacher_mac.set_evaluation_mode()
            self.teacher_mixer = copy.deepcopy(self.mixer)
            self.teacher_mixer.load_state_dict(th.load("{}/mixer.th".format(self.args.teacher_model_dir), map_location=lambda storage, loc: storage))
            if self.args.use_cuda:
                self.teacher_mixer.cuda()
            self.teacher_mixer.eval()

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        start_time = time.time()
        if self.args.use_cuda and str(self.mac.get_device()) == "cpu":
            self.mac.cuda()
        
        self.cheating_prob = self.cheating_schedule.eval(t_env)
        self.mac.cheating_prob = self.cheating_prob
        self.target_mac.cheating_prob = self.cheating_prob
        self.alpha = self.teacher_schedule.eval(t_env)
        
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        if self.args.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)

        # 训练预测器
        self.logger.log_stat("eval/pd_reg_loss_avg", self.predictor.eval_reg_loss_avg.item(), t_env)
        self.logger.log_stat("eval/pd_cla_loss_avg", self.predictor.eval_cla_loss_avg.item(), t_env)
        self.logger.log_stat("eval/pd_obs_loss_avg", self.predictor.eval_obs_loss_avg.item(), t_env)
        self.predictor.unfreeze()
        self.predictor.set_teacher_forcing(t_env)
        if self.args.env in ["sc2", "sc2_v2"] and self.args.predictor_mode == "single_step":
            predict_obs, pd_reg_loss_avg, pd_cla_loss_avg, pd_obs_loss_avg = self.predictor.minibatch_predict(batch)
        else:
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
            self.target_mac.cheating = True

        # Calculate estimated Q-Values
        self.mac.set_train_mode()
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t, predict_obs)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        # double DQN action
        mac_out[avail_actions == 0] = -9999999
        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        with th.no_grad():
            # Set target mac to testing mode
            self.target_mac.set_evaluation_mode()
            target_mac_out = []
            self.target_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                target_agent_outs = self.target_mac.forward(batch, t, predict_obs)
                target_mac_out.append(target_agent_outs)
            # We don't need the first timesteps Q-Value estimate for calculating targets
            target_mac_out = th.stack(target_mac_out, dim=1)  # Concat across time
            # target_mac_out = calculate_target_q(self.target_mac, batch)
            target_mac_out[avail_actions == 0] = -9999999
            # Max over target Q-Values/ Double q learning
            mac_out_detach = mac_out
            # mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach.max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)

            assert getattr(self.args, 'q_lambda', False) == False
            # Set target mixing net to testing mode
            self.target_mixer.eval()
            # Calculate n-step Q-Learning targets
            if self.args.mixer == "qatten":
                target_max_qvals, _, _ = self.target_mixer(target_max_qvals, batch["state"])
            else:
                target_max_qvals, _, _, _, _ = self.target_mixer(target_max_qvals, batch["state"])
            targets = build_td_lambda_targets(rewards, terminated, mask, target_max_qvals, self.args.gamma, self.args.td_lambda)
            targets = targets.detach()

        # Set mixing net to training mode
        self.mixer.train()
        # Mixer
        if self.args.mixer == "qatten":
            chosen_action_qvals, q_attend_regs, head_entropies = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
        else:
            chosen_action_qvals, w1, b1, w2, b2 = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
        
        if self.args.teacher_model_dir != "":
            teacher_mac_out = []
            self.teacher_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                teacher_agent_outs = self.teacher_mac.forward(batch, t, predict_obs)
                teacher_mac_out.append(teacher_agent_outs)
            teacher_mac_out = th.stack(teacher_mac_out, dim=1)  # Concat over time
            # double DQN action
            teacher_mac_out[avail_actions == 0] = -9999999
            # Pick the Q-Values for the actions taken by each agent
            teacher_chosen_action_qvals = th.gather(teacher_mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim
            if self.args.mixer == "qatten":
                teacher_chosen_action_qvals, teacher_q_attend_regs, teacher_head_entropies = self.teacher_mixer(teacher_chosen_action_qvals, batch["state"][:, :-1])
            else:
                teacher_chosen_action_qvals, teacher_w1, teacher_b1, teacher_w2, teacher_b2 = self.teacher_mixer(teacher_chosen_action_qvals, batch["state"][:, :-1])
            # 动作分布做软标签
            _, teacher_actions = th.max(th.nn.functional.softmax(teacher_mac_out, dim=-1), dim=-1)
            ts_action_loss = th.nn.functional.cross_entropy(mac_out.view(-1, self.args.n_actions), teacher_actions.view(-1))
            if teacher_w1 is None:
                # VDN不蒸馏mixer
                ts_value_loss = th.tensor(0.0, device=self.device)
                teacher_loss = ts_action_loss
            else:
                ts_value_loss = 0.1 * th.nn.functional.mse_loss(chosen_action_qvals, teacher_chosen_action_qvals) + th.nn.functional.mse_loss(w1, teacher_w1) + th.nn.functional.mse_loss(b1, teacher_b1) + th.nn.functional.mse_loss(w2, teacher_w2) + th.nn.functional.mse_loss(b2, teacher_b2)
                teacher_loss = self.args.alpha_mac * ts_action_loss + self.args.alpha_mixer * ts_value_loss

        td_error = (chosen_action_qvals - targets)
        td_error2 = 0.5 * td_error.pow(2)

        mask = mask.expand_as(td_error2)
        masked_td_error = td_error2 * mask

        mask_elems = mask.sum()
        td_loss = masked_td_error.sum() / mask_elems

        if self.args.mixer == "qatten":
            mix_loss = td_loss + q_attend_regs
        else:
            mix_loss = td_loss
        if self.args.teacher_model_dir != "":
            mix_loss = (1 - self.alpha) * mix_loss + self.alpha * teacher_loss
        # Optimise
        self.optimiser.zero_grad() 
        mix_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()
        
        self.mixer.eval()
        self.mac.set_evaluation_mode()  # RL训练完毕

        self.train_t += 1
        self.avg_time += (time.time() - start_time - self.avg_time) / self.train_t
        print("Avg cost {} seconds".format(self.avg_time))
        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num
        # 注意将mac和target_mac的cheating还原为False
        if cheating:
            self.mac.cheating = False
            self.target_mac.cheating = False
        
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            # For log
            with th.no_grad():
                mask_elems = mask_elems.item()
                td_error_abs = masked_td_error.abs().sum().item() / mask_elems
                q_taken_mean = (chosen_action_qvals * mask).sum().item() / (mask_elems * self.args.n_agents)
                target_mean = (targets * mask).sum().item() / (mask_elems * self.args.n_agents)
            self.logger.log_stat("train/td_loss", td_loss.item(), t_env)
            self.logger.log_stat("train/mix_loss", mix_loss.item(), t_env)
            if self.args.teacher_model_dir != "":
                self.logger.log_stat("train/ts_loss", teacher_loss.item(), t_env)
                self.logger.log_stat("train/ts_action_loss", ts_action_loss.item(), t_env)
                self.logger.log_stat("train/ts_value_loss", ts_value_loss.item(), t_env)
            self.logger.log_stat("train/grad_norm", grad_norm, t_env)
            self.logger.log_stat("train/td_error_abs", td_error_abs, t_env)
            self.logger.log_stat("train/q_taken_mean", q_taken_mean, t_env)
            self.logger.log_stat("train/target_mean", target_mean, t_env)
            self.logger.log_stat("train/cheating_prob", self.cheating_prob, t_env)
            self.logger.log_stat("train/teacher_forcing_prob", self.predictor.teacher_forcing_prob, t_env)
            self.logger.log_stat("train/alpha", self.alpha, t_env)
            self.logger.log_stat("train/cuda_memory", th.cuda.max_memory_allocated() / 1024**2, t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")
    
    def _update_targets_soft(self, tau):
        for target_param, param in zip(
            self.target_mac.parameters(), self.mac.parameters()
        ):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        if self.mixer is not None:
            for target_param, param in zip(
                self.target_mixer.parameters(), self.mixer.parameters()
            ):
                target_param.data.copy_(
                    target_param.data * (1.0 - tau) + param.data * tau
                )

    def cuda(self):
        self.predictor.cuda()
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.predictor.save_models(path)
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.predictor.load_models(path)
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
        # 将优化器状态移动到 CUDA
        for state in self.optimiser.state.values():
            for k, v in state.items():
                if isinstance(v, th.Tensor):
                    state[k] = v.cuda() 
    
    def log_memory(self, step_name: str):
        if self.args.use_cuda:
            print(f"\n=== {step_name} ===")
            print(f"分配显存: {th.cuda.memory_allocated() / 1024**2:.2f} MB")
            print(f"显存缓存: {th.cuda.memory_reserved() / 1024**2:.2f} MB")
            print(f"当前显存峰值: {th.cuda.max_memory_allocated() / 1024**2:.2f} MB\n")
    
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