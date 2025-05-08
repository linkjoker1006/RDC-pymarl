import random
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import RMSprop, Adam
import time
from components.epsilon_schedules import FlatThenDecayThenFlatSchedule


# 输入延迟观测+延迟动作队列，敌方延迟时间步列表+友方延迟时间步列表作为条件
# 输出1步后的延迟观测
class Predictor_gru(nn.Module):
    def __init__(self, input_shape, output_shape, args):
        # 分头预测，对于0，1的部分用分类模型，对于小数部分用回归模型
        super(Predictor_gru, self).__init__()
        self.args = args
        self.input_shape = input_shape
        self.output_shape_regression, self.output_shape_classification, self.output_shape = output_shape
        
        self.fc_shared = nn.Linear(input_shape, args.pd_hidden_dim)
        self.rnn = nn.GRUCell(args.pd_hidden_dim, args.pd_hidden_dim)
        self.fc_regression = nn.Linear(args.pd_hidden_dim, self.output_shape_regression)  # 对每个浮点数观测值做回归
        if self.args.res_classification:
            self.class_num = 3
        else:
            self.class_num = 2
        self.fc_classification = nn.Linear(args.pd_hidden_dim, self.output_shape_classification * self.class_num) if self.output_shape_classification > 0 else None
        self.softmax = nn.Softmax(dim=-1)
    
    def init_hidden(self, batch_size, max_t):
        # 使用rnn的权重来创建隐藏状态
        hidden_states = self.rnn.weight_hh.new(1, self.args.pd_hidden_dim).zero_()
        if hidden_states is not None:
            hidden_states = hidden_states.unsqueeze(0).expand(batch_size, max_t, self.args.n_agents, -1)  # bav
        return hidden_states
    
    def forward(self, inputs, hidden_state):
        b, t, n, d = inputs.shape
        
        inputs = inputs.reshape(-1, d)
        x = F.relu(self.fc_shared(inputs), inplace=True)
        h_in = hidden_state.reshape(-1, self.args.pd_hidden_dim)
        h = self.rnn(x, h_in)
        reg = self.fc_regression(h)
        reg = reg.view(b, t, n, self.output_shape_regression)
        cla_logits = self.fc_classification(h).reshape(-1, self.output_shape_classification, self.class_num)
        _, cla_indices = th.max(self.softmax(cla_logits), dim=-1)
        if self.args.res_classification:
            # 分类结果的索引-1，将结果从0,1,2变成-1,0,1
            cla_indices = cla_indices - 1
        cla_logits = cla_logits.view(b, t, n, self.output_shape_classification, self.class_num)
        cla_indices = cla_indices.view(b, t, n, self.output_shape_classification)
        
        return reg, cla_logits, cla_indices, h


class Predictor_Controller():
    def __init__(self, input_shape, output_shape, args):
        # 分头预测，对于0，1的部分用分类模型，对于小数部分用回归模型
        super(Predictor_Controller, self).__init__()
        self.args = args
        self._cal_max_delay()
        self.input_shape = input_shape
        self.output_shape_regression, self.output_shape_classification, self.output_shape = output_shape
        self.predictor = Predictor_gru(input_shape, output_shape, args)
        self.schedule = FlatThenDecayThenFlatSchedule(args.teacher_forcing_start_value, args.teacher_forcing_end_value, args.teacher_forcing_start_time, args.teacher_forcing_end_time, decay="linear")
        self.teacher_forcing_prob = self.schedule.eval(0)
        self.use_history = args.use_history
        self.params = self.predictor.parameters()
        if self.args.optimizer == 'adam':
            self.optimiser = Adam(params=self.params, lr=args.pd_lr, weight_decay=getattr(args, "weight_decay", 0))
        else:
            self.optimiser = RMSprop(params=self.params, lr=args.pd_lr, alpha=args.optim_alpha, eps=args.optim_eps)
        # 记录推理时预测误差，每次训练时归零
        self.eval_reg_loss_avg = th.tensor(0.0, device=next(self.predictor.parameters()).device)
        self.eval_cla_loss_avg = th.tensor(0.0, device=next(self.predictor.parameters()).device)
        self.eval_obs_loss_avg = th.tensor(0.0, device=next(self.predictor.parameters()).device)
        self.eval_cnt = 0
        # 重新设置随机种子，避免因为预测器不同导致与环境交互的随机结果不同
        th.manual_seed(args.seed)
        th.cuda.manual_seed(args.seed)

    def init_hidden(self, batch_size, max_t):
        return self.predictor.init_hidden(batch_size, max_t)

    def _cal_max_delay(self):
        if self.args.delay_type == "none":
            self.max_delay = self.args.delay_value
        elif self.args.delay_type == 'f':
            self.max_delay = self.args.delay_value
        elif self.args.delay_type == 'pf':
            self.max_delay = self.args.delay_value
        elif self.args.delay_type == 'uf':
            self.max_delay = self.args.delay_value + self.args.delay_scope
        else:
            assert False, "This type of delay(" + self.args.delay_type + ") is currently not supported"

    def set_teacher_forcing(self, t_env):
        self.teacher_forcing_prob = self.schedule.eval(t_env)

    def forward(self, inputs, hidden_state, batch, pd_step, t=-1):
        reg, cla_logits, cla_indices, h = self.predictor(inputs, hidden_state)
        type_bits = self.args.unit_type_bits
        # 预计算一些常用的索引值
        move_feats, (n_enemies, enemy_feats), (n_allies, ally_feats), own_feats = self.args.obs_component
        # 预计算结果tensor的总大小
        total_dim = (move_feats +  # 自身特征
                     n_enemies * enemy_feats +  # 敌人特征
                     n_allies * ally_feats +  # 盟友特征
                     own_feats)  # 剩余特征
        # 计算mask用来保留已经完成预测的内容
        enemy_mask, ally_mask = self._make_mask(batch, pd_step, t)
        enemy_mask = enemy_mask.unsqueeze(-1)
        ally_mask = ally_mask.unsqueeze(-1)
        # results直接是预测obs，而不是残差
        results = inputs[..., :total_dim].clone()
        # 填充移动特征
        curr_pos = move_feats
        # 填充敌人特征
        for e in range(n_enemies):
            # 分类结果
            if self.args.res_classification:
                results[..., curr_pos] = results[..., curr_pos] + cla_indices[..., e] * enemy_mask[..., e, 0]
                # 残差预测导致可能溢出，结果只需要0和1
                results[..., curr_pos] = th.where(results[..., curr_pos] == -1, th.tensor(0, device=results.device), results[..., curr_pos])
                results[..., curr_pos] = th.where(results[..., curr_pos] == 2, th.tensor(1, device=results.device), results[..., curr_pos])
            else:
                results[..., curr_pos] = results[..., curr_pos] * th.logical_not(enemy_mask[..., e, 0]) + cla_indices[..., e] * enemy_mask[..., e, 0]
            curr_pos += 1
            # 回归结果
            reg_start = e * (enemy_feats - type_bits - 1)
            reg_end = (e + 1) * (enemy_feats - type_bits - 1)
            feat_size = reg_end - reg_start
            results[..., curr_pos:curr_pos + feat_size] = results[..., curr_pos:curr_pos + feat_size] + reg[..., reg_start:reg_end] * enemy_mask[..., e, :].repeat(1, 1, 1, feat_size)
            curr_pos += feat_size
            # 单位类型
            curr_pos += type_bits
        # 填充盟友特征 (类似敌人特征的处理)
        reg_idx = n_enemies * (enemy_feats - type_bits - 1)
        for a in range(n_allies):
            # 分类结果
            if self.args.res_classification:
                results[..., curr_pos] = results[..., curr_pos] + cla_indices[..., n_enemies + a] * ally_mask[..., a, 0]
                # 残差预测导致可能溢出，结果只需要0和1
                results[..., curr_pos] = th.where(results[..., curr_pos] == -1, th.tensor(0, device=results.device), results[..., curr_pos])
                results[..., curr_pos] = th.where(results[..., curr_pos] == 2, th.tensor(1, device=results.device), results[..., curr_pos])
            else:
                results[..., curr_pos] = results[..., curr_pos] * th.logical_not(ally_mask[..., a, 0]) + cla_indices[..., n_enemies + a] * ally_mask[..., a, 0]
            curr_pos += 1
            # 回归结果
            reg_start = reg_idx + a * (ally_feats - type_bits - 1)
            reg_end = reg_idx + (a + 1) * (ally_feats - type_bits - 1)
            feat_size = reg_end - reg_start
            results[..., curr_pos:curr_pos + feat_size] = results[..., curr_pos:curr_pos + feat_size] + reg[..., reg_start:reg_end] * ally_mask[..., a, :].repeat(1, 1, 1, feat_size)
            curr_pos += feat_size
            # 单位类型
            curr_pos += type_bits
        # 填充其他特征
        curr_pos += own_feats
        assert curr_pos == total_dim
        
        return results, reg, cla_logits, h

    def _make_mask(self, batch, pd_step, t=-1):
        # 计算预测pd_step步时输入部分的延迟值(pd_step<0时没有意义，只是为了后面取形状)
        enemy_delay_values, ally_delay_values = self._update_delay_values(batch, pd_step, False, False)
        # 输入历史和不使用mask时返回全1掩码
        enemy_mask = th.zeros_like(enemy_delay_values) == 0
        ally_mask = th.zeros_like(ally_delay_values) == 0
        if self.args.mask_prediction and pd_step > -1:
            # 延迟大于0的地方需要预测
            enemy_mask = enemy_delay_values > 0
            ally_mask = ally_delay_values > 0
        if t > -1:
            return enemy_mask[:, t:t + 1 :, :], ally_mask[:, t:t + 1 :, :]
        else:
            return enemy_mask, ally_mask

    def _update_delay_values(self, batch, pd_step, time_step=False, onehot=False):
        # time_step为True时输出值含义为观测的时间步，为False时输出值含义为延迟值，注意只有False时onehot才可能为True
        assert not (time_step and onehot), "time_step和onehot不能同时为True"
        enemy_delay_values = batch["enemy_delay_values"].clone()
        ally_delay_values = batch["ally_delay_values"].clone()
        # 创建一个值为0到t-1的张量，用来限制1步延迟值的大小
        max_t = th.arange(enemy_delay_values.size(1), device=enemy_delay_values.device).view(1, -1, 1, 1)
        enemy_delay_values = th.min(enemy_delay_values + pd_step, max_t)
        if not time_step:
            enemy_delay_values = max_t - enemy_delay_values
            enemy_delay_values[enemy_delay_values > self.max_delay] = 0
        if onehot:
            enemy_delay_values = th.nn.functional.one_hot(enemy_delay_values, num_classes=self.max_delay + 1).view(enemy_delay_values.shape[0], enemy_delay_values.shape[1], enemy_delay_values.shape[2], -1)
        
        ally_delay_values = th.min(ally_delay_values + pd_step, max_t)
        if not time_step:
            ally_delay_values = max_t - ally_delay_values
            ally_delay_values[ally_delay_values > self.max_delay] = 0
        if onehot:
            ally_delay_values = th.nn.functional.one_hot(ally_delay_values, num_classes=self.max_delay + 1).view(ally_delay_values.shape[0], ally_delay_values.shape[1], ally_delay_values.shape[2], -1)

        return enemy_delay_values, ally_delay_values
    
    def _just_divide(self, x):
        move_feats, (n_enemies, enemy_feats), (n_allies, ally_feats), own_feats = self.args.obs_component
        unit_type_bits = self.args.unit_type_bits
        x_regression = th.zeros((x.shape[0], x.shape[1], x.shape[2], self.output_shape_regression), device=x.device)
        x_classification = th.zeros((x.shape[0], x.shape[1], x.shape[2], self.output_shape_classification), device=x.device)
        for e in range(n_enemies):
            x_regression[:, :, :, e * (enemy_feats - unit_type_bits - 1):(e + 1) * (enemy_feats - unit_type_bits - 1)] = x[:, :, :, move_feats + e * enemy_feats + 1:move_feats + (e + 1) * enemy_feats - unit_type_bits]
            x_classification[:, :, :, e] = x[:, :, :, move_feats + e * enemy_feats]
        for a in range(n_allies):
            x_regression[:, :, :, n_enemies * (enemy_feats - unit_type_bits - 1) + a * (ally_feats - unit_type_bits - 1):n_enemies * (enemy_feats - unit_type_bits - 1) + (a + 1) * (ally_feats - unit_type_bits - 1)] = x[:, :, :, move_feats + n_enemies * enemy_feats + a * ally_feats + 1:move_feats + n_enemies * enemy_feats + (a + 1) * ally_feats - unit_type_bits]
            x_classification[:, :, :, n_enemies + a] = x[:, :, :, move_feats + n_enemies * enemy_feats + a * ally_feats]

        return x_regression, x_classification
    
    def _build_label(self, batch, pd_step):
        real_obs = batch["real_obs"]
        move_feats, (n_enemies, enemy_feats), (n_allies, ally_feats), own_feats = self.args.obs_component
        # 计算pd_step步预测后观测对应的时间步
        enemy_delay_values, ally_delay_values = self._update_delay_values(batch, pd_step, True, False)
        # 从无延迟obs中提取一步预测的label
        labels = real_obs.clone()
        for e in range(n_enemies):
            indices = enemy_delay_values[:, :, :, e]
            feats = real_obs[:, :, :, move_feats + (e * enemy_feats): move_feats + ((e + 1) * enemy_feats)]
            labels[:, :, :, move_feats + (e * enemy_feats): move_feats + ((e + 1) * enemy_feats)] = th.gather(feats, 1, indices.unsqueeze(-1).expand(-1, -1, -1, enemy_feats))
        for a in range(n_allies):
            indices = ally_delay_values[:, :, :, a]
            feats = real_obs[:, :, :, move_feats + (n_enemies * enemy_feats) + (a * ally_feats): move_feats + (n_enemies * enemy_feats) + ((a + 1) * ally_feats)]
            labels[:, :, :, move_feats + (n_enemies * enemy_feats) + (a * ally_feats): move_feats + (n_enemies * enemy_feats) + ((a + 1) * ally_feats)] = th.gather(feats, 1, indices.unsqueeze(-1).expand(-1, -1, -1, ally_feats))
        
        return labels

    def _build_residual_label(self, batch, last_obs, pd_step, t=-1):
        labels = self._build_label(batch, pd_step)
        res_labels = labels - last_obs
        if self.args.res_classification:
            labels_regression, labels_classification = self._just_divide(res_labels)
        else:
            # 回归预测目标是残差
            _, labels_classification = self._just_divide(labels)
            labels_regression, _ = self._just_divide(res_labels)
        labels_classification = labels_classification.long()
        if t > -1:
            labels_regression = labels_regression[:, t:t + 1, :, :]
            labels_classification = labels_classification[:, t:t + 1, :, :] if labels_classification is not None else None

        return labels_regression, labels_classification
    
    def _build_input(self, batch, predict_obs, pd_step, t=-1):
        # 要用上一步的动作
        actions = th.roll(batch["actions_onehot"], shifts=1, dims=1)
        actions[:, 0, :, :] = th.zeros_like(actions[:, 0, :, :])
        if self.use_history:
            # 先输入历史数据
            history_idx = pd_step - self.args.n_expand_action + 1
            if history_idx < 0:
                # 计算0步预测后的延迟值的onehot编码
                if self.args.one_hot_delay: 
                    enemy_delay_values, ally_delay_values = self._update_delay_values(batch, 0, False, True)
                else:
                    enemy_delay_values, ally_delay_values = self._update_delay_values(batch, 0, False, False)
                expand_obs = th.cat([batch["obs"], actions, enemy_delay_values, ally_delay_values], dim=-1)
                # history_idx为负，右移后补最近的观测
                expand_obs = th.roll(expand_obs, shifts=-history_idx, dims=1)
                expand_obs[:, :-history_idx, :, :] = expand_obs[:, -history_idx, :, :].unsqueeze(1).repeat(1, -history_idx, 1, 1)
                if self.predictor.training:
                    return expand_obs
                else:
                    return expand_obs[:, t:t + 1]
            else:
                pd_step = history_idx
        # 如果使用历史数据，当历史数据输入完毕后，和不使用历史数据的输入相同
        # 计算pd_step步预测后的延迟值的onehot编码
        if self.args.one_hot_delay: 
            enemy_delay_values, ally_delay_values = self._update_delay_values(batch, pd_step, False, True)
        else:
            enemy_delay_values, ally_delay_values = self._update_delay_values(batch, pd_step, False, False)
        if self.predictor.training:
            if random.random() < self.teacher_forcing_prob:
                labels = self._build_label(batch, pd_step)
                expand_obs = th.cat([labels, actions, enemy_delay_values, ally_delay_values], dim=-1)
            else:
                # 训练时predict_obs形状为[b, t, n, -1]
                expand_obs = th.cat([predict_obs, actions, enemy_delay_values, ally_delay_values], dim=-1)
        else:
            # 推理时predict_obs形状为[b, 1, n, -1]
            expand_obs = th.cat([predict_obs, actions[:, t:t + 1], enemy_delay_values[:, t:t + 1], ally_delay_values[:, t:t + 1]], dim=-1)

        return expand_obs

    def do_predict(self, batch, t=-1):
        # 逐步预测、记录，需要额外开辟一个隐状态变量，并且需要记录每步的变化
        hidden_state = None
        pd_reg_loss_avg = th.tensor(0.0, device=batch["obs"].device)
        pd_cla_loss_avg = th.tensor(0.0, device=batch["obs"].device)
        pd_obs_loss_avg = th.tensor(0.0, device=batch["obs"].device)
        predict_obs = batch["obs"] if t == -1 else batch["obs"][:, t:t + 1]
        pd_loss = th.tensor(0.0, device=batch["obs"].device)
        # 对补0数据进行修改，创建loss计算所需要的mask
        batch_size, time_length, _ = batch["terminated"].shape
        if t == -1:
            mask_reg = th.ones((batch_size, time_length, self.args.n_agents, self.output_shape_regression), dtype=th.bool, device=batch["terminated"].device)
            mask_cla = th.ones((batch_size, time_length, self.args.n_agents, self.output_shape_classification), dtype=th.bool, device=batch["terminated"].device)
            mask_obs = th.ones((batch_size, time_length, self.args.n_agents, batch["obs"].shape[3]), dtype=th.bool, device=batch["terminated"].device)
        else:
            mask_reg = th.ones((batch_size, 1, self.args.n_agents, self.output_shape_regression), dtype=th.bool, device=batch["terminated"].device)
            mask_cla = th.ones((batch_size, 1, self.args.n_agents, self.output_shape_classification), dtype=th.bool, device=batch["terminated"].device)
            mask_obs = th.ones((batch_size, 1, self.args.n_agents, batch["obs"].shape[3]), dtype=th.bool, device=batch["terminated"].device)
        # TODO: 其实好像batch["filled"]记录了哪些数据是填充的，也可以利用这个来计算mask
        if (batch["terminated"] == 1).any():
            indices = th.where(batch["terminated"] == 1)[0:2]
            for b_idx, t_idx in zip(*indices):
                start_t = min(t_idx + 2, time_length)
                if start_t < time_length:
                    # 创建从start_t到time_length的时间步序列
                    t_range = th.arange(start_t, time_length, device=batch["terminated"].device)
                    # 扩展维度以匹配目标形状
                    t_values_enemy = t_range.view(-1, 1, 1).expand(-1, batch["enemy_delay_values"].size(2), batch["enemy_delay_values"].size(3))
                    t_values_ally = t_range.view(-1, 1, 1).expand(-1, batch["ally_delay_values"].size(2), batch["ally_delay_values"].size(3))
                    # 赋值
                    batch["enemy_delay_values"][b_idx, start_t:] = t_values_enemy
                    batch["ally_delay_values"][b_idx, start_t:] = t_values_ally
                if t == -1:
                    mask_reg[b_idx, start_t:, :, :] = False
                    mask_cla[b_idx, start_t:, :, :] = False
                    mask_obs[b_idx, start_t:, :, :] = False
                else:
                    pass
        # 不需要预测，直接返回观测值
        if self.args.predictor_mode == "none":
            inputs_regression, inputs_classification = self._just_divide(predict_obs)
            real_obs = batch["real_obs"] if t == -1 else batch["real_obs"][:, t:t + 1]
            labels_regression, labels_classification = self._just_divide(real_obs)
            pd_reg_loss = th.nn.functional.mse_loss(inputs_regression, labels_regression, reduction="none")
            pd_reg_loss_avg = (pd_reg_loss * mask_reg.float()).sum() / mask_reg.sum()
            pd_cla_loss = th.nn.functional.mse_loss(inputs_classification, labels_classification, reduction="none")
            pd_cla_loss_avg = (pd_cla_loss * mask_cla.float()).sum() / mask_cla.sum()
        # 进行单步多轮预测
        elif self.args.predictor_mode == "single_step":
            if self.use_history:
                max_i = self.max_delay + self.args.n_expand_action - 1
            else:
                max_i = self.max_delay
            for i in range(max_i):
                pd_step = i - self.args.n_expand_action + 1 if self.use_history else i
                inputs = self._build_input(batch, predict_obs, i, t)
                # 单步预测简化延迟值，变成0&1序列
                condition_dim = self.args.n_enemies + self.args.n_allies
                inputs[..., -condition_dim:] = (inputs[..., -condition_dim:] > 0).to(inputs.dtype)
                hidden_state = self.init_hidden(batch.batch_size, inputs.shape[1]) if hidden_state is None else hidden_state
                results, reg, cla_logits, hidden_state = self.forward(inputs, hidden_state, batch, pd_step, t)
                # 输入历史时不计算误差，也不自回归
                if not self.use_history or (self.use_history and i >= self.args.n_expand_action - 1):
                    # 计算误差，考虑了使用teacher forcing的情况
                    labels_regression, labels_classification = self._build_residual_label(batch, inputs[..., :self.args.obs_shape].clone(), pd_step + 1, t)
                    # 计算最小均方误差
                    pd_reg_loss = th.nn.functional.mse_loss(reg, labels_regression, reduction="none")
                    pd_reg_loss = (pd_reg_loss * mask_reg.float()).sum() / mask_reg.sum()
                    pd_reg_loss_avg = (pd_reg_loss_avg * pd_step + pd_reg_loss) / (pd_step + 1)
                    # 计算交叉熵损失
                    if self.args.res_classification:
                        labels_classification = labels_classification + 1
                    pd_cla_loss = th.nn.functional.cross_entropy(cla_logits.reshape(-1, self.predictor.class_num), labels_classification.reshape(-1), reduction="none")
                    pd_cla_loss = (pd_cla_loss * mask_cla.view(-1).float()).sum() / mask_cla.view(-1).sum()
                    pd_cla_loss_avg = (pd_cla_loss_avg * pd_step + pd_cla_loss) / (pd_step + 1)
                    # 自回归
                    predict_obs = results
        # 进行多步单轮预测
        elif self.args.predictor_mode == "multi_step":
            if self.use_history:
                max_i = 1 + self.args.n_expand_action - 1
            else:
                max_i = 1
            for i in range(max_i):
                pd_step = i - self.args.n_expand_action + 1 if self.use_history else i
                inputs = self._build_input(batch, predict_obs, i, t)
                hidden_state = self.init_hidden(batch.batch_size, inputs.shape[1]) if hidden_state is None else hidden_state
                results, reg, cla_logits, hidden_state = self.forward(inputs, hidden_state, batch, pd_step, t)
                # 输入历史时不计算误差
                if not self.use_history or (self.use_history and i >= self.args.n_expand_action - 1):
                    # 计算误差，考虑了使用teacher forcing的情况
                    labels_regression, labels_classification = self._build_residual_label(batch, inputs[..., :self.args.obs_shape].clone(), self.max_delay, t)
                    # 计算最小均方误差
                    pd_reg_loss = th.nn.functional.mse_loss(reg, labels_regression, reduction="none")
                    pd_reg_loss_avg = (pd_reg_loss * mask_reg.float()).sum() / mask_reg.sum()
                    # 计算交叉熵损失
                    if self.args.res_classification:
                        labels_classification = labels_classification + 1
                    pd_cla_loss = th.nn.functional.cross_entropy(cla_logits.reshape(-1, self.predictor.class_num), labels_classification.reshape(-1), reduction="none")
                    pd_cla_loss_avg = (pd_cla_loss * mask_cla.view(-1).float()).sum() / mask_cla.view(-1).sum()
                    # 对预测结果进行限制，防止溢出
                    predict_obs = results
        else:
            assert False, "predictor_mode error"

        if self.predictor.training:
            if self.args.predictor_mode != "none":
                pd_loss = 10 * pd_reg_loss_avg + pd_cla_loss_avg  # 对齐数量级
                self.optimiser.zero_grad()
                pd_loss.backward()
                grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
                self.optimiser.step()
            # 计算观测总误差
            pd_obs_loss_avg = th.nn.functional.mse_loss(predict_obs, batch["real_obs"])
            # 重置推理误差统计
            self.eval_reg_loss_avg = th.tensor(0.0, device=batch["obs"].device)
            self.eval_cla_loss_avg = th.tensor(0.0, device=batch["obs"].device)
            self.eval_obs_loss_avg = th.tensor(0.0, device=batch["obs"].device)
            self.eval_cnt = 0
        else:
            # 计算观测总误差
            pd_obs_loss_avg = th.nn.functional.mse_loss(predict_obs, batch["real_obs"][:, t:t + 1])
            # 计算推理误差
            self.eval_reg_loss_avg = (self.eval_reg_loss_avg * self.eval_cnt + pd_reg_loss_avg) / (self.eval_cnt + 1)
            self.eval_cla_loss_avg = (self.eval_cla_loss_avg * self.eval_cnt + pd_cla_loss_avg) / (self.eval_cnt + 1)
            self.eval_obs_loss_avg = (self.eval_obs_loss_avg * self.eval_cnt + pd_obs_loss_avg) / (self.eval_cnt + 1)
            self.eval_cnt += 1

        return predict_obs.detach(), pd_reg_loss_avg.detach(), pd_cla_loss_avg.detach(), pd_obs_loss_avg.detach()

    def cuda(self):
        self.predictor.cuda()
    
    def cpu(self):
        self.predictor.cpu()

    def freeze(self):
        self.predictor.requires_grad_(False)
        self.predictor.eval()  # 确保 BatchNorm, Dropout 等层也处于评估模式
        
    def unfreeze(self):
        self.predictor.requires_grad_(True)
        self.predictor.train()
    
    def save_models(self, path):
        th.save(self.predictor.state_dict(), "{}/predictor.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/predictor_opt.th".format(path))
    
    def load_models(self, path):
        self.predictor.load_state_dict(th.load("{}/predictor.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/predictor_opt.th".format(path), map_location=lambda storage, loc: storage))
        # 将优化器状态移动到 CUDA
        for state in self.optimiser.state.values():
            for k, v in state.items():
                if isinstance(v, th.Tensor):
                    state[k] = v.cuda() 

    def minibatch_predict(self, batch, t=-1):
        num_minibatch = self.args.num_minibatch_predictor
        results_1 = []
        results_2 = th.tensor(0.0, device=batch["obs"].device)
        results_3 = th.tensor(0.0, device=batch["obs"].device)
        results_4 = th.tensor(0.0, device=batch["obs"].device)
        if t ==-1:
            length = self.args.batch_size
            mini_length = length // num_minibatch
            # 分成4个minibatch
            for i in range(num_minibatch):
                predict_obs, pd_reg_loss_avg, pd_cla_loss_avg, pd_obs_loss_avg = self.do_predict(batch[i * mini_length:(i + 1) * mini_length])
                results_1.append(predict_obs)
                results_2 += pd_reg_loss_avg
                results_3 += pd_cla_loss_avg
                results_4 += pd_obs_loss_avg
            results_2 = results_2 / num_minibatch
            results_3 = results_3 / num_minibatch
            results_4 = results_4 / num_minibatch
            return th.cat(results_1, dim=0), results_2, results_3, results_4
        else:
            return self.do_predict(batch, t)  
