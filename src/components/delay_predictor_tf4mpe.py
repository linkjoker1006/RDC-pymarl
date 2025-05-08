import random
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import RMSprop, Adam
import math
import time
from components.epsilon_schedules import FlatThenDecayThenFlatSchedule


# 位置编码实现
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = th.zeros(max_len, d_model)
        position = th.arange(0, max_len, dtype=th.float).unsqueeze(1)
        div_term = th.exp(th.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = th.sin(position * div_term)
        pe[:, 1::2] = th.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# transformer 预测器，单步多步相同，可以纯encoder和encoder-decoder
class Predictor_transformer(nn.Module):
    def __init__(self, input_shape, output_shape, args):
        super(Predictor_transformer, self).__init__()
        self.args = args
        self.input_shape = input_shape
        self.output_shape_regression, self.output_shape_classification, self.output_shape = output_shape
        # encoder-only & encoder-decoder
        self.structure = args.transformer_structure
        
        # 输入投影层
        self.fc_encoder = nn.Linear(input_shape, args.pd_hidden_dim)
        self.fc_decoder = nn.Linear(self.args.obs_shape, args.pd_hidden_dim)
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=args.pd_hidden_dim,
            nhead=4,  # 可以通过args配置
            dim_feedforward=args.pd_hidden_dim * 4,
            dropout=0.1,  # 可以通过args配置
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=1  # 可以通过args配置
        )
        # Transformer解码器层
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=args.pd_hidden_dim,
            nhead=4,  # 可以通过args配置
            dim_feedforward=args.pd_hidden_dim * 4,
            dropout=0.1,  # 可以通过args配置
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=1  # 可以通过args配置
        )
        # 输出投影层
        if self.structure == "enoder-only":
            self.fc_shared = nn.Linear(args.pd_hidden_dim * args.n_expand_action, args.pd_hidden_dim)
        self.fc_regression = nn.Linear(args.pd_hidden_dim, self.output_shape_regression)
        if self.args.res_classification:
            self.class_num = 3
        else:
            self.class_num = 2
        self.fc_classification = nn.Linear(args.pd_hidden_dim, self.output_shape_classification * self.class_num) if self.output_shape_classification > 0 else None
        self.softmax = nn.Softmax(dim=-1)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(args.pd_hidden_dim)
        self.pos_decoder = PositionalEncoding(args.pd_hidden_dim)
    
    def forward(self, input_seq, target_seq, target_mask):
        b, t, input_len, n, d = input_seq.shape
        # 特征提取
        x = self.fc_encoder(input_seq)
        x = F.relu(x, inplace=True)
        # 重塑为序列形式 (batch * n_agents * max_t, input_len, hidden_dim)
        x = x.reshape(b * t * n, input_len, -1)
        # 添加位置编码
        x = self.pos_encoder(x)
        # Transformer编码
        memory = self.transformer_encoder(x)
        if self.structure == "encoder-decoder":
            # 目标序列处理
            target_seq = self.fc_decoder(target_seq)
            target_seq = F.relu(target_seq, inplace=True)
            target_len = target_seq.shape[2]
            target_seq = target_seq.reshape(b * t * n, target_len, -1)
            target_seq = self.pos_decoder(target_seq)
            # Transformer解码
            x = self.transformer_decoder(target_seq, memory, target_mask)
            # 重塑回原始维度
            x = x.view(b, t, target_len, n, -1)
            x = x.reshape(-1, self.args.pd_hidden_dim)
            # 回归预测
            reg = self.fc_regression(x)
            reg = reg.view(b, t, target_len, n, self.output_shape_regression)
            # 分类预测
            if self.output_shape_classification > 0:
                cla_logits = self.fc_classification(x).reshape(-1, self.output_shape_classification, self.class_num)
                _, cla_indices = th.max(self.softmax(cla_logits), dim=-1)
                if self.args.res_classification:
                    # 分类结果的索引-1，将结果从0,1,2变成-1,0,1
                    cla_indices = cla_indices - 1
                cla_logits = cla_logits.view(b, t, target_len, n, self.output_shape_classification, self.class_num)
                cla_indices = cla_indices.view(b, t, target_len, n, self.output_shape_classification)
            else:
                cla_logits = None
                cla_indices = None
        elif self.structure == "encoder-only":
            # 不使用解码器
            memory = memory.flatten(start_dim=1)
            x = self.fc_shared(memory)
            # 回归预测
            reg = self.fc_regression(x)
            reg = reg.view(b, t, 1, n, self.output_shape_regression)
            # 分类预测
            if self.output_shape_classification > 0:
                cla_logits = self.fc_classification(x).reshape(-1, self.output_shape_classification, self.class_num)
                _, cla_indices = th.max(self.softmax(cla_logits), dim=-1)
                if self.args.res_classification:
                    # 分类结果的索引-1，将结果从0,1,2变成-1,0,1
                    cla_indices = cla_indices - 1
                cla_logits = cla_logits.view(b, t, 1, n, self.output_shape_classification, self.class_num)
                cla_indices = cla_indices.view(b, t, 1, n, self.output_shape_classification)
            else:
                cla_logits = None
                cla_indices = None
        else:
            assert False, "This structure(" + self.structure + ") is currently not supported"

        return reg, cla_logits, cla_indices


class Predictor_Controller():
    def __init__(self, input_shape, output_shape, args):
        # 分头预测，对于0，1的部分用分类模型，对于小数部分用回归模型
        super(Predictor_Controller, self).__init__()
        self.args = args
        self._cal_max_delay()
        self.input_shape = input_shape
        self.output_shape_regression, self.output_shape_classification, self.output_shape = output_shape
        self.predictor = Predictor_transformer(input_shape, output_shape, args)
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

    def forward(self, input_seq, target_seq, target_mask, batch, pd_step, t=-1):
        reg, cla_logits, cla_indices = self.predictor(input_seq, target_seq, target_mask)
        reg = reg[:, :, -1]
        cla_logits = cla_logits[:, :, -1] if cla_logits is not None else None
        cla_indices = cla_indices[:, :, -1] if cla_indices is not None else None
        # 预计算一些常用的索引值
        move_feats, (n_enemies, enemy_feats), (n_allies, ally_feats), own_feats = self.args.obs_component
        # 预计算结果tensor的总大小
        total_dim = (move_feats +  # 自身特征
                     n_enemies * enemy_feats +  # 敌人特征
                     n_allies * ally_feats +  # 盟友特征
                     own_feats)  # 剩余特征
        # 计算mask用来保留已经完成预测的内容
        enemy_mask, ally_mask = self._make_mask(batch, pd_step, t)
        enemy_mask = enemy_mask.unsqueeze(-1).repeat(1, 1, 1, 1, enemy_feats)
        ally_mask = ally_mask.unsqueeze(-1).repeat(1, 1, 1, 1, ally_feats)
        # results直接是预测obs，而不是残差
        results = input_seq[:, :, -1, :, :total_dim].clone()
        # 填充移动特征
        curr_pos = move_feats
        # 填充敌人特征
        for e in range(n_enemies):
            # 回归结果
            reg_start = e * enemy_feats
            reg_end = (e + 1) * enemy_feats
            results[..., curr_pos:curr_pos + enemy_feats] = results[..., curr_pos:curr_pos + enemy_feats] + reg[..., reg_start:reg_end] * enemy_mask[..., e, :]
            curr_pos += enemy_feats
        # 填充盟友特征 (类似敌人特征的处理)
        reg_idx = n_enemies * enemy_feats
        for a in range(n_allies):
            reg_start = reg_idx + a * ally_feats
            reg_end = reg_idx + (a + 1) * ally_feats
            if self.args.env_args["key"] == "pz-mpe-simple-reference-v3":
                # 分类结果
                if self.args.res_classification:
                    results[..., curr_pos:curr_pos + ally_feats] = results[..., curr_pos:curr_pos + ally_feats] + cla_indices[..., :] * ally_mask[..., a, :]
                    # 残差预测导致可能溢出，结果只需要0和1
                    results = th.where(results == -1, th.tensor(0, device=results.device), results)
                    results = th.where(results == 2, th.tensor(1, device=results.device), results)
                else:
                    results[..., curr_pos:curr_pos + ally_feats] = results[..., curr_pos:curr_pos + ally_feats] * th.logical_not(ally_mask[..., a, :]) + cla_indices[..., :] * ally_mask[..., a, :]
            elif self.args.env_args["key"] == "pz-mpe-simple-spread-v3" or self.args.env_args["key"] == "pz-mpe-simple-tag-v3":
                # 回归结果
                results[..., curr_pos:curr_pos + ally_feats] = results[..., curr_pos:curr_pos + ally_feats] + reg[..., reg_start:reg_end] * ally_mask[..., a, :]
            curr_pos += ally_feats
        # 填充其他特征
        curr_pos += own_feats
        assert curr_pos == total_dim
        
        return results, reg, cla_logits
    
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

    def _update_delay_values(self, batch, pd_step, time_step=True, onehot=False):
        # time_step为True时输出值含义为观测的时间步，为False时输出值含义为延迟值，注意只有False时onehot才可能为True
        assert not (time_step and onehot), "time_step和onehot不能同时为True"
        enemy_delay_values = batch["enemy_delay_values"]
        ally_delay_values = batch["ally_delay_values"]
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
        x_regression = th.zeros((x.shape[0], x.shape[1], x.shape[2], self.output_shape_regression), device=x.device)
        x_classification = th.zeros((x.shape[0], x.shape[1], x.shape[2], self.output_shape_classification), device=x.device) if self.output_shape_classification > 0 else None
        for e in range(n_enemies):
            x_regression[:, :, :, e * enemy_feats:(e + 1) * enemy_feats] = x[:, :, :, move_feats + e * enemy_feats:move_feats + (e + 1) * enemy_feats]
        for a in range(n_allies):
            if self.args.env_args["key"] == "pz-mpe-simple-reference-v3":
                # 分类标签
                x_classification[:, :, :, a * ally_feats:(a + 1) * ally_feats] = x[:, :, :, move_feats + n_enemies * enemy_feats + a * ally_feats:move_feats + n_enemies * enemy_feats + (a + 1) * ally_feats]
            elif self.args.env_args["key"] == "pz-mpe-simple-spread-v3" or self.args.env_args["key"] == "pz-mpe-simple-tag-v3":
                # 回归标签
                x_regression[:, :, :, n_enemies * enemy_feats + a * ally_feats:n_enemies * enemy_feats + (a + 1) * ally_feats] = x[:, :, :, move_feats + n_enemies * enemy_feats + a * ally_feats:move_feats + n_enemies * enemy_feats + (a + 1) * ally_feats]
        
        return x_regression, x_classification
    
    def _build_label(self, batch, pd_step):
        real_obs = batch["real_obs"]
        move_feats, (n_enemies, enemy_feats), (n_allies, ally_feats), own_feats = self.args.obs_component
        # 计算pd_step步预测后观测对应的时间步
        enemy_delay_values, ally_delay_values = self._update_delay_values(batch, pd_step, True, False)
        # 从无延迟obs理论预测后的obs
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
    
    def _build_residual_label(self, batch, last_target_seq, pd_step, t=-1):
        labels = self._build_label(batch, pd_step)
        # 上一步输入的值，可能是自回归结果，也可能是ground truth
        last_obs = last_target_seq[:, :, -1]
        res_labels = labels - last_obs
        if self.args.res_classification:
            labels_regression, labels_classification = self._just_divide(res_labels)
        else:
            # 回归预测目标是残差
            _, labels_classification = self._just_divide(labels)
            labels_regression, _ = self._just_divide(res_labels)
        if self.output_shape_classification > 0:
            labels_classification = labels_classification.long()
        if t > -1:
            labels_regression = labels_regression[:, t:t + 1, :, :]
            labels_classification = labels_classification[:, t:t + 1, :, :] if labels_classification is not None else None
        
        return labels_regression, labels_classification
    
    def _build_input_seq(self, batch, last_input_seq, predict_obs, pd_step, t=-1):
        # 计算pd_step步预测后的延迟值的onehot编码
        if self.args.one_hot_delay: 
            enemy_delay_values, ally_delay_values = self._update_delay_values(batch, pd_step, False, True)
        else:
            enemy_delay_values, ally_delay_values = self._update_delay_values(batch, pd_step, False, False)
        # 要用上一步的动作
        actions = th.roll(batch["actions_onehot"], shifts=1, dims=1)
        actions[:, 0, :, :] = th.zeros_like(actions[:, 0, :, :])
        if self.predictor.training:
            # 此时predict_obs形状为[b, t, n, -1]
            expand_obs = th.cat([predict_obs, actions, enemy_delay_values, ally_delay_values], dim=-1)
        else:
            # 推理时predict_obs形状为[b, 1, n, -1]
            expand_obs = th.cat([predict_obs, actions[:, t:t + 1], enemy_delay_values[:, t:t + 1], ally_delay_values[:, t:t + 1]], dim=-1)
        # 创建输入序列
        if last_input_seq is None:
            if self.use_history:
                input_seq = th.zeros_like(expand_obs.unsqueeze(2).expand(-1, -1, self.args.n_expand_action, -1, -1))
                if self.predictor.training:
                    for i in range(predict_obs.shape[1]):
                        history_obs = expand_obs[:, max(0, i - self.args.n_expand_action + 1):i + 1]
                        input_seq[:, i, -min(i + 1, self.args.n_expand_action):] = history_obs.clone()
                        # 没有更早的历史时用最早的历史填充
                        if i + 1 < self.args.n_expand_action:
                            input_seq[:, i, :-(i + 1)] = history_obs[:, 0:1].repeat(1, self.args.n_expand_action - i - 1, 1, 1)
                else:
                    history_obs = th.cat([batch["obs"], actions, enemy_delay_values, ally_delay_values], dim=-1)
                    input_seq[:, :, -min(t + 1, self.args.n_expand_action):] = history_obs[:, max(0, t - self.args.n_expand_action + 1):t + 1].unsqueeze(1).clone()
                    # 没有更早的历史时用最早的历史填充
                    if t + 1 < self.args.n_expand_action:
                        input_seq[:, :, :-(t + 1)] = history_obs[:, 0:1].unsqueeze(1).repeat(1, 1, self.args.n_expand_action - t - 1, 1, 1)
                    
            else:
                input_seq = expand_obs.unsqueeze(2)
        else:
            input_seq = th.cat([last_input_seq, expand_obs.unsqueeze(2)], dim=2)
        
        return input_seq
    
    def _build_target_seq(self, batch, last_target_seq, predict_obs, pd_step, t=-1):
        if last_target_seq is None:
            if self.use_history:
                target_seq = th.zeros_like(predict_obs.unsqueeze(2).expand(-1, -1, self.args.n_expand_action, -1, -1))
                if self.predictor.training:
                    for i in range(predict_obs.shape[1]):
                        history_obs = predict_obs[:, max(0, i - self.args.n_expand_action + 1):i + 1]
                        target_seq[:, i, -min(i + 1, self.args.n_expand_action):] = history_obs.clone()
                        # 没有更早的历史时用最早的历史填充
                        if i + 1 < self.args.n_expand_action:
                            target_seq[:, i, :-(i + 1)] = history_obs[:, 0:1].repeat(1, self.args.n_expand_action - i - 1, 1, 1)
                else:
                    history_obs = batch["obs"][:, max(0, t - self.args.n_expand_action + 1):t + 1]
                    target_seq[:, :, -min(t + 1, self.args.n_expand_action):] = history_obs.unsqueeze(1).clone()
                    # 没有更早的历史时用最早的历史填充
                    if t + 1 < self.args.n_expand_action:
                        target_seq[:, :, :-(t + 1)] = history_obs[:, 0:1].unsqueeze(1).repeat(1, 1, self.args.n_expand_action - t - 1, 1, 1)
            else:
                target_seq = predict_obs.unsqueeze(2).clone()
        else:
            # 训练时可以使用teacher forcing，推理时用输出结果自回归
            if self.predictor.training and random.random() < self.teacher_forcing_prob:
                labels = self._build_label(batch, pd_step)
                target_seq = th.cat([last_target_seq, labels.unsqueeze(2)], dim=2)
            else:
                target_seq = th.cat([last_target_seq, predict_obs.unsqueeze(2)], dim=2)
        
        return target_seq

    def do_predict(self, batch, t=-1):
        # 逐步预测、记录，需要额外开辟一个隐状态变量，并且需要记录每步的变化
        pd_reg_loss_avg = th.tensor(0.0, device=batch["obs"].device)
        pd_cla_loss_avg = th.tensor(0.0, device=batch["obs"].device)
        pd_obs_loss_avg = th.tensor(0.0, device=batch["obs"].device)
        predict_obs = batch["obs"] if t == -1 else batch["obs"][:, t:t + 1]
        pd_loss = th.tensor(0.0, device=batch["obs"].device)
        # 不需要预测，直接返回观测值
        if self.args.predictor_mode == "none":
            inputs_regression, inputs_classification = self._just_divide(predict_obs)
            real_obs = batch["real_obs"] if t == -1 else batch["real_obs"][:, t:t + 1]
            labels_regression, labels_classification = self._just_divide(real_obs)
            pd_reg_loss_avg = th.nn.functional.mse_loss(inputs_regression, labels_regression)
            if self.output_shape_classification > 0:
                pd_cla_loss_avg = th.nn.functional.mse_loss(inputs_classification, labels_classification)
        # 进行单步多轮预测
        elif self.args.predictor_mode == "single_step":
            input_seq = None
            target_seq = None
            for i in range(self.max_delay):
                input_seq = self._build_input_seq(batch, input_seq, predict_obs, i, t)
                # 单步预测简化延迟值，变成0&1序列
                condition_dim = self.args.n_enemies + self.args.n_allies
                input_seq[..., -condition_dim:] = (input_seq[..., -condition_dim:] > 0).to(input_seq.dtype)
                target_seq = self._build_target_seq(batch, target_seq, predict_obs, i, t)
                target_mask = th.tril(th.ones(target_seq.shape[2], target_seq.shape[2], device=target_seq.device), diagonal=0) == 0
                results, reg, cla_logits = self.forward(input_seq, target_seq, target_mask, batch, i, t)
                # 计算误差
                labels_regression, labels_classification = self._build_residual_label(batch, target_seq, i + 1, t)
                # 计算最小均方误差
                pd_reg_loss = th.nn.functional.mse_loss(reg, labels_regression)
                pd_reg_loss_avg = (pd_reg_loss_avg * i + pd_reg_loss) / (i + 1)
                if self.output_shape_classification > 0:
                    # 计算交叉熵损失
                    if self.args.res_classification:
                        labels_classification = labels_classification + 1
                    pd_cla_loss = th.nn.functional.cross_entropy(cla_logits.reshape(-1, self.predictor.class_num), labels_classification.reshape(-1))
                    pd_cla_loss_avg = (pd_cla_loss_avg * i + pd_cla_loss) / (i + 1)
                # 自回归
                predict_obs = results
        # 进行多步单轮预测
        elif self.args.predictor_mode == "multi_step":
            input_seq = None
            target_seq = None
            if self.max_delay > 0:
                input_seq = self._build_input_seq(batch, input_seq, predict_obs, 0, t)
                target_seq = self._build_target_seq(batch, target_seq, predict_obs, 0, t)
                target_mask = th.tril(th.ones(target_seq.shape[2], target_seq.shape[2], device=target_seq.device), diagonal=0) == 0
                results, reg, cla_logits = self.forward(input_seq, target_seq, target_mask, batch, 0, t)
                # 计算误差
                labels_regression, labels_classification = self._build_residual_label(batch, target_seq, self.max_delay, t)
                # 计算最小均方误差
                pd_reg_loss_avg = th.nn.functional.mse_loss(reg, labels_regression)
                if self.output_shape_classification > 0:
                    # 计算交叉熵损失
                    if self.args.res_classification:
                        labels_classification = labels_classification + 1
                    pd_cla_loss_avg = th.nn.functional.cross_entropy(cla_logits.reshape(-1, self.predictor.class_num), labels_classification.reshape(-1))
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
