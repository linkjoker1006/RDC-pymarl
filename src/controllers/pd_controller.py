import numpy as np
import torch as th
import time

from .basic_controller import BasicMAC


# This multi-agent controller shares parameters between agents
class PDMAC(BasicMAC):
    def __init__(self, scheme, groups, args):
        super(PDMAC, self).__init__(scheme, groups, args)
        self.cheating_prob = args.cheating_start_value
        # 由learner控制，当learner决定作弊时，将cheating设置为True，这样可以防止mac和traget_mac不匹配
        self.cheating = False

    def set_predictor(self, predictor):
        self.predictor = predictor
    
    def forward(self, ep_batch, t, predict_obs=None, test_mode=False):
        # 如果没有外部提供predict_obs，则在内部调用预测器，只对一个时间步进行预测；外部提供的predict_obs是多个时间步的
        if predict_obs is None:
            # 冻结预测器参数
            self.predictor.freeze()
            # start_time = time.time()
            predict_obs, pd_reg_loss_avg, pd_cla_loss_avg, pd_obs_loss_avg = self.predictor.do_predict(ep_batch, t)
            # end_time = time.time()
            # print("predict_obs time: {}".format(end_time - start_time))
            # 使用真实观测代替预测结果
            if self.cheating or np.random.random() < self.cheating_prob:
                predict_obs = ep_batch["real_obs"][:, t:t + 1]
        agent_inputs = self._build_inputs(ep_batch, predict_obs, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)
        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":
            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                agent_outs = agent_outs.reshape(ep_batch.batch_size * self.n_agents, -1)
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)
    
    def _build_inputs(self, batch, predict_obs, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        # 如果传入的是多个时间步的预测结果，则只取当前时间步的结果
        if predict_obs.shape[1] > 1:
            predict_obs = predict_obs[:, t]
        inputs = []
        inputs.append(predict_obs)  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))
        inputs = th.cat([x.reshape(bs, self.n_agents, -1) for x in inputs], dim=-1)

        return inputs
    
    def cuda(self):
        self.predictor.cuda()
        self.agent.cuda()
    
    def cpu(self):
        self.predictor.cpu()
        self.agent.cpu()

    def set_train_mode(self):
        self.agent.train()
    
    def set_evaluation_mode(self):
        self.agent.eval()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))
    
