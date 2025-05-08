from .nq_learner import NQLearner
from .pd_nq_learner import PDNQLearner
from .ppo_learner import PPOLearner
from .pd_ppo_learner import PDPPOLearner

REGISTRY = {}

REGISTRY["nq_learner"] = NQLearner
REGISTRY["pd_nq_learner"] = PDNQLearner
REGISTRY["ppo_learner"] = PPOLearner
REGISTRY["pd_ppo_learner"] = PDPPOLearner
