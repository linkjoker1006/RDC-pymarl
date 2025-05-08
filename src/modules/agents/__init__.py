from .hpns_rnn_agent import HPNS_RNNAgent
from .n_rnn_agent import NRNNAgent
from .n_mlp_agent import NMLPAgent


REGISTRY = {}

REGISTRY["n_mlp"] = NMLPAgent
REGISTRY["n_rnn"] = NRNNAgent
REGISTRY["hpns_rnn"] = HPNS_RNNAgent
