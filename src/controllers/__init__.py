from .hpn_controller import HPNMAC
from .basic_controller import BasicMAC
from .n_controller import NMAC
from .pd_controller import PDMAC


REGISTRY = {}

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["n_mac"] = NMAC
REGISTRY["hpn_mac"] = HPNMAC
REGISTRY["pd_mac"] = PDMAC
