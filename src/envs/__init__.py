from functools import partial
import sys
import os

from .multiagentenv import MultiAgentEnv

try:
    smac = True
    from .smac_v1.StarCraft2EnvWrapper import StarCraft2EnvWrapper
except Exception as e:
    print(e)
    smac = False

try:
    smacv2 = True
    from .smac_v2.StarCraft2Env2Wrapper import StarCraft2Env2Wrapper
except Exception as e:
    print(e)
    smacv2 = False

try:
    mpe = True
    from .mpe.GymmaEnvWrapper import GymmaEnvWrapper
except Exception as e:
    print(e)
    mpe = False

def __check_and_prepare_smac_kwargs(kwargs):
    assert "common_reward" in kwargs and "reward_scalarisation" in kwargs
    assert kwargs[
        "common_reward"
    ], "SMAC only supports common reward. Please set `common_reward=True` or choose a different environment that supports general sum rewards."
    del kwargs["common_reward"]
    del kwargs["reward_scalarisation"]
    assert "map_name" in kwargs, "Please specify the map_name in the env_args"
    return kwargs

def smac_fn(env, **kwargs) -> MultiAgentEnv:
    kwargs = __check_and_prepare_smac_kwargs(kwargs)
    return env(**kwargs)

def gymma_fn(env, **kwargs) -> MultiAgentEnv:
    assert "common_reward" in kwargs and "reward_scalarisation" in kwargs
    return env(**kwargs)


REGISTRY = {}

if smac:
    REGISTRY["sc2"] = partial(smac_fn, env=StarCraft2EnvWrapper)
    if sys.platform == "linux":
        os.environ.setdefault("SC2PATH",
                              os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))
else:
    print("SMAC V1 is not supported...")

if smacv2:
    REGISTRY["sc2_v2"] = partial(smac_fn, env=StarCraft2Env2Wrapper)
    if sys.platform == "linux":
        os.environ.setdefault("SC2PATH",
                              os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))
else:
    print("SMAC V2 is not supported...")

if mpe:
    REGISTRY["gymma"] = partial(gymma_fn, env=GymmaEnvWrapper)
else:
    print("MPE is not supported...")

print("Supported environments:", REGISTRY)
