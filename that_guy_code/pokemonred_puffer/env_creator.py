import functools
import random
import uuid
from typing import Optional

import gymnasium
import pufferlib.emulation
from that_guy_code.pokemonred_puffer.environment import RedGymEnv


def env_creator(name="pokemon_red"):
    return functools.partial(make, name)


def make(name, **kwargs):
    """Pokemon Red"""
    env = RedGymEnv(kwargs)
    print("reset complete)")
    # Looks like the following will optionally create the object for you
    # Or use theo ne you pass it. I'll just construct it here.
    return pufferlib.emulation.GymnasiumPufferEnv(
        env=env, postprocessor_cls=pufferlib.emulation.BasicPostprocessor
    )
