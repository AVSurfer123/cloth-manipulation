from os.path import join
import importlib
import json
import collections

import torch
import numpy as np

from rlpyt.envs.dm_control_env import DMControlEnv
from rlpyt.samplers.serial.sampler import SerialSampler


def load_agent(snapshot_dir, max_q_eval_mode):
    snapshot_file = join(snapshot_dir, 'params.pkl')
    config_file = join(snapshot_dir, 'params.json')

    params = torch.load(snapshot_file, map_location='cpu')
    with open(config_file, 'r') as f:
        config = json.load(f)

    itr, cum_steps = params['itr'], params['cum_steps']
    print(f'Loading experiment at itr {itr}, cum_steps {cum_steps}')

    agent_state_dict = params['agent_state_dict']

    sac_agent_module = 'rlpyt.agents.qpg.{}'.format(config['sac_agent_module'])
    sac_agent_module = importlib.import_module(sac_agent_module)
    SacAgent = sac_agent_module.SacAgent

    agent = SacAgent(max_q_eval_mode=max_q_eval_mode, **config["agent"])
    agent.load_state_dict(agent_state_dict)

    agent.to_device(cuda_idx=0)
    agent.eval_mode(0)
    agent.get_action = get_action.__get__(agent)
    return agent

def get_action(agent, image, location):
    """Gets the action from the SAC model given the image observation and pick location.

    Args:
        image: 64x64x3 np array
        location: flattened (4,) array of 2 concatenated pick locations between 0 and 63
    Returns:
        action: (6,) array of deltas for both pick points between -1 and 1
    """
    obs = collections.namedtuple('Observation', ['location, pixels'])
    obs.pixels = image
    obs.location = np.tile(location, 50).reshape(-1).astype('float32') / 63.
    action, log_pi, dist_info = agent.pi(obs, None, None)
    return action

if __name__ == '__main__':
    main()
