import math
import torch
import numpy as np

import gymnasium as gym
from gymnasium.spaces import Dict, Box, MultiDiscrete


class MadronaHideAndSeekWrapper: #gym.Wrapper):
    def __init__(self, sim, nSeekers=3, nHiders=2):
        # super(MadronaHideAndSeekWrapper, self).__init__(sim)
        self.sim = sim

        self.N = sim.agent_data_tensor().to_torch().shape[0] # = num_worlds * (seekers + hiders)
        self.nSeekers = nSeekers
        self.nHiders = nHiders
        self.unknown_dim_size = 5 # perhaps maxSeekers + maxHiders?

        # Define observation space components (using provided shapes)
        # [num_worlds * (seekers + hider), [?, ?, ?, ?, ?], [pos.x, pos.y, vel.x, vel.y]]
        prep_counter = sim.prep_counter_tensor().to_torch()[0:self.N * self.unknown_dim_size, ...]
        agent_type = sim.agent_type_tensor().to_torch()[0:self.N * self.unknown_dim_size, ...]
        agent_data = sim.agent_data_tensor().to_torch()[0:self.N * self.unknown_dim_size, ...]
        box_data = sim.box_data_tensor().to_torch()[0:self.N * self.unknown_dim_size, ...]
        ramp_data = sim.ramp_data_tensor().to_torch()[0:self.N * self.unknown_dim_size, ...]
        visible_agents_mask = sim.visible_agents_mask_tensor().to_torch()[0:self.N * self.unknown_dim_size, ...]
        visible_boxes_mask = sim.visible_boxes_mask_tensor().to_torch()[0:self.N * self.unknown_dim_size, ...]
        visible_ramps_mask = sim.visible_ramps_mask_tensor().to_torch()[0:self.N * self.unknown_dim_size, ...]
        lidar_tensor = sim.lidar_tensor().to_torch()[0:self.N * self.unknown_dim_size, ...]
        done_mask = sim.done_tensor().to_torch()[0:self.N * self.unknown_dim_size, ...]

        # Add in an agent ID tensor
        id_tensor = self.get_id_tensor()

        agent_data_space = Box(low=-np.inf, high=np.inf, shape=agent_data.shape[1:], dtype=np.float32)
        # self.relative_agent_obs_space = Box(low=-np.inf, high=np.inf, shape=sim.agent_data_tensor().to_torch().shape[1:], dtype=np.float32)
        relative_box_obs_space = Box(low=-np.inf, high=np.inf, shape=box_data.shape[1:], dtype=np.float32)
        relative_ramp_obs_space = Box(low=-np.inf, high=np.inf, shape=ramp_data.shape[1:], dtype=np.float32)
        visible_agents_mask_space = Box(low=0, high=1, shape=visible_agents_mask.shape[1:], dtype=np.float32)
        visible_boxes_mask_space = Box(low=0, high=1, shape=visible_boxes_mask.shape[1:], dtype=np.float32)
        visible_ramps_mask_space = Box(low=0, high=1, shape=visible_ramps_mask.shape[1:], dtype=np.float32)
        lidar_space = Box(low=0, high=np.inf, shape=lidar_tensor.shape[1:], dtype=np.float32)
        prep_counter_space = Box(low=0, high=np.inf, shape=prep_counter.shape[1:], dtype=np.int32)
        agent_type_mask = Box(low=0, high=1, shape=agent_type.shape[1:], dtype=np.int32)
        # done_space = Box(low=0, high=1, shape=done_mask.shape[1:], dtype=np.int32)
        # agent_mask_space = Box(low=0, high=1, shape=sim.agent_mask_tensor().to_torch().shape[1:], dtype=np.float32)
        # global_positions_space = Box(low=-np.inf, high=np.inf, shape=sim.global_positions_tensor().to_torch().shape[1:], dtype=np.float32)
        
        id_tensor_shape = Box(low=0, high=self.nHiders + self.nSeekers, 
                              shape=id_tensor.shape[1:], dtype=np.int32)

        
        obs_tensors = [
                prep_counter,
                agent_type,
                agent_data,
                lidar_tensor,
                id_tensor,
            ]

        self.num_obs_features = 0 # 53
        for tensor in obs_tensors:
            self.num_obs_features += math.prod(tensor.shape[1:])

        ent_tensors = [
            box_data,
            ramp_data,
        ]

        self.num_ent_features = 0 # 73
        for tensor in ent_tensors:
            self.num_ent_features += math.prod(tensor.shape[1:])

        obs_tensors += ent_tensors
        
        obs_tensors += [
            visible_agents_mask,
            visible_boxes_mask,
            visible_ramps_mask,
        ]
        
        # Create dictionary observation space
        self.observation_space = Dict({
            # observation data
            "agent_data": agent_data_space,
            "lidar": lidar_space,
            "prep_counter": prep_counter_space,
            'agent_type_mask': agent_type_mask,
            'id_tensor': id_tensor_shape,
            
            # entity data
            "relative_box_obs": relative_box_obs_space,
            "relative_ramp_obs": relative_ramp_obs_space,
            
            # visibility masks from env
            "visible_agents_mask": visible_agents_mask_space,
            "visible_boxes_mask": visible_boxes_mask_space,
            "visible_ramps_mask": visible_ramps_mask_space,

        })

        self.action_space = MultiDiscrete([11, 11, 11, 2, 2])
        # action_space[0] = [-5, 5] move amount distritized
        # action_space[1] = [-5, 5] move angle distritized
        # action_space[2] = [-5, 5] rotation amount distritized
        # action_space[3] = [0, 1]  grab yes/no
        # action_space[4] = [0, 1]  lock yes/no

    def get_id_tensor(self):
        prep_counter = self.sim.prep_counter_tensor().to_torch()[0:self.N * self.unknown_dim_size, ...]
        id_tensor = torch.arange(self.unknown_dim_size).float()

        id_tensor = id_tensor.to(device=prep_counter.device)
        id_tensor = id_tensor.view(1, self.unknown_dim_size).expand(prep_counter.shape[0] // self.unknown_dim_size, 
                                                                    self.unknown_dim_size).reshape(
                                                                               prep_counter.shape[0], 1)
        return id_tensor
    
    def flatten(self, tensor, keepdim=-1):
        return tensor.view(tensor.shape[0], keepdim)

    def get_obs(self):
        obs = {
            "agent_data": self.sim.agent_data_tensor().to_torch(),
            "relative_box_obs": self.sim.box_data_tensor().to_torch(),
            "relative_ramp_obs": self.sim.ramp_data_tensor().to_torch(),
            "visible_agents_mask": self.sim.visible_agents_mask_tensor().to_torch(),
            "visible_boxes_mask": self.sim.visible_boxes_mask_tensor().to_torch(),
            "visible_ramps_mask": self.sim.visible_ramps_mask_tensor().to_torch(),
            "lidar": self.sim.lidar_tensor().to_torch(),
            "prep_counter": self.sim.prep_counter_tensor().to_torch(),
            'agent_type_mask': self.sim.agent_type_tensor().to_torch(),
            'id_tensor': self.get_id_tensor(),
        }
        
        return obs
    
    def reset(self, **kwargs):
        # Reset the simulator is usually done inside the sim.step function
        # however, with the split task graph, we can manually reset it as well.
        # self.sim.reset_and_update() # reset ALL of the batched worlds
        
        # Collect observations and return as a dictionary
        obs = self.get_obs()
        return obs, {}


    def step(self, action_dict):
        # Extract actions from the dictionary
        move_amount = action_dict["move_amount"]
        move_angle = action_dict["move_angle"]
        turn = action_dict["turn"]
        grab = action_dict["grab"]
        lock = action_dict["lock"]

        # Get the action tensor from the simulator
        action_tensor = self.sim.action_tensor().to_torch()

        # Fill in the action tensor with the extracted actions
        # move amount, angle and turn are all in [-5, 5] and 
        #  the logits are in [0, 10] so we need to center those properly
        action_tensor[..., 0] = move_amount - 5
        action_tensor[..., 1] = move_angle - 5
        action_tensor[..., 2] = turn - 5
        action_tensor[..., 3] = grab
        action_tensor[..., 4] = lock
        
        # copy the action tensor to the simulator
        self.sim.action_tensor().copy_(action_tensor)

        # Apply the modified action tensor to the simulator
        # self.sim.setAction(action_tensor)
        self.sim.step()
        # self.sim.prestep()
        # self.sim.poststep()

        # Collect observations, rewards, dones, and info
        obs = self.get_obs()
        reward = self.sim.reward_tensor().to_torch()
        done = self.sim.done_tensor().to_torch()
        info = {}  # Add any additional info if needed

        return obs, reward, done, done, info
    
    def __del__(self):
        del self.sim
    
class MadronaHideAndSeekWrapperSplitTaskGraph(MadronaHideAndSeekWrapper):
    def __init__(self, sim, nSeekers=3, nHiders=2):
        super().__init__(sim, nSeekers, nHiders)
        
        # intial reward function uses the c++ hide and seek calculated reward
        def reward_fn(agent_data, relative_box_obs, relative_ramp_obs, visible_agents_mask, 
                      visible_boxes_mask, visible_ramps_mask, lidar, prep_counter, agent_type_mask, id_tensor):
                        
            return self.sim.reward_tensor().to_torch()
        
        initial_rf = reward_fn

        self.rfs = [initial_rf]
        self.sampled_reward_fn_id = 0
        self.dynamic_reward_fn = self.rfs[self.sampled_reward_fn_id]

    def step(self, action_dict):
        # Extract actions from the dictionary
        move_amount = action_dict["move_amount"]
        move_angle = action_dict["move_angle"]
        turn = action_dict["turn"]
        grab = action_dict["grab"]
        lock = action_dict["lock"]

        # Get the action tensor from the simulator
        action_tensor = self.sim.action_tensor().to_torch()

        # Fill in the action tensor with the extracted actions
        # move amount, angle and turn are all in [-5, 5] and 
        #  the logits are in [0, 10] so we need to center those properly
        action_tensor[..., 0] = move_amount - 5
        action_tensor[..., 1] = move_angle - 5
        action_tensor[..., 2] = turn - 5
        action_tensor[..., 3] = grab
        action_tensor[..., 4] = lock
        
        # copy the action tensor to the simulator
        action_tensor.copy_(action_tensor)

        # Apply the modified action tensor to the sim
        self.sim.simulate()
        
        # get the state of the environment post action 
        # (env is updated at the end of simulate, but envs are not reset yet)
        obs = self.get_obs()
        
        # calculate the reward using state + action
        reward = self.dynamic_reward_fn(**obs)
        
        # update the environment if finished and collect next states
        self.sim.reset_and_update()

        # Collect observations, rewards, dones, and info
        obs = self.get_obs()
        done = self.sim.done_tensor().to_torch()
        info = {}
        
        return obs, reward, done, done, info

    def set_reward_function(self, reward_fn_string, sample=False):
        # parse the generated reward function into a callable 
        # make sure that the function signature matches what we expect:
        # def reward_fn(agent_data, relative_box_obs, relative_ramp_obs, visible_agents_mask, 
        #               visible_boxes_mask, visible_ramps_mask, lidar, prep_counter, agent_type_mask, id_tensor): -> reward
        self.rfs.append(reward_fn_string)
        function = eval(reward_fn_string, globals(), locals())
        self.dynamic_reward_fn = function


if __name__ == '__main__':
    import gpu_hideseek
    import torch
    import numpy as np
    import sys
    import time
    import PIL
    import PIL.Image
    torch.manual_seed(0)
    import random
    random.seed(0)

    num_worlds = 2
    num_steps = 2500
    entities_per_world = 2
    reset_chance = 0.


    sim = gpu_hideseek.HideAndSeekSimulator(
			exec_mode = gpu_hideseek.madrona.ExecMode.CUDA,
			gpu_id = 0,
			num_worlds = num_worlds,
			sim_flags = gpu_hideseek.SimFlags.Default,
			rand_seed = 10,
			min_hiders = 2,
			max_hiders = 2,
			min_seekers = 3,
			max_seekers = 3,
            num_pbt_policies = 0,
	)
    sim.init()

    env = MadronaHideAndSeekWrapper(sim)
    print(env.observation_space)
    obs = env.reset()
    action = {'move_amount': 1, 'move_angle': 8, 'turn': 7, 'grab': 0, 'lock': 0}
    ns, rew, done, tr, i = env.step(action)

    print(ns.keys())

    del env