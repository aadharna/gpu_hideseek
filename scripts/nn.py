import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# class HideAndSeekNetwork(nn.Module):
#     def __init__(self, obs_space, action_space, hidden_dim=256, num_heads=4, 
#                  entity_embedding_dim=128):
#         super(HideAndSeekNetwork, self).__init__()

#         # Extract observation and action dimensions
#         self.agent_data_dim = obs_space["agent_data"].shape[1]
#         self.relative_box_obs_dim = obs_space["relative_box_obs"].shape[1]
#         self.relative_ramp_obs_dim = obs_space["relative_ramp_obs"].shape[1]
#         self.lidar_dim = obs_space["lidar"].shape[1]
#         self.move_action_dim = action_space.nvec[0]
#         self.move_angle_dim = action_space.nvec[1]
#         self.turn_dim = action_space.nvec[2]
#         self.grab_action_dim = action_space.nvec[3]
#         self.lock_action_dim = action_space.nvec[4]


#         # Get number of entities from observation space
#         self.num_agents = obs_space["agent_data"].shape[0]
#         self.num_boxes = obs_space["relative_box_obs"].shape[0]
#         self.num_ramps = obs_space["relative_ramp_obs"].shape[0]

#         # Entity embedding layers (shared weights for each entity type)
#         self.entity_embedding = nn.Linear(self.agent_data_dim + self.lidar_dim, entity_embedding_dim)
#         self.box_embedding = nn.Linear(self.relative_box_obs_dim, entity_embedding_dim)
#         self.ramp_embedding = nn.Linear(self.relative_ramp_obs_dim, entity_embedding_dim)

#         # Self-attention layer
#         self.self_attn = nn.MultiheadAttention(entity_embedding_dim, num_heads, batch_first=True)

#         # Pooling and Dense Layers
#         self.entity_pooling = nn.AdaptiveAvgPool1d(1)
#         self.post_pooling_dense = nn.Linear(self.entity_embedding_dim, hidden_dim)
#         self.post_pooling_layernorm = nn.LayerNorm(hidden_dim)

#         # LSTM Layer
#         self.lstm = nn.LSTM(hidden_dim + self.agent_data_dim, hidden_dim, batch_first=True)
#         self.lstm_layernorm = nn.LayerNorm(hidden_dim)

#         # Separate action heads
#         self.action_head_move_amount = nn.Linear(hidden_dim, self.move_action_dim)
#         self.action_head_move_angle = nn.Linear(hidden_dim, self.move_angle_dim)
#         self.action_head_turn = nn.Linear(hidden_dim, self.turn_dim)
#         self.action_head_grab = nn.Linear(hidden_dim, self.grab_action_dim)
#         self.action_head_lock = nn.Linear(hidden_dim, self.lock_action_dim)

#         # Value head
#         self.value_head = nn.Linear(hidden_dim, 1)

#     def forward(self, obs, mask_dict=None):
#         # Embed entities
#         agent_data = obs["agent_data"]
#         lidar = obs["lidar"]
#         combined_agent_lidar = torch.cat([agent_data, lidar], dim=-1)
#         agent_embeddings = self.entity_embedding(combined_agent_lidar)

#         # Reshape box and ramp embeddings to match expected entity count
#         box_embeddings = self.box_embedding(obs["relative_box_obs"]).view(-1, self.num_boxes, self.entity_embedding_dim)
#         ramp_embeddings = self.ramp_embedding(obs["relative_ramp_obs"]).view(-1, self.num_ramps, self.entity_embedding_dim)

#         # Concatenate embeddings
#         entity_embeddings = torch.cat([
#             agent_embeddings.unsqueeze(1), box_embeddings, ramp_embeddings
#         ], dim=1)

#         # Apply masking (if provided)
#         if mask_dict is not None:
#             mask = torch.cat([
#                 torch.ones_like(agent_data[..., :1]),  # Agent always sees itself
#                 mask_dict["visible_agents_mask"],
#                 mask_dict["visible_boxes_mask"],
#                 mask_dict["visible_ramps_mask"],
#             ], dim=-1)
#             mask = mask.unsqueeze(1)  # Expand for multi-head attention (batch_first=True)

#             # Apply mask in self-attention and pooling
#             attn_output, _ = self.self_attention(entity_embeddings, entity_embeddings, entity_embeddings, 
#                                                  attn_mask=mask)
#             pooled_entities = self.entity_pooling(attn_output.permute(0, 2, 1) * mask).squeeze(-1)
#         else:
#             attn_output, _ = self.self_attention(entity_embeddings, entity_embeddings, entity_embeddings)
#             pooled_entities = self.entity_pooling(attn_output.permute(0, 2, 1)).squeeze(-1)

#         # Dense layer and LayerNorm after pooling
#         hidden = F.relu(self.post_pooling_dense(pooled_entities))
#         hidden = self.post_pooling_layernorm(hidden)

#         # Combine with agent data and global positions, pass through LSTM
#         combined = torch.cat([agent_data, hidden], dim=-1) # , obs["global_positions"]
#         lstm_out, _ = self.lstm(combined.unsqueeze(1))
#         hidden = lstm_out.squeeze(1)
#         hidden = self.lstm_layernorm(hidden)

#         # Separate action heads
#         move_amount_logits = self.action_head_move_amount(hidden)
#         move_angle_logits = self.action_head_move_angle(hidden)
#         turn_logits = self.action_head_turn(hidden)
#         grab_logits = self.action_head_grab(hidden)
#         lock_logits = self.action_head_lock(hidden)

#         # Generate action probabilities
#         move_amount_probs = F.softmax(move_amount_logits, dim=-1)
#         move_angle_probs = F.softmax(move_angle_logits, dim=-1)
#         turn_probs = F.softmax(turn_logits, dim=-1)
#         grab_probs = F.softmax(grab_logits, dim=-1)
#         lock_probs = F.softmax(lock_logits, dim=-1)

#         # Generate value estimate
#         value = self.value_head(hidden)

#         return move_amount_probs, move_angle_probs, turn_probs, grab_probs, lock_probs, value

def setup_obs(sim):
    N = sim.reset_tensor().to_torch().shape[0]

    prep_counter = sim.prep_counter_tensor().to_torch()[0:N * 5, ...]
    agent_type = sim.agent_type_tensor().to_torch()[0:N * 5, ...]
    agent_data = sim.agent_data_tensor().to_torch()[0:N * 5, ...]
    box_data = sim.box_data_tensor().to_torch()[0:N * 5, ...]
    ramp_data = sim.ramp_data_tensor().to_torch()[0:N * 5, ...]
    visible_agents_mask = sim.visible_agents_mask_tensor().to_torch()[0:N * 5, ...]
    visible_boxes_mask = sim.visible_boxes_mask_tensor().to_torch()[0:N * 5, ...]
    visible_ramps_mask = sim.visible_ramps_mask_tensor().to_torch()[0:N * 5, ...]
    lidar_tensor = sim.lidar_tensor().to_torch()[0:N * 5, ...]

    # Add in an agent ID tensor
    id_tensor = torch.arange(5).float()

    id_tensor = id_tensor.to(device=prep_counter.device)
    id_tensor = id_tensor.view(1, 5).expand(prep_counter.shape[0] // 5, 5).reshape(
        prep_counter.shape[0], 1)

    obs_tensors = [
        prep_counter,
        agent_type,
        agent_data,
        lidar_tensor,
        id_tensor,
    ]

    num_obs_features = 0
    for tensor in obs_tensors:
        num_obs_features += math.prod(tensor.shape[1:])

    ent_tensors = [
        box_data,
        ramp_data,
    ]

    num_ent_features = 0
    for tensor in ent_tensors:
        num_ent_features += math.prod(tensor.shape[1:])

    obs_tensors += ent_tensors
    
    obs_tensors += [
        visible_agents_mask,
        visible_boxes_mask,
        visible_ramps_mask,
    ]

    return obs_tensors, num_obs_features, num_ent_features

def flatten(tensor):
    return tensor.view(tensor.shape[0], -1)


def process_obs(prep_counter,
                agent_data,
                lidar,
                id_tensor,
                relative_box_obs,
                relative_ramp_obs,
                agent_type_mask,
                visible_agents_mask,
                visible_boxes_mask,
                visible_ramps_mask,
            ):
    assert(not torch.isnan(prep_counter).any())
    assert(not torch.isinf(prep_counter).any())

    assert(not torch.isnan(agent_type_mask).any())
    assert(not torch.isinf(agent_type_mask).any())

    assert(not torch.isnan(agent_data).any())
    assert(not torch.isinf(agent_data).any())

    assert(not torch.isnan(relative_box_obs).any())
    assert(not torch.isinf(relative_box_obs).any())

    assert(not torch.isnan(relative_ramp_obs).any())
    assert(not torch.isinf(relative_ramp_obs).any())

    assert(not torch.isnan(lidar).any())
    assert(not torch.isinf(lidar).any())

    assert(not torch.isnan(visible_agents_mask).any())
    assert(not torch.isinf(visible_agents_mask).any())

    assert(not torch.isnan(visible_boxes_mask).any())
    assert(not torch.isinf(visible_boxes_mask).any())

    assert(not torch.isnan(visible_ramps_mask).any())
    assert(not torch.isinf(visible_ramps_mask).any())


    common = torch.cat([
            flatten(prep_counter.float() / 200),
            flatten(agent_type_mask),
            id_tensor,
        ], dim=1)

    not_common = [
            lidar,
            agent_data,
            relative_box_obs,
            relative_ramp_obs,
            visible_agents_mask,
            visible_boxes_mask,
            visible_ramps_mask,
        ]

    return (common, not_common)


# The attention-based architecture from hide and seek
class EntitySelfAttentionNet(nn.Module):

    # num_obs_features, num_entity_features, 128, num_channels, 4
    def __init__(self, obs_features, entity_features, num_embed_channels=128, num_out_channels=256, num_heads=4):
        super(EntitySelfAttentionNet, self).__init__()
        self.num_embed_channels = num_embed_channels
        self.num_out_channels = num_out_channels
        self.num_heads = num_heads

        self.lidar_conv = nn.Conv1d(in_channels=34, out_channels=30,
                                    kernel_size=3, padding=1, # groups=50,
                                    padding_mode='circular')
        
        # Initialize the embedding layer for self
        self.self_embed = nn.Sequential(
            # remove other agents from obs for this
            nn.Linear(obs_features - 20, num_embed_channels),
            nn.LayerNorm(num_embed_channels)
        )

        # Initialize a single embedding layer for the entities
        # self.entity_embed = nn.Sequential(
        #     nn.Linear(entity_features, num_embed_channels),
        #     nn.LayerNorm(num_embed_channels)
        # )

        # embed other agents directly
        self.others_embed = nn.Sequential(
            nn.Linear(4, num_embed_channels),
            nn.LayerNorm(num_embed_channels)
        )
        
        # Initialize a single embedding layer for the entities
        self.box_entity_embed = nn.Sequential(
            nn.Linear(7, num_embed_channels),
            nn.LayerNorm(num_embed_channels)
        )

        self.ramp_entity_embed = nn.Sequential(
            nn.Linear(5, num_embed_channels),
            nn.LayerNorm(num_embed_channels)
        )

        # Attention and feed-forward layers
        self.multihead_attn = nn.MultiheadAttention(embed_dim=num_embed_channels, num_heads=num_heads)
        self.ff = nn.Sequential(
            nn.Linear(num_embed_channels, num_out_channels),
            nn.LayerNorm(num_out_channels),
            nn.LeakyReLU(),
            nn.Linear(num_out_channels, num_out_channels),
            nn.LayerNorm(num_out_channels)
        )

    def forward(self, x):

        common, (lidar, agent_data, box_data, ramp_data, visible_agents_mask, visible_boxes_mask, visible_ramps_mask) = x

        B = common.shape[0]
        N = agent_data.shape[1]
        _, NE_O, F_agents = agent_data.shape
        _, NE_B, F_box = box_data.shape
        _, NE_R, F_ramp = ramp_data.shape

        inds = torch.arange(B)
        spread_indices = torch.arange(B) % N
        # spread_indices = torch.cat([torch.arange(N) for _ in range((B // N) + 1)])[inds]
        self_observables = agent_data[inds, spread_indices, :]

        # print(B, N, self_observables.shape)
        lidar_plus_agent_data = torch.hstack((lidar, self_observables))
        
        lidar_processed = self.lidar_conv(lidar_plus_agent_data.unsqueeze(2))

        x_self = torch.cat([
            common, 
            flatten(lidar_processed),
                            ], dim=1)
        x_self = x_self.unsqueeze(-2)
        embed_self = F.leaky_relu(self.self_embed(x_self))
        
        other_agents_embedding = self.others_embed(agent_data.view(-1, F_agents)).view(B, NE_O, -1)
        box_embedding = self.box_entity_embed(box_data.view(-1, F_box)).view(B, NE_B, -1)
        ramp_embedding = self.ramp_entity_embed(ramp_data.view(-1, F_ramp)).view(B, NE_R, -1)

        masked_box_embedding = box_embedding * visible_boxes_mask
        masked_ramp_embedding = ramp_embedding * visible_ramps_mask
        masked_other_agent_embedding = other_agents_embedding * visible_agents_mask

        embedded_entities = torch.cat([embed_self, masked_other_agent_embedding, 
                                       masked_box_embedding, masked_ramp_embedding], dim=-2)
        # print(embedded_entities.shape) = torch.Size([num_worlds * (seekers + hiders), 1 + NE_O + NE_B + NE_R, 128])
        
        # need the unsqueezes for the attention calculation
        attn_output, attn_output_weights = self.multihead_attn(embedded_entities, embedded_entities, embedded_entities)
        attn_output = attn_output.mean(dim=-2)
        
        # Feedforward network
        ff_out = self.ff(attn_output)

        return ff_out


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

    from wrapper import MadronaHideAndSeekWrapper
    import numpy as np

    num_worlds = 25
    num_steps = 2500
    entities_per_world = 0
    reset_chance = 0.

    device = torch.device('cuda')


    sim = gpu_hideseek.HideAndSeekSimulator(
                        exec_mode = gpu_hideseek.madrona.ExecMode.CUDA,
                        gpu_id = 0,
                        num_worlds = num_worlds,
                        sim_flags = gpu_hideseek.SimFlags.Default,
                        rand_seed = 10,
                        min_hiders = 3,
                        max_hiders = 3,
                        min_seekers = 2,
                        max_seekers = 2,
                        num_pbt_policies = 0,
        )
    sim.init()

    env = MadronaHideAndSeekWrapper(sim)
    obs, _ = env.reset()
    num_obs_features = env.num_obs_features
    num_entity_features = env.num_ent_features
    # obs_tensors, num_obs_features, num_entity_features = setup_obs(sim)

    network = EntitySelfAttentionNet(num_obs_features, num_entity_features).to('cuda:0')

    x = process_obs(**obs)
    
    print(network)

    out = network(x)
    # move_amount, move_angle, turn, grab, lock, value

    action = {'move_amount': 1, 'move_angle': 8, 'turn': 7, 'grab': 0, 'lock': 0}
    ns, rew, done, tr, i = env.step(action)
    _ = 0

