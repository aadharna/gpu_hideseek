import gpu_hideseek
import torch
import sys
import time
import PIL
import PIL.Image
torch.manual_seed(0)
import random
random.seed(0)

num_worlds = 16000
num_steps = 1920
entities_per_world = 0
reset_chance = 0.


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

actions = sim.action_tensor().to_torch()
resets = sim.reset_tensor().to_torch()
print(actions.shape, actions.dtype)
print(resets.shape, resets.dtype)

reset_no = torch.zeros_like(resets[:, 0], dtype=torch.int32)
reset_yes = torch.ones_like(resets[:, 0], dtype=torch.int32)
reset_rand = torch.zeros_like(resets[:, 0], dtype=torch.float32)


#def dump_obs(dump_dir, step_idx):
#    N = rgb_observations.shape[0]
#    A = rgb_observations.shape[1]
#
#    num_wide = min(64, N * A)
#
#    reshaped = rgb_observations.reshape(N * A // num_wide, num_wide, *rgb_observations.shape[2:])
#    grid = reshaped.permute(0, 2, 1, 3, 4)
#
#    grid = grid.reshape(N * A // num_wide * render_height, num_wide * render_width, 4)
#    grid = grid.type(torch.uint8).cpu().numpy()
#
#    img = PIL.Image.fromarray(grid)
#    img.save(f"{dump_dir}/{step_idx}.png", format="PNG")

move_action_slice = actions[..., 0:2]
move_action_slice.copy_(torch.zeros_like(move_action_slice))

for i in range(5):
    sim.step()

#resets[:, 0] = 1

start = time.time()

for i in range(num_steps):
    # sim.step()
    sim.simulate()
    sim.resetandupdate()

    #torch.rand(reset_rand.shape, out=reset_rand)

    #reset_cond = torch.where(reset_rand < reset_chance, reset_yes, reset_no)
    #resets[:, 0].copy_(reset_cond)

    torch.randint(-5, 5, move_action_slice.shape,
                  out=move_action_slice,
                  dtype=torch.int32, device=torch.device('cuda'))

    #if len(sys.argv) > 5:
    #    dump_obs(sys.argv[5], i)

end = time.time()

duration = end - start
print(duration, " total runtime")
print(num_worlds * num_steps, " total frames")
print(num_worlds * num_steps / duration, " SPS")

del sim
