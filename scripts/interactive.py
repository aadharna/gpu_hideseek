import madrona_python
import gpu_hideseek_python
import torch
import torchvision
import sys
import termios
import tty

def get_single_char():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    tty.setraw(sys.stdin.fileno())
    
    ch = sys.stdin.read(1)
    
    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    return ch

class Action:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.r = 0
        self.g = 0
        self.l = 0
        self.reset = 0

def get_keyboard_action():
    while True:
        key_action = get_single_char()

        result = Action()

        if key_action == 'w':
            result.y = 5
        elif key_action == 'a':
            result.x = -5
        elif key_action == 'd':
            result.x = 5
        elif key_action == 's':
            result.y = -5
        elif key_action == 'q':
            result.r = 1
        elif key_action == 'e':
            result.r = -1
        elif key_action == 'g':
            result.g = 1
        elif key_action == 'l':
            result.l = 1
        elif key_action == 'x':
            result.reset = -1
        elif key_action == '1':
            result.reset = 1
        elif key_action == '2':
            result.reset = 2
        elif key_action == '3':
            result.reset = 3
        elif key_action == '4':
            result.reset = 4
        elif key_action == '5':
            result.reset = 5
        elif key_action == '6':
            result.reset = 6
        elif key_action == '7':
            result.reset = 7
        elif key_action == '8':
            result.reset = 8
        elif key_action == ' ':
            pass
        else:
            continue

        return result

exec_mode_str = sys.argv[1]

if exec_mode_str == "CUDA":
    exec_mode = madrona_python.ExecMode.CUDA
elif exec_mode_str == "CPU":
    exec_mode = madrona_python.ExecMode.CPU
else:
    print("interactive.py (CUDA | CPU) [out.png]")
    sys.exit(1)

sim = gpu_hideseek_python.HideAndSeekSimulator(
        exec_mode = exec_mode,
        gpu_id = 0,
        num_worlds = 1,
        render_width = 1536,
        render_height = 1024,
        enable_batch_render = True,
        debug_compile = False,
)

actions = sim.action_tensor().to_torch()
rewards = sim.reward_tensor().to_torch()
agent_valid_masks = sim.agent_mask_tensor().to_torch()
agent_visibility_masks = sim.visible_agents_mask_tensor().to_torch()
box_visibility_masks = sim.visible_boxes_mask_tensor().to_torch()
ramp_visibility_masks = sim.visible_ramps_mask_tensor().to_torch()
agent_data = sim.agent_data_tensor().to_torch()
resets = sim.reset_tensor().to_torch()
prep_counter = sim.prep_counter_tensor().to_torch()
dones = sim.done_tensor().to_torch()
global_pos = sim.global_positions_tensor().to_torch()
lidar = sim.lidar_tensor().to_torch()

if len(sys.argv) > 2:
    rgb_observations = sim.rgb_tensor().to_torch()

print(actions.shape, actions.dtype)
print(resets.shape, resets.dtype)

print(agent_visibility_masks.shape)
print(agent_data.shape)

resets[..., 0] = 1
resets[..., 1] = 2
resets[..., 2] = 2
actions[...] = 0
sim.step()

while True:
    print("\nObs:")
    print(prep_counter[0])

    if len(sys.argv) > 2:
        torchvision.utils.save_image((rgb_observations[0][0].float() / 255).permute(2, 0, 1), sys.argv[2])

    #print(rewards[0:16 * 4])
    #print(rewards[0][:4] * agent_valid_masks[0][:4])
    #print(agent_visibility_masks[0][:4] * agent_valid_masks[0][:4].unsqueeze(dim = 2))
    #print(box_visibility_masks[0][:4] * agent_valid_masks[0][:4].unsqueeze(dim = 2))
    #print(ramp_visibility_masks[0][:4] * agent_valid_masks[0][:4].unsqueeze(dim = 2))

    #print(global_pos)
    #print(lidar[0][0])

    action = get_keyboard_action()
    #action = Action()
    
    if action.reset < 0:
        break

    resets[..., 0] = action.reset
    actions[0][0] = action.x
    actions[0][1] = action.y
    actions[0][2] = action.r
    actions[0][3] = action.g
    actions[0][4] = action.l

    sim.step()
    print("Stepping:")

    print(dones[0])
    print(rewards[0])

del sim
