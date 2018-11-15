from threading import Timer
import numpy as np
import torch
import os
import cv2
from mss import mss
import win32.win32gui as wgui
import re
import win32process
import time
from ctypes import *
import ctypes.wintypes as wtypes
import directkeys as dk
import keyboard
import torch.nn.functional as F
# In-project imports
from models import Model
from memory_operations import read_memory, write_memory
from replay_buffer import ReplayBuffer

# Settings
# - Window settings
SCALE = 1.25  # Windows scaling for high DPI displays, need to enter this manually
monitor = {'top': 0, 'left': 0, 'width': 800, 'height': 480}  # Default recording window
MONITOR_WINDOW_MARGIN = [35, 2, 2, 2]
MONITOR_GAME_AREA_RATIO = [0.03, 0.05, 390 / 650, 450 / 480]
# - Recording settings
FRAMES_SKIP = 1

# - Model settings
FRAMES_FEED = 4  # How many frames the model should take as input
DEATH_PENALTY = 5000
KEY_DURATION_SECONDS = 0

# - Training parameters
BATCH_SIZE = 32
LEARNING_RATE = 0.03
GAMMA = 0.99
GRAD_CLIP = 10
TARGET_MODEL_UPDATE_FREQ = 1500
TRAIN_FREQ = 64

# - Debug settings
DEBUG = True
DEBUG_MEMORY = False
DEBUG_CONTROL = False
SHOW_DATAPOINTS = False

# - Misc settings and constants
SCORE_ADDR = c_void_p(0x0049E440)
LIVES_ADDR = c_void_p(0x0049E484)
TH_PATH = "D:\\Games\\Steam\\steamapps\\common\\th16tr\\th16.exe"
PROCESS_ALL_ACCESS = 0x1F0FFF

# - Global variables
pid = 0  # Game process id
active = True
game_paused = False
action_stack = []  # Holds the current actions
DEVICE = torch.device("cpu")


class ExplorationScheduler:
    def __init__(self, val=1, decay: float = 0.90, minval: float = 0.02):
        self.val = val
        self.decay = decay
        self.min = minval

    def value(self, t: int) -> float:
        return max(self.min, self.val * self.decay ** t)


# https://ai.intel.com/demystifying-deep-reinforcement-learning/
def main_loop(handle, possible_actions: list, model: Model, target_model: Model):
    exp_schedule = ExplorationScheduler()
    target_model.load_state_dict(model.state_dict())
    optimizer = torch.optim.RMSprop(model.parameters())
    with mss() as sct:
        counter = 0
        frame_counter = 0
        frame_skip_counter = 0
        score = 0
        lives = 3
        frame_times = [0, 0, 0, 0]
        replay_buffer = ReplayBuffer(200, (3 * FRAMES_FEED, monitor['height'], monitor['width']), FRAMES_FEED)
        t = 0
        action = 0
        while True:
            if not active:
                time.sleep(0.5)  # Wait some time and check if recording should be resumed.
                continue

            startMillis = time.time()  # Time
            # Grab frames
            frame, frame_cv2 = grab_screen(monitor, sct)

            # Show frame
            if DEBUG:
                cv2.imshow('window1', frame_cv2)
            # Check if frame will be skipped. Not skipped if counter is 0
            if frame_skip_counter == 0:
                reward, score, lives = get_reward(handle, lives, score)

                print(action, reward)
                if replay_buffer.waiting_for_effect:
                    replay_buffer.add_effects(action, reward)
                replay_buffer.push_frame(frame)
                if replay_buffer.buffer_init() and np.random.random() > exp_schedule.value(t):
                    action = choose_action(replay_buffer.encode_last_frame(), model)
                else:
                    action = np.random.randint(0, len(possible_actions) - 1)

                execute_actions([possible_actions[int(action)], dk.SCANCODES["z"]])

                # Logic to deal with a ready datapoint
                if replay_buffer.can_sample(BATCH_SIZE) and t % TRAIN_FREQ == 0:
                    pause_game()
                    optimize_model(model, target_model, replay_buffer, optimizer, num_actions=len(possible_actions))

                    # Copy model weights to target
                    if t % TARGET_MODEL_UPDATE_FREQ == 0:
                        target_model.load_state_dict(model.state_dict())
                        target_model.eval()
                    pause_game()

            frame_skip_counter += 1
            frame_skip_counter = frame_skip_counter % FRAMES_SKIP

            # Frame timings and other utility
            endMillis = time.time()
            frame_time = endMillis - startMillis
            frame_times[counter % 4] = frame_time
            t += 1
            #if counter % 4 == 0:
            #    print("frame time: %s" % (np.mean(frame_times)))
            counter += 1
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break


def get_reward(handle, lives, score):
    """Calculates reward based on difference of current score and lives"""
    new_score = get_score(handle)
    if new_score is None:
        new_score = 0
    reward = new_score - score
    new_lives = get_lives(handle)
    if new_lives is None:
        new_lives = 0
    if new_lives < lives:
        # Give a score penalty for dying
        reward = reward - DEATH_PENALTY
    if new_lives == 1:
        # Add a life to extend play time
        set_lives(handle, 2)
    return reward, new_score, new_lives


def one_hot(indexes: torch.LongTensor, depth: int):
    mask = torch.zeros(indexes.shape[0], depth).to(DEVICE)
    print(indexes.shape)
    print(mask.shape)
    print(indexes.unsqueeze(0).shape)
    #ones = torch.ones(indexes.shape[0], depth).to(DEVICE)

    mask.scatter_(1, indexes.unsqueeze(1), 1.)
    return mask


def optimize_model(policy_net: Model, target_net: Model, replay_buffer: ReplayBuffer, optimizer, num_actions):
    """Takes a random batch from :experience_memory and trains the policy_net on it for one iteration"""
    # Sample training data
    obs_batch, actions_batch, rewards_batch, obs_next_batch = replay_buffer.sample(BATCH_SIZE)
    print(type(actions_batch))
    actions_batch = torch.from_numpy(actions_batch).to(DEVICE)
    print(type(actions_batch))
    action_mask = one_hot(actions_batch, num_actions)
    obs_batch_torch = torch.from_numpy(obs_batch).to(DEVICE)
    obs_next_batch_torch = torch.from_numpy(obs_next_batch).to(DEVICE)

    q_t_batch = policy_net(obs_batch_torch)
    q_t_ac = torch.sum(q_t_batch * action_mask, dim=1)

    with torch.no_grad():
        q_tp1 = policy_net(obs_next_batch_torch)
        q_tp1_biggest, q_tp1_actions = q_tp1.max(1)
        tp1_action_mask = one_hot(q_tp1_actions, num_actions)

        q_tp1_target = target_net(obs_next_batch_torch)
        q_target = rewards_batch + GAMMA * torch.sum(tp1_action_mask * q_tp1_target, dim=1)

    loss = F.smooth_l1_loss(q_t_ac, q_target.float().to(DEVICE))
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), GRAD_CLIP)
    optimizer.step()


def choose_action(obs, model: Model):
    if DEBUG and DEBUG_CONTROL:
        print("choose_action")
    obs_torch = torch.from_numpy(obs).to(DEVICE)
    action_scores = model.forward(obs_torch.unsqueeze_(0))
    return np.argmax(action_scores.detach().cpu().numpy())


def grab_screen(monitor, recorder):
    """Makes a screen shot of the screen area defined in @monitor using @recorder
    :returns @frame and opencv compatible @frame_cv2"""
    im = recorder.grab(monitor=monitor)
    im = np.array(im, dtype=np.uint8)
    frame_cv2 = np.flip(im[:, :, :3], 2, ).astype(np.uint8)
    frame = np.moveaxis(frame_cv2, 2, 0, ).astype(np.float32, copy=False) / 255
    return frame.astype(np.float32, copy=False), frame_cv2


def execute_actions(actions):
    global action_stack
    if DEBUG and DEBUG_CONTROL:
        print("execute_actions")
    if type(actions) is list:

        # Release any keys not in the new action command
        for action in action_stack:
            if action not in actions:
                release(action)

        # Add any keys that weren't in the previous action stack
        for action in actions:
            if action not in action_stack:
                press(action, 0)
    else:
        press(actions, 0)


def release(key):
    """Releases the given key"""
    if DEBUG and DEBUG_CONTROL:
        print("release", key)
    global action_stack
    if key in action_stack:
        del action_stack[action_stack.index(key)]
    dk.ReleaseKey(key)


def press(key, time=0):
    """Presses the key :key for :time seconds. :time 0 is infinite"""
    if key is None:
        return
    if DEBUG:
        print("press", key)
    if key not in action_stack:
        action_stack.append(key)
    dk.PressKey(key)
    if time:
        Timer(time, release, key)


def get_score(handle):
    """Reads score from program memory"""
    if DEBUG and DEBUG_MEMORY:
        print("get_score")
    buffer = wtypes.LPVOID(0)
    buffer, bytes_read = read_memory(handle, SCORE_ADDR, buffer, sizeof(c_uint32))
    return buffer.value


def get_lives(handle):
    """Reads number of lives from program memory"""
    if DEBUG and DEBUG_MEMORY:
        print("get_lives")
    buffer = wtypes.LPVOID(0)
    buffer, bytes_read = read_memory(handle, LIVES_ADDR, buffer, sizeof(c_uint32))
    return buffer.value


def set_lives(handle, lives):
    if DEBUG and DEBUG_MEMORY:
        print("set_lives")
    return write_memory(handle, LIVES_ADDR, lives, sizeof(c_int32))


def set_score(handle, score):
    return write_memory(handle, SCORE_ADDR, score, sizeof(c_int32))


def fit_window_callback(window, arg):
    """Checks if the given window handle matches the program we want,
     and saves window data to the global monitor variable"""
    reg = re.compile('.*Hidden Star in Four Seasons.*')
    if reg.match(wgui.GetWindowText(window)):
        global monitor
        rect = wgui.GetWindowRect(window)
        monitor['top'] = int(rect[1] * SCALE) + MONITOR_WINDOW_MARGIN[0]
        monitor['left'] = int(rect[0] * SCALE) + MONITOR_WINDOW_MARGIN[1]
        monitor['width'] = int((rect[2] - rect[0]) * SCALE) - MONITOR_WINDOW_MARGIN[2] - MONITOR_WINDOW_MARGIN[1]
        monitor['height'] = int((rect[3] - rect[1]) * SCALE) - MONITOR_WINDOW_MARGIN[3] - MONITOR_WINDOW_MARGIN[0]
        monitor['top'] += int(monitor['height'] * MONITOR_GAME_AREA_RATIO[0])
        monitor['left'] += int(monitor['width'] * MONITOR_GAME_AREA_RATIO[1])
        monitor['width'] = int(monitor['width'] * MONITOR_GAME_AREA_RATIO[2])
        monitor['height'] = int(monitor['height'] * MONITOR_GAME_AREA_RATIO[3])


def fit_window():
    """Iterates through windows"""
    wgui.EnumWindows(fit_window_callback, None)


def open_process(pid, access_flags):
    """Opens process with desired access level"""
    if DEBUG:
        print("open_process")
    handle = windll.kernel32.OpenProcess(access_flags, False, pid)
    if handle == 0:
        print("Error opening process", pid, "Error:", windll.kernel32.GetLastError())
    return handle


def initialize_process(executable_path):
    if DEBUG:
        print("initialize_process")
    start_obj = win32process.STARTUPINFO()
    proc_handle, thread_handle, pid, thread_id = win32process.CreateProcess(
        executable_path, None, None, None, 8, 8, None, None, start_obj
    )
    return proc_handle, thread_handle, pid, thread_id


def pause(*param):
    global active
    if not active:
        # Resume actions
        for key in action_stack:
            dk.PressKey(key)
        print("Resuming.")
    else:
        # Release keys while paused
        for key in action_stack:
            dk.ReleaseKey(key)
        print("Pausing.")
    active = not active


def pause_game():
    global game_paused
    if game_paused:
        # Resume actions
        dk.PressKey(dk.SCANCODES["ESC"])
        time.sleep(0.2)
        dk.ReleaseKey(dk.SCANCODES["ESC"])
        for key in action_stack:
            dk.PressKey(key)
        print("Resuming Game.")
    else:
        # Release keys while paused
        for key in action_stack:
            dk.ReleaseKey(key)
        dk.PressKey(dk.SCANCODES["ESC"])
        time.sleep(0.2)
        dk.ReleaseKey(dk.SCANCODES["ESC"])
        print("Pausing Game.")
    game_paused = not game_paused


def main():
    # Initiate process
    proc_handle, thread_handle, pid, thread_id = initialize_process(TH_PATH)
    adv_handle = open_process(pid, PROCESS_ALL_ACCESS)

    time.sleep(4)  # Wait for program to load
    # set_lives(adv_handle, 0)
    # set_score(adv_handle, 0)
    fit_window()
    possible_actions = [  # dk.SCANCODES["z"], dk.SCANCODES["x"],
        None,
        dk.SCANCODES["UP"], dk.SCANCODES["DOWN"],
        dk.SCANCODES["LEFT"], dk.SCANCODES["RIGHT"]]
    model = Model(FRAMES_FEED, len(possible_actions), monitor["width"], monitor['height']).to(DEVICE)
    target_model = Model(FRAMES_FEED, len(possible_actions), monitor["width"], monitor['height']).to(DEVICE)
    pause()
    keyboard.on_press_key(" ", pause)
    try:
        main_loop(adv_handle, possible_actions, model, target_model)
    finally:
        pass
        # windll.kernel32.CloseHandle(adv_handle)
        # windll.kernel32.CloseHandle(proc_handle)


def state_reshape(state_with_time_channel):
    """Concatenates the memory (timesteps) channels to RGB channels). A helper function"""
    shape_orig = state_with_time_channel.shape
    return np.reshape(state_with_time_channel,
                      (shape_orig[0], shape_orig[1] * shape_orig[2], shape_orig[3], shape_orig[4]))


if __name__ == "__main__":
    # global DEVICE
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")
    main()
