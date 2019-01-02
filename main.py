from threading import Timer
import os
import re
import time
import pickle
from ctypes import *
import ctypes.wintypes as wtypes
from collections import deque

# Library imports
import numpy as np
import torch
import cv2
from mss import mss
import win32.win32gui as wgui
import win32process
import keyboard
import torch.nn.functional as F

# In-project imports
import directkeys as dk
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
FRAMES_FEED = 3  # How many frames the model should take as input
DEATH_PENALTY = 5000
KEY_DURATION_SECONDS = 0

# - Training parameters
BATCH_SIZE = 16
LEARNING_RATE = 0.03
GAMMA = 0.9
GRAD_CLIP = 10
TARGET_MODEL_UPDATE_FREQ = 1500
TRAIN_FREQ = 128
BATCHES_PER_TRAIN = 5
REPLAY_BUFFER_SIZE = 500
N_STEP_REWARD = 3

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
PAUSE_ON_TRAIN = True
LOAD_MODEL = False
MODEL_PATH = "latest_model_state.pkl"
RESIZE_HEIGHT = 200
RESIZE_WIDTH = 120
# REFIT_WINDOW = False

# - Global variables
pid = 0  # Game process id
active = True
game_paused = False
action_stack = []  # Holds the current actions
DEVICE = torch.device("cpu")

DATA_GATHERING_MODE = False # Set true to record a dataset
DATA_GATHERING_N_FRAMES = 4  # every Nth frame will be saved
DATA_GATHERING_SAVE_RATE = 1024  # When N frames are captured, save to disk

MEAN, STD = [0.41650942, 0.3781172,  0.3798624], [0.29010016, 0.2663686, 0.26838425]

class ExplorationScheduler:
    def __init__(self, val=0.95, decay: float = 0.999, minval: float = 0.05):
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
    if DATA_GATHERING_MODE:
        frames = []
    with mss() as sct:
        counter = 0
        frame_skip_counter = 0
        score = 0
        lives = 3
        frame_times = [0, 0, 0, 0]
        replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE,
                                     (3 * FRAMES_FEED, RESIZE_HEIGHT, RESIZE_WIDTH),
                                     FRAMES_FEED,
                                     baseline_priority=1,
                                     gamma=GAMMA,
                                     reward_steps=N_STEP_REWARD)
        t = 0
        action = 0
        while True:
            if not active:
                time.sleep(0.5)  # Wait some time and check if recording should be resumed.
                continue

            startMillis = time.time()  # Time

            # Grab frames
            frame, frame_cv2 = grab_screen(monitor, sct)
            if DATA_GATHERING_MODE:

                t += 1
                print(t)
                if t % DATA_GATHERING_N_FRAMES == 0:
                    frames.append(frame)
                    if len(frames) == 2000:
                        np.save(f"frames_{t}", np.array(frames))
                        frames = []
                continue
            # Show frame
            if DEBUG:
                cv2.imshow('window1', frame_cv2)
            # Check if frame will be skipped. Not skipped if counter is 0
            if frame_skip_counter == 0:
                reward, score, lives = get_reward(handle, lives, score)

                # print(action, reward)
                if replay_buffer.waiting_for_effect:
                    replay_buffer.add_effects(action, reward)
                replay_buffer.push_frame(frame)
                if replay_buffer.buffer_init() and np.random.random() > exp_schedule.value(t):
                    action = choose_action(replay_buffer.encode_last_frame(), model)
                else:
                    action = np.random.randint(0, len(possible_actions))

                execute_actions([possible_actions[int(action)]]),  # dk.SCANCODES["z"]

                # Logic to deal with a ready datapoint
                if replay_buffer.can_sample(BATCH_SIZE) and t % TRAIN_FREQ == 0:
                    if PAUSE_ON_TRAIN:
                        pause_game()
                    for _ in range(BATCHES_PER_TRAIN):
                        optimize_model(model, target_model, replay_buffer, optimizer, num_actions=len(possible_actions))
                    if PAUSE_ON_TRAIN:
                        pause_game()

                # Copy model weights to target
                if t % TARGET_MODEL_UPDATE_FREQ == 0:
                    print("Saving model")
                    state_dict = model.state_dict()
                    torch.save(state_dict, MODEL_PATH)
                    print("done pickling")
                    target_model.load_state_dict(state_dict)
                    target_model.eval()

            frame_skip_counter += 1
            frame_skip_counter = frame_skip_counter % FRAMES_SKIP

            # Frame timings and other utility
            endMillis = time.time()
            frame_time = endMillis - startMillis
            frame_times[counter % 4] = frame_time
            t += 1
            # if counter % 4 == 0:
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
    reward -= 1  # Discourage idleness
    return reward, new_score, new_lives


def optimize_model(policy_net: Model, target_net: Model, replay_buffer: ReplayBuffer, optimizer, num_actions):
    """Takes a random batch from :experience_memory and trains the policy_net on it for one iteration"""
    # Sample training data
    obs_batch, actions_batch, rewards_batch, obs_next_batch, sample_ix = replay_buffer.sample(BATCH_SIZE)
    actions_batch: torch.Tensor = torch.from_numpy(actions_batch).to(DEVICE)
    obs_batch_torch: torch.Tensor = torch.from_numpy(obs_batch).to(DEVICE)
    obs_next_batch_torch: torch.Tensor = torch.from_numpy(obs_next_batch).to(DEVICE)

    q_t_batch = policy_net(obs_batch_torch)
    q_t_ac = q_t_batch.gather(1, actions_batch.unsqueeze_(1))
    print(obs_next_batch_torch.shape)
    with torch.no_grad():
        rewards_batch_torch = torch.from_numpy(rewards_batch).float().to(DEVICE)
        q_tp1 = policy_net(obs_next_batch_torch)
        _, q_tp1_maxind = q_tp1.max(1)
        q_tp1_target = target_net(obs_next_batch_torch)
        q_target = rewards_batch_torch.unsqueeze_(1) + GAMMA * q_tp1_target.gather(1, q_tp1_maxind.unsqueeze(1))

    errors: torch.Tensor = F.smooth_l1_loss(q_t_ac, q_target.float().to(DEVICE), reduction='none')
    loss = errors.mean(dim=0)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), GRAD_CLIP)
    optimizer.step()
    # Update replay buffer priorities
    print(errors.shape)
    replay_buffer.add_errors(sample_ix, errors.detach().squeeze_().cpu().numpy())


def choose_action(obs, model: Model):
    if DEBUG and DEBUG_CONTROL:
        print("choose_action")
    with torch.no_grad():
        obs_torch = torch.from_numpy(obs).to(DEVICE)
        action_scores = model.forward(obs_torch.unsqueeze_(0))
    return np.argmax(action_scores.detach().cpu().numpy())


def grab_screen(monitor, recorder):
    """Makes a screen shot of the screen area defined in @monitor using @recorder
    :returns @frame and opencv compatible @frame_cv2"""
    im = recorder.grab(monitor=monitor)
    im = np.array(im, dtype=np.uint8)
    im = cv2.resize(im, (RESIZE_WIDTH, RESIZE_HEIGHT))
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
    if DEBUG and DEBUG_CONTROL:
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
    possible_actions = [dk.SCANCODES["z"],  # dk.SCANCODES["x"],
                        None,
                        dk.SCANCODES["UP"], dk.SCANCODES["DOWN"],
                        dk.SCANCODES["LEFT"], dk.SCANCODES["RIGHT"]]
    model = Model(FRAMES_FEED, len(possible_actions), RESIZE_WIDTH, RESIZE_HEIGHT).to(DEVICE)
    model.encoder.load_state_dict(torch.load("encoder_state_dict.pth"))
    if LOAD_MODEL:
        model.load_state_dict(torch.load(MODEL_PATH))
    target_model = Model(FRAMES_FEED, len(possible_actions), RESIZE_WIDTH, RESIZE_HEIGHT).to(DEVICE)
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
    gpu = True
    if torch.cuda.is_available() and gpu:
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")
    main()
