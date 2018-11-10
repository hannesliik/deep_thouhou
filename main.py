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
# In-project imports
from models import Model
from recording import create_recording, Recording, get_transition
from memory_operations import read_memory, write_memory
from replay_buffer import ReplayBuffer

# Settings
# - Window settings
SCALE = 1.25  # Windows scaling for high DPI displays, need to enter this manually
monitor = {'top': 0, 'left': 0, 'width': 800, 'height': 480}  # Default recording window
MONITOR_WINDOW_MARGIN = [35, 2, 2, 2]
MONITOR_GAME_AREA_RATIO =[0.03, 0.05, 390/650, 450/480]
# - Recording settings
FRAMES_SKIP = 1
EXPERIENCE_LENGTH = 60*60  # aka. experience length
RECORDING_PATH = "recordings\\"

# - Model settings
FRAMES_FEED = 5  # How many frames the model should take as input
DEATH_PENALTY = 5000
KEY_DURATION_SECONDS = 0

# - Training parameters
BATCH_SIZE = 64
LEARNING_RATE = 0.03

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
action_stack = []  # Holds the current actions

# https://ai.intel.com/demystifying-deep-reinforcement-learning/


def main_loop(handle, possible_actions: list, model: Model):
    with mss() as sct:
        counter = 0
        frame_counter = 0
        frame_skip_counter = 0
        new_recording = False
        score = 0
        lives = 3
        frame_times = [0, 0, 0, 0]
        replay_buffer = ReplayBuffer(200, (3 * FRAMES_FEED, monitor['height'], monitor['width']))

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
                # Logic for used frame
                recording.experience[frame_counter, :, :, :] = frame  # Save the frame in the experience record
                recording.counter += 1
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
                lives = new_lives
                score = new_score
                recording.rewards.append(reward)

                # TODO Logic to deal with a ready datapoint
                if recording.state_ready(FRAMES_FEED):
                    action_index, q_scores = choose_action(torch.from_numpy(recording.get_current_state(FRAMES_FEED).astype('float32')).cuda().unsqueeze_(0), model)
                    recording.q_scores.append(q_scores)
                    recording.actions.append(action_index)
                    execute_actions([possible_actions[int(action_index)], dk.SCANCODES["z"]])
                    optimize_model(model, None, recording)
                if DEBUG and SHOW_DATAPOINTS:
                    # Display data point for debugging
                    for i in range(recording.experience.shape[0]):
                        i1 = recording.experience[i, :, :, :]
                        i2 = np.moveaxis(i1, 0, 2)
                        cv2.imshow('datapoint_' + str(i), i2.astype(np.uint8))

            frame_skip_counter += 1
            frame_skip_counter = frame_skip_counter % FRAMES_SKIP

            frame_counter += 1
            frame_counter = frame_counter % EXPERIENCE_LENGTH
            new_recording = frame_counter == 0
            if new_recording:
                recording.save(RECORDING_PATH)
            # Frame timings and other utility
            endMillis = time.time()
            frame_time = endMillis - startMillis
            frame_times[counter % 4] = frame_time
            if counter % 4 == 0:
                print("frame time: %s"%(np.mean(frame_times)))
            counter += 1
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break


def optimize_model(policy_net: Model, target_net: Model, experience_memory: Recording):
    """Takes a random batch from :experience_memory and trains the policy_net on it for one iteration"""
    # Sample training data
    states, actions, next_states, rewards, q_scores_1, q_scores_2 =\
        experience_memory.sample_random_transitions(BATCH_SIZE, FRAMES_FEED)

    # If there is nothing to train on, return
    if states.shape[0] < 1:
        print("No examples given")
        return
    optimizer = torch.optim.RMSprop(policy_net.parameters())

    state_action_values = policy_net(torch.from_numpy(state_reshape(states).astype("float32")).cuda())
    rewards = torch.FloatTensor(rewards).cuda()

    # Create a Q target tensor
    expected_state_action_values = torch.zeros(state_action_values.size(), dtype=torch.float32, requires_grad=False).cuda()
    for i, action in enumerate(actions):
        q_score = torch.cuda.FloatTensor(q_scores_2[i])
        q_score = torch.squeeze(q_score)[action]
        reward = rewards[i]
        expected_state_action_values[i][action] = reward + q_score
        for j in range(state_action_values.size()[1]):
            if j != action:
                state_action_values[i][j] = 0
    loss = torch.nn.functional.mse_loss(state_action_values, expected_state_action_values)
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()



def choose_action(state, model: Model):
    if DEBUG and DEBUG_CONTROL:
        print("choose_action")
    action_scores = model.forward(state)
    return np.argmax(action_scores.detach().cpu().numpy()), action_scores


def grab_screen(monitor, recorder):
    """Makes a screen shot of the screen area defined in @monitor using @recorder
    :returns @frame and opencv compatible @frame_cv2"""
    im = recorder.grab(monitor=monitor)
    im = np.array(im, dtype=np.uint8)
    frame_cv2 = np.flip(im[:, :, :3], 2, ).astype(np.uint8)
    frame = np.moveaxis(frame_cv2, 2, 0, )
    return frame, frame_cv2


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
    if DEBUG:
        print("release", key)
    global action_stack
    if key in action_stack:
        del action_stack[action_stack.index(key)]
    dk.ReleaseKey(key)


def press(key, time=0):
    """Presses the key :key for :time seconds. :time 0 is infinite"""
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


def main():
    # Initiate process
    proc_handle, thread_handle, pid, thread_id = initialize_process(TH_PATH)
    adv_handle = open_process(pid, PROCESS_ALL_ACCESS)

    time.sleep(4)  # Wait for program to load
    #set_lives(adv_handle, 0)
    #set_score(adv_handle, 0)
    fit_window()
    possible_actions = [  # dk.SCANCODES["z"], dk.SCANCODES["x"],
                        dk.SCANCODES["UP"], dk.SCANCODES["DOWN"],
                        dk.SCANCODES["LEFT"], dk.SCANCODES["RIGHT"]]
    model = Model(FRAMES_FEED, len(possible_actions), monitor["width"], monitor['height']).cuda()

    pause()
    keyboard.on_press_key(" ", pause)
    try:
        main_loop(adv_handle, possible_actions, model)
    finally:
        pass
        #windll.kernel32.CloseHandle(adv_handle)
        #windll.kernel32.CloseHandle(proc_handle)


def state_reshape(state_with_time_channel):
    """Concatenates the memory (timesteps) channels to RGB channels). A helper function"""
    shape_orig = state_with_time_channel.shape
    return np.reshape(state_with_time_channel, (shape_orig[0], shape_orig[1] * shape_orig[2], shape_orig[3], shape_orig[4]))

if __name__ == "__main__":
    main()

