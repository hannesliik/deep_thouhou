# Deep Thouhou
Using DQN to play touhou

* Uses Win API to read score from memory and modify number of lives.
* MSS to read screen pixels
* Pytorch for the Q-Network
* Emulates key presses that touhou will recognize
* OpenCV to visualize the observations

Challenges:
* Replay buffer takes a LOT of memory for raw images.
* Observations are not evenly spaced: they are fetched as fast as possible (60fps without any burdening code like running the model to choose an action. Much less otherwise.)

Usage:
1) Install thouhou 16 and change its path in main.py. (This is tested on the steam demo. Might have different memory addresses for other versions).
2) Don't run main.py, it's not ready yet :)
