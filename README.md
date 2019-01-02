# Deep Touhou
Using DQN to play touhou

![Hidden Star in Four seasons](https://github.com/hannesliik/deep_thouhou/blob/master/touhou.PNG)
Work in progress...
* Uses Win API to read score from memory and modify number of lives.
* MSS to read screen pixels
* Pytorch for the Q-Network
* Emulates key presses that touhou will recognize
* OpenCV to resize and visualize the observations

Challenges:
* Replay buffer takes a LOT of memory for raw images.
* Observations are not evenly spaced: they are fetched as fast as possible (60fps without any burdening code like running the model to choose an action. Much less otherwise.)

Usage:
1) Install touhou 16: Hidden Star in Four Seasons and change its path in main.py. (This is tested on the steam demo. Might have different memory addresses for other versions).
2) Tweak paramaters in main.py
3) run main.py

Interesting observations so far:
* Learns that shooting gives reward and doesn't like to do anything else.
* Once learned that going up brings more reward than shooting (there is a boundary above which all points are collected automatically).
