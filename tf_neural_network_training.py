import sys
sys.path.insert(1, "Pytris") #A bit hacky but good enough for now
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import pytris
import threading
import time
import json

import tensorflow as tf
import numpy as np

print("Hello world :3")

#Just for reference when I'm coding. Not at all necessary
sample_state = {
  "board": [
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
    [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
    [ 0, 0, 0, 1, 1, 0, 1, 0, 0, 0 ],
    [ 0, 0, 0, 1, 1, 0, 1, 1, 0, 1 ],
    [ 0, 1, 0, 1, 1, 1, 1, 1, 0, 1 ],
    [ 1, 1, 1, 1, 1, 1, 1, 1, 0, 1 ]
  ],
  "cur_block": 2,
  "swapped": False,
  "queue": [ 3, 5, 3 ],
  "held_block": -1
}

#Define variables
board_shape = pytris.grid_size
cnn_input_shape = (1, board_shape[0], board_shape[1], 1)

def gen_nn(seed_nn = None):
    cnn_input_shape = (1, board_shape[0], board_shape[1], 1)
    to_return = {}
    if seed_nn == None:
        to_return['cnn'] = tf.keras.layers.Conv2D(1, 4, activation='relu', input_shape = cnn_input_shape[1:], padding = "same")
        to_return['cnn'].trainable = False
        to_return['dense'] = [ #doing this bc I was having some trouble with sequentials
            tf.keras.layers.Dense(128, input_shape = (1, 226)),
            tf.keras.layers.Dense(128),
            tf.keras.layers.Dense(80) #10 columns * 4 possible orientations * 2 possible blocks = 80
        ]
#   else:
#       to_return = seed_nn

#       #nudge cnn layer
#       to_return['cnn'].weights[0]

#       #nudge other layers
    return to_return

def nn_process_state(nn, cur_state):
    prepared_board = tf.constant(cur_state['board'], shape = cnn_input_shape, dtype='float')

#Prepare layers:

#Preprocess with one convolutional layer. Then pool into a dense network with n layers
    processed_board = nn['cnn'](prepared_board).numpy().reshape(1, -1)

    rest_of_model_input = cur_state['queue'] + [cur_state['cur_block'], int(cur_state['swapped']), cur_state['held_block']]
    full_model_input = np.append(processed_board, rest_of_model_input).reshape(226, 1)

#Build rest of model linearly
    output = full_model_input 
    for i in nn['dense']:
        output = i(output)
    return output

def process_outputs(output_tensor):
    command = tf.argmax(output_tensor, axis=1)[0]
    keypress_list = []

    #0-39 = this block; 40-79 = held block
    if command >= 40:
        keypress_list.append(pygame.K_LSHIFT)
        command -= 40
    
    keypress_list += [pygame.K_UP * command // 10] # 0-3 rotations depending on which #

    if command % 10 < 5: #01234 = columns to the left
        keypress_list += [pygame.K_LEFT * command % 10]
    elif command % 10 > 5: #56789 = columns to the right
        keypress_list += [pygame.K_RIGHT * ((command % 10) - 5)]

#    keypress_list.append(pygame.K_DOWN) #No T-spins or fancy cat stuff like that yet
    return keypress_list
    

for i in pytris.main(hardcode_speed = 1):
    cur_state = json.loads(i)
    if "LOSS" in i:
        print("Got", cur_state["LOSS"])
        break
    nn_output = nn_process_state(gen_nn(), cur_state)
    keypresses = process_outputs(nn_output)
    for i in keypresses:
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {'key':i}))
