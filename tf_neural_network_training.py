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

#Define global variables
board_shape = pytris.grid_size
cnn_input_shape = (1, board_shape[0], board_shape[1], 1)
g = tf.random.Generator.from_non_deterministic_state()
alambda = 30 #1/learning rate
num_iter = 3000

def gen_nn(seed_nn = None):
    cnn_input_shape = (1, board_shape[0], board_shape[1], 1)
    to_return = {}
    if seed_nn == None:
        to_return['cnn'] = tf.keras.layers.Conv2D(1, 4, activation='relu', input_shape = cnn_input_shape[1:], padding = "same", kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1.0))
        to_return['cnn'].trainable = False
        to_return['dense'] = [ #doing this bc I was having some trouble with sequentials
            tf.keras.layers.Dense(128, input_shape = (1, 226), kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1.0)),
            tf.keras.layers.Dense(128, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1.0)),
            tf.keras.layers.Dense(80, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1.0)) #10 columns * 4 possible orientations * 2 possible blocks = 80
        ]
    else:
        to_return = seed_nn

        #nudge cnn layer
        cnn_weight_deltas = tf.math.divide(g.normal(to_return['cnn'].weights[0].shape), alambda) 
        cnn_new_weights = tf.math.add(to_return['cnn'].weights[0], cnn_weight_deltas)
        #this next part is just to get these new weights to fit.
        cnn_new_weights_acceptable = [tf.Variable(cnn_new_weights), tf.Variable([0.0])]
        to_return['cnn'].set_weights(cnn_new_weights_acceptable)

        #nudge other layers
        for i in range(len(to_return['dense'])):
            d_weight_deltas = tf.math.divide(g.normal(to_return['dense'][i].weights[0].shape), alambda)
            d_new_weights = tf.math.add(to_return['dense'][i].weights[0], d_weight_deltas)
            d_bias_deltas = tf.math.divide(g.normal(to_return['dense'][i].weights[1].shape), alambda)
            d_new_biases = tf.math.add(to_return['dense'][i].weights[1], d_bias_deltas)
            d_new_weights_acceptable = [tf.Variable(d_new_weights), tf.Variable(d_new_biases)]
            to_return['dense'][i].set_weights(d_new_weights_acceptable)
    return to_return

def nn_process_state(nn, cur_state):
    prepared_board = tf.constant(cur_state['board'], shape = cnn_input_shape, dtype='float')

#Preprocess with one convolutional layer. Then pool into a dense network with n layers
    processed_board = nn['cnn'](prepared_board).numpy().reshape(1, -1)

    rest_of_model_input = cur_state['queue'] + [cur_state['cur_block'], int(cur_state['swapped']), cur_state['held_block']]
    full_model_input = np.append(processed_board, rest_of_model_input).reshape(226, 1)

#Build rest of model linearly
    output = full_model_input 
    for i in nn['dense']:
        output = i(output)
    return output

def process_outputs(output_tensor, ff=True):
    command = tf.argmax(output_tensor, axis=1)[0]
    command = int(command)
    keypress_list = []

    #0-39 = this block; 40-79 = held block
    if command >= 40:
        keypress_list.append(pygame.K_LSHIFT)
        command -= 40
    
    keypress_list += [pygame.K_UP * command // 10] # 0-3 rotations depending on which #

    if command % 10 < 5: #01234 = columns to the left
        keypress_list += [pygame.K_LEFT for i in range(command % 10)]
    elif command % 10 > 5: #56789 = columns to the right
        keypress_list += [pygame.K_RIGHT for i in range((command % 10) - 5)]

    if ff:
        keypress_list.append(pygame.K_DOWN) #No T-spins or fancy cat stuff like that yet
#   print(command, "->", end="")
#   for i in keypress_list:
#       if i == pygame.K_LSHIFT:
#           print("LSHIFT", end=", ")
#       elif i == pygame.K_UP:
#           print("UP", end = ", ")
#       elif i == pygame.K_DOWN:
#           print("DOWN", end = ", ")
#       elif i == pygame.K_LEFT:
#           print("LEFT", end = ", ")
#       elif i == pygame.K_RIGHT:
#           print("RIGHT", end = ", ")
#       else:
#           print(i)
#   print("")
    return keypress_list
    

nn_array  = [gen_nn() for i in range(100)]
top_n = [(None, None) for i in range(5)]
for h in range(num_iter):
    print("Iteration", h)
    for nn in nn_array:
        for i in pytris.main(hardcode_speed = 1):
            cur_state = json.loads(i)
            if "LOSS" in i:
                score = cur_state["LOSS"]
#                print("Got", score)
                assert(type(score) == int)
                for j in range(len(top_n)):
                    if top_n[j][0] == None or score > top_n[j][0]:
                        top_n[j] = (score, nn)
                        break
                break
            nn_output = nn_process_state(nn, cur_state)
            keypresses = process_outputs(nn_output)
            for j in keypresses:
                event = pygame.event.Event(pygame.KEYDOWN, key = j)
                pygame.event.post(event)
    nn_array = []
    print("TOP PERFORMERS:")
    for i in top_n:
        print(i[0])
        nn_array += [gen_nn(i[1]) for j in range(20)]

print("TOP PICKS:")
input("Press ENTER to see results!")
for i in top_n:
    for j in pytris.main(hardcode_speed = 3):
        cur_state = json.loads(j)
        if "LOSS" in j:
            score = cur_state["LOSS"]
            print("Final score:", score)
            break
        nn_output = nn_process_state(i, cur_state)
        keypresses = process_outputs(nn_output)
        for k in keypresses:
            event = pygame.event.Event(pygame.KEYDOWN, key = k)
            pygame.event.post(event)
