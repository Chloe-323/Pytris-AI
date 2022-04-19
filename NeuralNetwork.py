import sys
sys.path.insert(1, "Pytris") #A bit hacky but good enough for now
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import pytris
import threading
import time
import json
import copy
import uuid

from multiprocessing.pool import ThreadPool

import tensorflow as tf
import numpy as np

class TetrisAI: pass
board_shape = pytris.grid_size
cnn_input_shape = (1, board_shape[0], board_shape[1], 1)
g = tf.random.Generator.from_non_deterministic_state()
alambda = 30 #1/learning rate
num_iter = 3000
global_timer = time.monotonic()
#total_input_size = 248

class TetrisAI(tf.keras.Model):
    def __init__(self, parent = None):
        super().__init__()

        if type(parent) == TetrisAI:
            self.cn_layers = copy.deepcopy(parent.cn_layers)
            #nudge CN layers in random direction
            for i in self.cn_layers:
                cnn_weight_deltas = tf.math.divide(g.normal(i.weights[0].shape), alambda) 
                cnn_new_weights = tf.math.add(i.weights[0], cnn_weight_deltas)
                cnn_new_weights_acceptable = [tf.Variable(cnn_new_weights), tf.Variable([0.0])]
                i.set_weights(cnn_new_weights_acceptable)

            #nudge dense layers in random direction
            self.dense_layers = copy.deepcopy(parent.dense_layers)
            for i in self.dense_layers:
                d_weight_deltas = tf.math.divide(g.normal(i.weights[0].shape), alambda)
                d_new_weights = tf.math.add(i.weights[0], d_weight_deltas)
                d_bias_deltas = tf.math.divide(g.normal(i.weights[1].shape), alambda)
                d_new_biases = tf.math.add(i.weights[1], d_bias_deltas)
                d_new_weights_acceptable = [tf.Variable(d_new_weights), tf.Variable(d_new_biases)]
                i.set_weights(d_new_weights_acceptable)

        else:
            self.cn_layers = [
                    tf.keras.layers.Conv2D(1, 4, activation='sigmoid', input_shape = cnn_input_shape[1:], padding = "same", kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1.0)),
                    tf.keras.layers.Conv2D(1, (1, 10), activation='relu', input_shape = cnn_input_shape[1:], kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1.0)),
#                    tf.keras.layers.Conv2D(1, (22, 3), activation='relu', input_shape = cnn_input_shape[1:], kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1.0))
            ]
            self.dense_layers = [
                tf.keras.layers.Dense(256, 
                    kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1.0),
                    bias_initializer=tf.keras.initializers.RandomNormal(stddev=1.0)),
                tf.keras.layers.Dense(256,
                    kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1.0),
                    bias_initializer=tf.keras.initializers.RandomNormal(stddev=1.0)),
                tf.keras.layers.Dense(128,
                    kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1.0),
                    bias_initializer=tf.keras.initializers.RandomNormal(stddev=1.0)),
                tf.keras.layers.Dense(80, #10 columns * 4 possible orientations * 2 possible blocks = 80
                    kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1.0),
                    bias_initializer=tf.keras.initializers.RandomNormal(stddev=1.0))
            ]
            self.evaluate( { #warm up
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
            })




    def evaluate(self, cur_state):
        prepared_board = tf.constant(cur_state['board'], shape = cnn_input_shape, dtype='float')

        dense_layers_input = np.array(cur_state['queue'] + [cur_state['cur_block'], int(cur_state['swapped']), cur_state['held_block']]).reshape(-1, 1)
        for i in self.cn_layers:
            dense_layers_input = np.append(
                    dense_layers_input,
                    i(prepared_board).numpy().reshape(-1, 1)
                    )


        dense_layers_input = np.array(dense_layers_input).reshape(-1, 1)
        for i in self.dense_layers:
            dense_layers_input = i(dense_layers_input)

        return int(tf.argmax(dense_layers_input, axis = 1)[0]) #best way to do this?

    def process_outputs(self, command, ff=True):
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

        if not ff:
            print("normal speed")
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

    def play_tetris(self, headless = True):
#       filename = "games/" + uuid.uuid4().hex 
#       f = open(filename, "w+")
        local_headless_input = [] 
        score = 0
        prev_state = None
        for i in pytris.main(hardcode_speed = 1, headless = headless, headless_input = local_headless_input):
 #           print('0x' + hex(id(local_headless_input)))
#            f.write(i + "\n")
            cur_state = json.loads(i)
            if cur_state == prev_state:
                continue
            else:
                prev_state = cur_state
            if not headless:
                print(cur_state)
            if "LOSS" in i:
#                print(str(time.monotonic() - timer)[:4] + 's')
#                print(cur_state['LOSS'])
                return cur_state['LOSS']
            nn_output = self.evaluate(cur_state)
            keypresses = self.process_outputs(nn_output, ff = headless)

            #The problem with multithreading lies here: all the threads share one pool.
            #You need to find another way to get input through
            for j in keypresses:
                event = pygame.event.Event(pygame.KEYDOWN, key = j)
                if headless:
                    local_headless_input.append(event)
                else:
                    pygame.event.post(event)
        return score

    def avg_score_singlethread(self, n = 12):
        timer = time.monotonic()
        sum_score = 0
        for i in range(n):
            sum_score += self.play_tetris()
#        print("ST: ", str(time.monotonic() - timer)[:4] + 's')
        return int(sum_score / n)
    
    def avg_score(self, n=12):
        timer = time.monotonic()
        sum_score = 0
        pool = ThreadPool(processes=n)
        async_results = [pool.apply_async(self.play_tetris, ()) for i in range(n)]
        pool.close()
        pool.join()
        for i in async_results:
            sum_score += i.get()
#        print("MT", str(time.monotonic() - timer)[:4] + 's')
        return int(sum_score / n)


def generation(nn_array, per_gen = 128, top_picks = 8):
    top_n = [(None, None) for i in range(top_picks)]
    for nn in nn_array:
        print('#', end="")
        sys.stdout.flush()
#        nn_score = nn.avg_score_singlethread()
        nn_score = nn.avg_score()
        for j in range(len(top_n)):
            if top_n[j][0] == None or nn_score > top_n[j][0]:
                for k in range(len(top_n) - 2, j, -1):
                    top_n[k + 1] = top_n[k]
                top_n[j] = (nn_score, nn)
                break

    return top_n


def train_model(generations = 2500, per_gen = 32, top_picks = 8):
    nn_array = [TetrisAI() for i in range(per_gen)]
    top_n = [(None, None) for i in range(top_picks)]
    for i in range(generations):
        print("Generation ", i + 1, ": ", sep = "", end = "")
        sys.stdout.flush()
        top_n = generation(nn_array, per_gen, top_picks)
        nn_array = [j[1] for j in top_n]
        nn_array += [TetrisAI(top_n[0][1]) for j in range(per_gen // 4)] #1/4 for the top scorer
        for nn in top_n[1:]:
            nn_array += [TetrisAI(nn[1]) for j in range(per_gen // 2 // top_picks)] #1/16 for the top other ones

        nn_array += [TetrisAI() for j in range(per_gen - len(nn_array))] #the rest are randomized
        print("\n", top_n[0][0])
    return top_n[0][1]

def multithread_testing():
    pool = ThreadPool(processes = 12)
    a1, a2 = TetrisAI(), TetrisAI()
    async_results = [
            pool.apply_async(a1.play_tetris, ()),
            pool.apply_async(a2.play_tetris, ())
            ]
    pool.close()
    pool.join()
    for i in async_results:
        print(i.get())

#multithread_testing()
my_tetris_bot = train_model()
print("Your AI is ready madam")
while(1):
    input()
    my_tetris_bot.play_tetris(headless = False)
