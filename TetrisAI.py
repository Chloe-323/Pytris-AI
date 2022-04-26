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
import random
import IPython
import pickle

import tensorflow as tf
import numpy as np

class TetrisAI: pass
board_shape = pytris.grid_size
decision_tree_depth = 4
decision_tree_shape = (1, board_shape[0], board_shape[1], 1 + 80 + 80**2 + 80**3)
cnn_input_shape = (1, board_shape[0], board_shape[1], 1)
g = tf.random.Generator.from_non_deterministic_state()
alambda = 30 #1/learning rate
num_iter = 3000
global_timer = time.monotonic()
threads = 12
total_input_size = 248

class TetrisAI:
    def __init__(self, model_path = None):
        if model_path is None:
            self.model = self.build_model()
        else:
            self.model = tf.keras.models.load_model(model_path)
        return

    def build_model(self):
        #input layer
        board_input = tf.keras.Input(shape = decision_tree_shape)
        #convolutional layers
        board_conv = tf.keras.layers.Conv2D(filters = 32, kernel_size = (3, 3), padding = "same", activation = "relu")(board_input)
        column_conv = tf.keras.layers.Conv2D(filters = 32, kernel_size = (3, board_shape[1]), padding = "same", activation = "relu")(board_input)
        #concatenate and flatten
        concat_layer = tf.keras.layers.concatenate([board_conv, column_conv])
        flatten_layer = tf.keras.layers.Flatten()(concat_layer)
        #dense layers to process
        dense_layer_1 = tf.keras.layers.Dense(units = 64, activation = "relu")(flatten_layer)
        dense_layer_2 = tf.keras.layers.Dense(units = 64, activation = "relu")(dense_layer_1)
        dense_layer_3 = tf.keras.layers.Dense(units = 64, activation = "relu")(dense_layer_2)
        dense_layer_4 = tf.keras.layers.Dense(units = 64, activation = "relu")(dense_layer_3)
        #output layer
        output_layer = tf.keras.layers.Dense(units = 1, activation = "sigmoid")(dense_layer_4)
        model = tf.keras.models.FunctionalModel(
                inputs = board_input,
                outputs = output_layer
                )
        print(model.summary())
        return model

class TetrisAIHelper:
    def _predict_outcome(cur_state, keypress):
        keypress_list = TetrisAIHelper.process_outputs(keypress)
        local_headless_input = [pygame.event.Event(pygame.KEYDOWN, key = i) for i in keypress_list]
        return json.loads(pytris.main(json_state = json.dumps(cur_state), hardcode_speed = 1, headless = True, headless_input = local_headless_input).__next__())

    def process_outputs(command, ff=True):
        keypress_list = []

        #0-39 = this block; 40-79 = held block
        if command >= 40:
            keypress_list.append(pygame.K_LSHIFT)
            command -= 40
        
        keypress_list += [pygame.K_UP for i in range(command // 10)] # 0-3 rotations depending on which #

        if command % 10 < 5: #01234 = columns to the left
            keypress_list += [pygame.K_LEFT for i in range(command % 10)]
        elif command % 10 >= 5: #56789 = columns to the right
            keypress_list += [pygame.K_RIGHT for i in range((command % 10) - 4)]

        if not ff:
            print("normal speed")
        if ff:
            keypress_list.append(pygame.K_DOWN) #No T-spins or fancy cat stuff like that yet
        return keypress_list

    def _score(state):
        return state["score"]

#   def _get_max_q_value(self, depth, cur_state):
#       score = self._score(cur_state)
#       if depth == 0:
#           return score
#       to_return = score
#       for i in range(80):
#           new_state = _predict_outcome(cur_state, process_outputs(i))
#           new_q = -1
#           if new_state["LOSS"]:
#               new_q = -1
#           else:
#               new_q = self._get_max_q_value(depth - 1, new_state)
#           if new_q > to_return:
#               to_return = new_q
#       return to_return

#TODO: filter out duplicate states
class DecisionTreeNode:
    def __init__(self, depth, cur_state, gamma = 0.5, score_func = TetrisAIHelper._score):
        self.depth = depth
        self.cur_state = cur_state
        if "LOSS" in self.cur_state:
            self.score = -1
            self.q_value = -1
            self.best_move = 0
            return
        self.children = [None for i in range(80)]
        self.gamma = gamma
        self.score_func = score_func
        self.generate_children()
        self.update_q_value()
        return
    def update_q_value(self):
        if self.depth == 0:
            self.q_value = self.score_func(self.cur_state)
            self.best_move = None
            return
        self.q_value = 0
        score = self.score_func(self.cur_state)
        for i in range(len(self.children)):
            this_branch_qval = score + self.gamma * self.children[i].q_value
            if this_branch_qval > self.q_value:
                self.q_value = this_branch_qval
                self.best_move = i
        return
    def generate_children(self):
        if self.depth == 0:
            return
        for i in range(80):
            #check for duplicates
            this_outcome = TetrisAIHelper._predict_outcome(self.cur_state, i)
            if this_outcome in self.children:
                continue
            self.children[i] = DecisionTreeNode(self.depth - 1, this_outcome, self.gamma, self.score_func)
    def print_tree(self, tree_depth = 0):
        print("\t" * tree_depth + "Decision Tree Node:")
        print("\t" * tree_depth + "Depth: " + str(self.depth))
        print("\t" * tree_depth + "Score: " + str(self._score))
        print("\t" * tree_depth + "Q-Value: " + str(self.q_value))

        for child in self.children:
            child.print_tree()
        return

class DecisionTree:
    def __init__(self, depth, cur_state, gamma = 0.5, score_func = TetrisAIHelper._score):
        timer = time.monotonic()
        print("Building decision tree...")
        self.depth = depth
        self.cur_state = cur_state
        self.gamma = gamma
        self.score_func = score_func
        self.root = DecisionTreeNode(depth, cur_state, gamma, score_func)
        print("Decision tree built in " + str(time.monotonic() - timer)[:4] + " seconds.")
        return
    def get_q_value(self):
        return self.root.q_value
    def get_best_move(self):
        return self.root.best_move
    def update_q_value(self, cur_state):
        self.root.update_q_value(cur_state)
        return
    def print_tree(self):
        self.root.print_tree(tree_depth = self.depth)
        return


#Pop a shell if needed
if len(sys.argv) == 1:
    print("No arguments given. Pop a shell.")
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
        [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 0 ],
        [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 0 ],
        [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 0 ],
        [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 0 ]
      ],
      "cur_block": 3,
      "swapped": False,
      "queue": [ 3, 5, 3 ],
      "held_block": 2,
      "score": 20
    }
    IPython.embed()
    exit()

#Normal code

#Generate training data
print("Generating training data...")
training_data = []
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
    [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 0 ],
    [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 0 ],
    [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 0 ],
    [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 0 ]
  ],
  "cur_block": 3,
  "swapped": False,
  "queue": [ 3, 5, 3 ],
  "held_block": 2,
  "score": 20
}

#Generate initial training data
for i in range(20000):
    print("\tGenerating sample state...")
    decision_tree = DecisionTree(depth = 2, cur_state = sample_state)
    #randomize sample_state
    training_data.append((sample_state, decision_tree.get_q_value()))
    sample_state["board"] = [
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
        ]
    for i in range(19):
        sample_state["board"].append([random.randint(0,1) for j in range(10)])
    sample_state["cur_block"] = random.randint(0, 6)
    sample_state["swapped"] = bool(random.randint(0, 1))
    sample_state["queue"] = [random.randint(0, 6) for i in range(3)]
    sample_state["held_block"] = random.randint(0, 6)
    sample_state["score"] = random.randint(0, 100)
    print(sample_state)

print("Training data generated.")

#convert training data to numpy array
training_data = np.array(training_data)
print("Training data converted to numpy array.")
print(training_data)

#write training data to file
print("Writing training data to file...")
with open(sys.argv[1], "w+") as f:
    json.dump(training_data.tolist(), f)
print("Training data written.")
