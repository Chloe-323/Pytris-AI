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

physical_device_name = tf.config.experimental.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(physical_device_name, True)

class TetrisAI: pass
board_shape = pytris.grid_size
cnn_input_shape = (1, board_shape[0], board_shape[1], 1)
g = tf.random.Generator.from_non_deterministic_state()
num_iter = 3000
global_timer = time.monotonic()
threads = 12
total_input_size = 257

class TetrisAI:
    def __init__(self, model_path = None):
        if model_path is None:
            self.model = self.build_model()
        else:
            self.model = tf.keras.models.load_model(model_path)
        return

    def build_model(self):
        #input layer
        board_input = tf.keras.Input(shape = cnn_input_shape)
        total_input = tf.keras.Input(shape = total_input_size)
        #convolutional layers
        board_conv = tf.keras.layers.Conv2D(filters = 4, kernel_size = (3, 3), padding = "same", activation = "elu")(board_input)
        #concatenate and flatten
        flatten_board = tf.keras.layers.Flatten()(board_conv)
        flatten_total = tf.keras.layers.Flatten()(total_input)
        concat_total = tf.keras.layers.concatenate([flatten_board, flatten_total])
        #dense layers to process
        dense_1  = tf.keras.layers.Dense(units = 102, activation = "tanh")(concat_total)
        dense_2  = tf.keras.layers.Dense(units = 100, activation = "tanh")(dense_1)
        dense_3  = tf.keras.layers.Dense(units = 98, activation = "tanh")(dense_2)
        dense_4  = tf.keras.layers.Dense(units = 96, activation = "tanh")(dense_3)
        dense_5  = tf.keras.layers.Dense(units = 94, activation = "tanh")(dense_4)
        dense_6  = tf.keras.layers.Dense(units = 92, activation = "tanh")(dense_5)
        dense_7  = tf.keras.layers.Dense(units = 90, activation = "tanh")(dense_6)
        dense_8  = tf.keras.layers.Dense(units = 88, activation = "tanh")(dense_7)
        dense_9  = tf.keras.layers.Dense(units = 86, activation = "tanh")(dense_8)
        dense_10 = tf.keras.layers.Dense(units = 84, activation = "tanh")(dense_9)
        dense_11 = tf.keras.layers.Dense(units = 82, activation = "tanh")(dense_10)
        dense_12 = tf.keras.layers.Dense(units = 80, activation = "tanh")(dense_11)
        dense_13 = tf.keras.layers.Dense(units = 78, activation = "tanh")(dense_12)
        dense_14 = tf.keras.layers.Dense(units = 76, activation = "tanh")(dense_13)
        dense_15 = tf.keras.layers.Dense(units = 74, activation = "tanh")(dense_14)
        dense_16 = tf.keras.layers.Dense(units = 72, activation = "tanh")(dense_15)
        dense_17 = tf.keras.layers.Dense(units = 70, activation = "tanh")(dense_16)
        dense_18 = tf.keras.layers.Dense(units = 68, activation = "tanh")(dense_17)
        dense_19 = tf.keras.layers.Dense(units = 66, activation = "tanh")(dense_18)
        dense_20 = tf.keras.layers.Dense(units = 64, activation = "tanh")(dense_19)
        dense_21 = tf.keras.layers.Dense(units = 64, activation = "tanh")(dense_20)
        dense_22 = tf.keras.layers.Dense(units = 64, activation = "tanh")(dense_21)
        dense_23 = tf.keras.layers.Dense(units = 64, activation = "tanh")(dense_22)
        dense_24 = tf.keras.layers.Dense(units = 64, activation = "tanh")(dense_23)
        dense_25 = tf.keras.layers.Dense(units = 64, activation = "tanh")(dense_24)
        dense_26 = tf.keras.layers.Dense(units = 64, activation = "tanh")(dense_25)
        dense_27 = tf.keras.layers.Dense(units = 64, activation = "tanh")(dense_26)
        dense_28 = tf.keras.layers.Dense(units = 64, activation = "tanh")(dense_27)
        dense_29 = tf.keras.layers.Dense(units = 64, activation = "tanh")(dense_28)
        dense_30 = tf.keras.layers.Dense(units = 64, activation = "tanh")(dense_29)
        dense_31 = tf.keras.layers.Dense(units = 64, activation = "tanh")(dense_30)
        dense_32 = tf.keras.layers.Dense(units = 64, activation = "tanh")(dense_31)

        #output layer
        output_layer = tf.keras.layers.Dense(units = 1, activation = "tanh")(dense_32)


        model = tf.keras.models.Model(
                inputs = [board_input, total_input],
                outputs = output_layer,
                name = "Tetris_AI"
                )
        print(model.summary())
        return model

    #TODO: learning rate goes down over time
    def train(self, x_train, y_train, x_test, y_test, epochs = 10, learn_rate = 0.002):
        self.model.compile(
                optimizer = tf.keras.optimizers.Adam(learning_rate = learn_rate),
                loss = "mean_squared_error",
                metrics = ["mean_absolute_error", "mean_squared_error"]
                )
        #board is first 220 elements
        board_train = x_train[:,:220].reshape(x_train.shape[0],1, board_shape[0], board_shape[1], 1)
        board_test = x_test[:,:220].reshape(x_test.shape[0],1, board_shape[0], board_shape[1], 1)
        self.model.fit(
                x = [board_train, x_train],
                y = y_train,
                epochs = int(epochs / 4),
                validation_data = ([board_test, x_test], y_test),
                batch_size = 32,
                verbose = 1
                )
        tf.keras.backend.set_value(self.model.optimizer.lr, learn_rate / 2)
        self.model.fit(
                x = [board_train, x_train],
                y = y_train,
                epochs = int(epochs / 4) * 3,
                validation_data = ([board_test, x_test], y_test),
                batch_size = 32,
                verbose = 1
                )
#       for i in range(epochs):
#           print("Epoch: {}".format(i))
#           print("Learning rate: {}".format(learn_rate * (1 - (i / epochs))))
#           self.model.fit(
#                   x = [board_train, x_train],
#                   y = y_train,
#                   epochs = 1,
#                   batch_size = 256,
#                   validation_data = (
#                       [board_test, x_test],
#                       y_test
#                       )
#                   )
#           tf.keras.backend.set_value(self.model.optimizer.lr, learn_rate * (1 - i/epochs))
        return

    def call(self, x):
        board_input = x[:220].reshape(-1, 1, board_shape[0], board_shape[1], 1)
        board_tensor = tf.convert_to_tensor(board_input)
        x_tensor = tf.convert_to_tensor(x.reshape(-1, total_input_size))

        output_tensor = self.model(
                [board_tensor, x_tensor],
                training = False
                )
        return output_tensor.numpy()[0][0]

    def save(self, path):
        self.model.save(path)
        return

    def play_tetris(self, headless = False):
        local_headless_input = []
        score = 0
        prev_state = None
        for i in pytris.main(hardcode_speed = 1, headless = headless, headless_input = local_headless_input):
            cur_state = json.loads(i)
            if "LOSS" in cur_state:
                return score
            if prev_state is not None:
                if cur_state["board"] == prev_state["board"]:
                    continue
            prev_state = cur_state
            top_predicted_move = 0
            top_predicted_score = -1000000
            for j in range(80):
                predicted_state = TetrisAIHelper.predict_outcome(cur_state, j)
                if "LOSS" in predicted_state:
                    continue
                score = self.call(
                        np.array(convert_state_to_input(predicted_state))
                        )
                print("Score for move ", j, ": ", score, sep = "")
                if score > top_predicted_score:
                    top_predicted_move = j
                    top_predicted_score = score
            print("Top move:", top_predicted_move)
            keypress_list = TetrisAIHelper.process_outputs(top_predicted_move, ff = headless)
            for keypress in keypress_list:
                event = pygame.event.Event(pygame.KEYDOWN, key = keypress)
                if headless:
                    local_headless_input.append(event)
                else:
                    pygame.event.post(event)


class TetrisAIHelper:
    def predict_outcome(cur_state, keypress):
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
#           new_state = predict_outcome(cur_state, process_outputs(i))
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
            this_outcome = TetrisAIHelper.predict_outcome(self.cur_state, i)
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


def onehot(piece):
    return [0 if piece != n else 1 for n in range(7)]

def convert_state_to_input(state):
    output = []
    #board as array of 0s and 1s
    for i in range(len(state["board"])):
        for j in range(len(state["board"][i])):
            output.append(state["board"][i][j])
    #one hot encoding of current block
    output += onehot(state["cur_block"])
    #one hot encoding of held block
    output += onehot(state["held_block"])
    #one hot encoding of next blocks
    for i in range(len(state["queue"])):
        output += onehot(state["queue"][i])
    #swapped
    output.append(state["swapped"])
    output.append(state["delta_score"])
    return output


def main():
    if len(sys.argv) != 5:
        print("Usage:", sys.argv[0], "<training data file> <save neural network file> <epochs> <learn rate>")
        exit(1)
    print("Data source:", sys.argv[1])
    print("Saved model location:", sys.argv[2])
    print("Number of epochs:", sys.argv[3])
#load the data
    print("Loading data...")
    x_list = []
    y_list = []
    f = open(sys.argv[1], "r")
    lines = f.readlines()
    random.shuffle(lines)
    line_split = [line.split("||") for line in lines]
    x_list = [convert_state_to_input(json.loads(ls[0][1:-1])) for ls in line_split]
    y_list = [float(ls[1]) for ls in line_split]

#fit the data
    print("Fitting data...")
#fit such that max is 1 and min is -1
    max_y = max(y_list)
    min_y = min(y_list)
    y_list = [(y - min_y) / (max_y - min_y) for y in y_list]

    ######
    
    mean = np.mean(np.array([xl[256] for xl in x_list]))
    std = np.std(np.array([xl[256] for xl in x_list]))
    x_list = [(xl - mean) / std for xl in x_list]

#convert to numpy array
    print("Converting to numpy array...")
    x_array = np.array(x_list)
    y_array = np.array(y_list)
    print("Converted.")


#split into training and testing data
    print("Splitting into training and testing data...")
    train_size = 0.8
    x_train = x_array[:int(len(x_array) * train_size)]
    y_train = y_array[:int(len(y_array) * train_size)]
    x_test = x_array[int(len(x_array) * train_size):]
    y_test = y_array[int(len(y_array) * train_size):]
    print("Training data: " + str(len(x_train)))
    print("Testing data: " + str(len(x_test)))


    ai = TetrisAI()
    ai.train(x_train, y_train, x_test, y_test, epochs = int(sys.argv[3]), learn_rate = float(sys.argv[4]))
    ai.save(sys.argv[2])
    while True:
        print("Your AI is ready madam")
        input()
        ai.play_tetris()

main()
