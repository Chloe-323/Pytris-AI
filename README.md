# Pytris-AI
AI to play my pytris game

# Thought process
Set up a with 2d convolutional layers that scan the board, once proximity wise and once row wise, and a Dense NN to select the best place to drop the block
Set up Dense NN to play the game. Each NN has random weights, and the ones that perform the best move on to the next generation
Each generation, the best performing NNs get picked and have their weights shifted slightly.
Performance metric is the final score.
Over time, a set of weights that plays well will develop through natural selection. 

# TODO
Right now the code is a mess. If I want to improve on this NN I need to clean it up.
I would like to implement multithreading.
Once the code is neater and I can run generations faster I might have a better selection process, or just go for a more complex NN.
