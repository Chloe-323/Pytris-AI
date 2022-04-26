# Pytris-AI
AI to play my pytris game

# Project Log

## Attempt 1:
### Thought Process
Set up a with 2d convolutional layers that scan the board, once proximity wise and once row wise, and a Dense NN to select the best place to drop the block.
Set up Dense NN to play the game. Each NN has random weights, and the ones that perform the best move on to the next generation.
Each generation, the best performing NNs get picked and have their weights shifted slightly..
Performance metric is the final score.
Over time, a set of weights that plays well will develop through natural selection. 

### Code
`tf_neural_network_training.py` is the first attempt at implementing this, and `NeuralNetwork.py` is that same code just cleaned up and with some extra work done on it.

### Issues
I believe this can work but it was giving me weird results. I think there were a lot of bugs in the code. The way I envisoned this, with randomly stumbling around
and hoping to come across something that works, is probably not feasible unless I have a supercomputer. 

## Attempt 2:
### Thought Process
So I did some reading as to what others did, because it doesn't seem normal to me that you need a supercomputer to play Tetris. From what I can tell, the way to do
this is to set up what's called a Q-learning algorithm. The basic idea is that you have a scoring function s(p) to grade each position based on some metrics (e.g. score, holes, bumpiness, etc...).
Then you have a Q function which is defined as Q(p) = (s(p) + g(Q(p+1))), where s(p) is the score function and g is some gamma value between 0 and 1. You can read more about this here (https://en.wikipedia.org/wiki/Q-learning),
but the idea is that calculating an accurate Q value will take a while (see the python implementation which cannot go past a depth of 2 within a reasonable timeframe), so your NN will approximate the Q value.

## Implementation
The way I did this for Tetris is with a decision tree. Each state is a node on a tree, and you can travel from one to the next by taking an action in the game. The implementation is depth-first, and Q-values are automatically updated.
In Python, a tree with a depth of 2 takes about 2 seconds to calculate, and a tree with a depth of 3 takes about 120 seconds. This is not acceptable, so I decided to reimplement it in C++, which has a 100x faster runtime than Python.
Indeed, in C++, a tree with a depth of 2 takes about 0.04 seconds to calculate and a tree of depth 3 takes about 1.6 seconds.

I did some additional optimizations in that nodes that have bad scores get pruned right away. This kind of defeats the purpose of Q-learning in that these bad scores might lead to better outcomes down the road, and I plan to rework this
to either only prune if there is no way that the node can score better than the top condender, or to prune after a given number of unfruitful iterations. The reason I had to do this is that with a depth of 4 and no pruning, my computer
literally runs out of memory before it can finish. In Tetris, there are a lot of garbage moves that aren't worth looking at so I think that this is a worthwhile tradeoff. 

With this optimization, a tree with a depth of 4 and a cutoff of 15 (top 15 nodes get picked) can be built in 2.3 seconds. A tree with a depth of 5 and a cutoff of 5 can be built in 13 seconds. Moreover, this is starting from an empty board
where there are a lot of identical scoring moves. In these cases, the code will consider both. So even if you set the cutoff to 5, if there are 30 moves that have the same score, then all 30 of those moves will be considered.

My goal now is to use this decision tree to generate training examples as either (state, move) pairs or (state, Q-value) pairs. Then, I use these training examples to train the neural network.

### Code
`TetrisAI.py` is the python code containing the actual neural network as well as my first implementation of the decision tree. `decision_tree.cpp` is the C++ code to generate the decision tree.
