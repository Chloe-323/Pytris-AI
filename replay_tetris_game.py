import sys
import json

filename = sys.argv[1]
with open(filename) as file:
    for line in file:
        if "LOSS" in line:
            break
        state = json.loads(line.strip())
        board = state['board']
        for i in board:
            print(['_' if j == 0 else 'X' for j in i])
        input()
        print('\x1Bc')
