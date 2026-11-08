# Simple deep Q-learning code for mancala game

This repo is a implementation of ai playing mancala.

## Requirement
Tested with python 3.12 with
- pytorch
- numpy

## Content
- **mancala_game.py**: file which contains the rule of mancala (see [wikipedia](https://en.wikipedia.org/wiki/Mancala)for more details).
- **model.py**: the pytorch model and the trainer.
- **agent.py**: the main agent and the training function.
- **player_vs_ai**: to launch a game between a human and ai.

## Run
To train the ai use 
```bash
python agent.py
```

To play with the ai use
```bash
python player_vs_ai.py
```

## Training method 
The code is inspired from [freeCodeCamp.org](https://www.youtube.com/watch?v=L8ypSXwyBds).

The agent play against itself to train both first turn and second turn.

The rewards I set are the following:
- One point for each stone which enter in the own goal.
- 5 points if put the last stone in the own goal (in other gain one more turn).
- -3 points if enters in the opponent side.
- 100 if won the game.
- -100 if lost the game.
