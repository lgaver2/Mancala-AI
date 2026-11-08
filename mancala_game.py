import numpy as np
import random

class MancalaGame:

    def __init__(self, gui, mode):
        self.plate = np.zeros((2, 7))
        self.gui = gui
        self.mode = mode
        self.reset()

    def reset(self):
        for i in range(2):
            for j in range(6):
                self.plate[i,j] = 4
        self.plate[0, 6] = 0
        self.plate[1, 6] = 0
        self.player = 0 #random.randint(0, 1)

    def show(self):
        print(f"Turn {self.player}")
        print(self.plate)

    def play_step(self, action):
        new_turn, reward = self.rule(action) 

        # Game end logic
        if np.sum(self.plate[0, 0:6]) == 0 or \
            np.sum(self.plate[1, 0:6]) == 0:

            for p in range(2):
                self.plate[p, 6] += np.sum(self.plate[p, 0:6])

                for i in range(6):
                    self.plate[p, i] = 0

            # final reward
            my_score = self.plate[self.player, 6]
            oppo_score = self.plate[(self.player + 1) % 2, 6]
            if my_score > oppo_score:
                reward += 100
            elif my_score < oppo_score:
                reward -= 100
            return False, reward

        if not new_turn:
            self.player = (self.player + 1) % 2

        return True, reward


    # return repeat
    def rule(self, idx):
        stones = self.plate[self.player, idx]
        self.plate[self.player, idx] = 0
        overTurn = self.player 
        reward = 0
        while stones > 0:
            idx += 1
            if idx < 7:
                stones -= 1
                self.plate[overTurn, idx] += 1
                if idx == 6:
                    if overTurn == self.player:
                        reward += 1
                    else:
                        # if oppoent basket do nothing (skip)
                        stones += 1
                        continue
                if idx == 6 and stones == 0 and overTurn == self.player:
                    if self.gui:
                        print("perfect")
                    reward += 5
                    return True, reward
                elif stones ==0 and overTurn == self.player and self.plate[overTurn, idx] == 1:
                    if self.gui:
                        print("capture")
                    oppo = (overTurn + 1)%2
                    capture = self.plate[oppo, idx] + 1
                    self.plate[overTurn, idx] = 0
                    self.plate[oppo, idx] = 0
                    self.plate[self.player, 6] += capture
                    reward += capture
            else:
                # inside opponent zone
                idx = -1 
                overTurn = (overTurn + 1) % 2
                reward -= 3
        return False, reward


    def playable_actions(self):
        return [i for i in range(6) if self.plate[self.player, i] != 0]
    
    def action(self):
        print(f"playable action {self.playable_actions()}")
        while True:
            i = int(input("which index?"))
            if self.correctIdx(i):
                break
        return i

    def correctIdx(self, idx):
        return idx >= 0 and idx < 6 and self.plate[self.player, idx] != 0