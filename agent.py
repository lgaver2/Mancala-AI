import torch
import random
import numpy as np
from collections import deque
from mancala_game import MancalaGame 
from model import Linear_QNet, QTrainer


MAX_MEMORY = 100000
BATCH_SIZE = 1000
EPSILON = 1000
EPOCH = 5000 
LR = 0.001


class Agent:

    def __init__(self, model=None):
       self.n_games = 0
       self.epsilon = 0 
       self.gamma = 0.9 #smaller than 1
       self.memory = deque(maxlen=MAX_MEMORY) # if exceed memory do popleft
       self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
       self.model = Linear_QNet(14, 256, 6).to(self.device) if model is None else model
       self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma, device=self.device)

    def get_state(self, game):
        state = np.concatenate((game.plate[game.player], game.plate[(game.player+1)%2]))
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuple
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones) 

    def train_short_memory(self, state, action, reward, next_state, done):
       self.trainer.train_step(state, action, reward, next_state, done) 

    def get_action(self, game: MancalaGame, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = EPSILON - self.n_games
        playable = game.playable_actions()
        if random.randint(0, 200) < self.epsilon:
            action = random.choice(playable)
        else:
            state0 = torch.tensor(state, dtype=torch.float).to(self.device)
            prediction = self.model(state0)

            for i in range(6):
                if i not in playable:
                    prediction[i] = -float("inf")
            action = torch.argmax(prediction).item()
        return action


def train():
    plot_scores = []
    record = [0, 0] 
    score = [0, 0]
    agent = Agent()
    game = MancalaGame(False, [1,1])
    step = 0
    while step < EPOCH:
        # get old state
        state_old = agent.get_state(game)

        # get action
        final_action = agent.get_action(game, state_old)

        # perform move and get new state
        done, reward = game.play_step(final_action)
        done = not done
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_action, reward, state_new, done)

        # remember
        agent.remember(state_old, final_action, reward, state_new, done)

        if done:
            # train long memory, plot the result
            agent.n_games += 1
            agent.train_long_memory()
            for i in range(2):
                score[i] = game.plate[i, 6]  
                if score[i] > record[i]:
                    record[i] = score[i]
            print(f"Game, {step + 1}, score, {score[0]} vs {score[1]}, record {max(record)}")

            plot_scores.append(score)

            game.reset()
            #plot(plot_scores, plot_mean_score)
            step += 1
    
    agent.model.save()

if __name__ == "__main__":
    train()