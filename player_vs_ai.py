import torch
from mancala_game import MancalaGame
from agent import Agent
from model import Linear_QNet

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    game = MancalaGame(True, [0, 1])
    model = Linear_QNet(14, 256, 6).to(device)
    model.load_state_dict(torch.load("model/model1.pth", map_location=torch.device(device), weights_only=True))
    model.eval()
    agent = Agent(model)
    ge = True
    while ge:
        game.show()
        if game.player == 0:
            idx = game.action()
        else:
            state = agent.get_state(game)
            idx = agent.get_action(game, state) 
        ge, _ = game.play_step(idx)
    print("Game end")
    print(f"Winner {0 if game.plate[0, 6] > game.plate[1, 6] else 1}")

if __name__ == "__main__":
    main()