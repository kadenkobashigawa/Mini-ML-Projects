import torch
import torch.nn as nn
import torch.nn.functional as F
import random



class blackjack_model(nn.Module):


    def __init__(self):

        '''four fully connected layers...'''

        super(blackjack_model, self).__init__()
        
        #inputs (5): player total, soft aces, dealer card, split, true count...
        self.fc1 = nn.Linear(5, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 4)
        #outputs (4): hit, stand, double, split...

        self.optimizer = torch.optim.Adam(self.parameters(), lr = 1e-6)
        self.device = 'cpu'


    def forward(self, x):

        '''simple mlp to decide move weights...'''

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    

    def encode_state(self, player_total, soft_ace, dealer_card_val, split, true_count):

        '''converts game stats into tensor form...'''

        return torch.tensor([
            player_total,
            soft_ace,
            dealer_card_val,
            split,
            true_count,
        ], dtype = torch.float32).to(self.device)
    

    def hit_or_stand(self, state, epsilon = 0.1):

        '''uses model to choose a move based on hand...'''

        if state.dim() == 1:
            state = state.unsqueeze(0)

        if random.random() < epsilon:
            move = random.randint(0, 3)
        else:
            with torch.no_grad():
                q_values = self.forward(state)
                move = torch.argmax(q_values).item()

        return ['h', 's', 'double', 'split'][move]
    

    def learn(self, state, action, i_balance, j_balance, bet_ratio):
 
        '''predicted profit is compared to actual outcome
        for parameter adjustment...'''

        action_to_index = {
            'h': 0,
            's': 1,
            'double': 2,
            'split': 3
        }

        if state.dim() == 1:
            state = state.unsqueeze(0)

        q_values = self.forward(state)
        index = action_to_index[action]
        q_value = q_values[0, index]
        reward = (j_balance - i_balance) / i_balance * bet_ratio
        reward = max(min(reward, 1.0), -1.0)
        target = torch.tensor(reward, dtype = torch.float32)

        self.loss = F.mse_loss(q_value, target)

        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        torch.save(self.state_dict(), 'blackjack_model.pth')