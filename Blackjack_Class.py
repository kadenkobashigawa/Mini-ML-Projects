import random
import time
import torch
import matplotlib.pyplot as plt
from Blackjack_Model import blackjack_model



class blackjack:


    def build_deck(self, n_decks = 1):
        
        '''builds full 52 card decks as a list of tuples:
        (card name, card value)...'''

        values = {}
        cards = []
        for n in range(2,11):
            values[str(n)] = n
            cards.append(str(n))
        faces = ['Jack', 'Queen', 'King']
        for face in faces:
            values[face] = 10
            cards.append(face)
        cards.append('Ace')
        values['Ace'] = 11
        suits = [' of Spades', ' of Clubs', ' of Hearts', ' of Diamonds']
        deck = []
        for i in range(n_decks):
            for suit in suits:
                for card in cards:
                    deck.append((card + suit, values[card]))
        return deck


    def draw(self, deck = []):
        
        '''draws one card removing it from the deck...'''

        if len(deck) == 0:
            print('Deck empty. Reshuffling.')
            deck = self.build_deck(self.n_decks)
            self.running_count = 0
        i = random.randint(0, len(deck) - 1)
        card = deck[i]
        deck.pop(i)
        return card
    
    
    def count_card(self, card):

        '''updates card counting variable for ai...'''

        value = card[1]
        if value <= 6:
            self.running_count += 1
        elif value >= 10:
            self.running_count -= 1


    def deal(self, deck = [], player = [], announce = True):
        
        '''deals a card to a player and updates the player\'s 
        score and history...'''

        time.sleep(self.delay)
        round_statstics = player[-1]
        card = self.draw(deck)
        if card[1] == 11:
            round_statstics[1] += 1
        if announce:
            print(f'{player[0]} draws: {card[0]}.')
            self.count_card(card)
        else:
            print(f'{player[0]} draws: *******.')
        round_statstics[0][-1] += card[1]
        round_statstics[2] += [card]
        player[-1] = round_statstics
    

    def check_for_bust(self, player):

        '''if bust, reduces aces or ends round...'''

        round_statstics = player[-1]
        if round_statstics[0][-1] > 21:
            if round_statstics[1] == 0:
                time.sleep(self.delay)
                if player[0] == 'House':
                    print(f'Bust! Y\'all got lucky!')
                    round_statstics[0][-1] = 0
                else:
                    print('Bust! House wins.')
                    round_statstics[0][-1] = 0
                    round_statstics[-1] += ['L']
                player[-1] = round_statstics
                return True
            else:
                round_statstics[1] -= 1
                round_statstics[0][-1] -= 10
                player[-1] = round_statstics
                return False
        else:
            return False
    

    def h_or_s(self, deck = [], player = []):

        '''hit or stand? the ball is in the player\'s field...'''
        
        while True:
            
            #natural delay...
            time.sleep(self.delay)

            #check if player has blackjack...
            round_statstics = player[-1]
            if round_statstics[0][-1] == 21:
                print(f'Blackjack! {player[0]} wins.')
                self.showdown_queue.append(player)
                break
            
            #if ai is playing...
            if player[2]:

                #check if split is a valid move...
                if len(round_statstics[2]) >= 2:
                    if round_statstics[2][-1][1] == round_statstics[2][-2][1]:
                        split = 1
                    else:
                        split = 0
                else: 
                    split = 0

                #then have ai move based on game stats...
                current_score = round_statstics[0][-1]
                ace_count = round_statstics[1]
                value_house_card = self.house[1][2][0][1]
                decks_left = len(self.deck) / 52
                self.true_count = self.running_count / decks_left

                state = self.ai.encode_state(current_score, 
                                             ace_count, 
                                             value_house_card, 
                                             split, 
                                             self.true_count)
                move = self.ai.hit_or_stand(state)
                
                round_statstics[3].append((state, move))
            
            #if person is playing, input function...
            else:
                move = input('\nHit or Stand h/s)?\n')

            #hit: deal a card and check for bust...
            if move == 'h':
                self.deal(deck, player, announce = True)
                if self.check_for_bust(player):
                    break
            
            #double: deal one more card and double inital bet...
            elif move == 'double':
                print(f'{player[0]} double downs.')
                player[-1][-1] += ['D']
                self.deal(deck, player, announce = True)
                if self.check_for_bust(player):
                    break
                if player[-1][0][-1] == 21:
                    time.sleep(self.delay)
                    print(f'Blackjack! {player[0]} wins.')
                    self.showdown_queue.append(player)
                else:
                    self.showdown_queue.append(player)
                break
            
            #split: remove last card from the round...
            elif move == 'split':
                
                #if player has more than one card...
                if len(round_statstics[2]) >= 2:

                    #and the last two cards have the same value...
                    if round_statstics[2][-1][1] == round_statstics[2][-2][1]:

                        #last card is removed from the round...
                        time.sleep(self.delay)
                        card = player[-1][2].pop()
                        print(f'Split! The {card[0]} was put to the side.')
                        player[-1][0][-1] -= card[1]
                        if card[1] == 11:
                            player[-1][1] -= 1
                        self.split.append(card)
                    
                    #splitting two unequal cards error...
                    else:
                        time.sleep(self.delay)
                        print('You can\'t split unless your last two cards are the same.')
                        if player[2]:
                            time.sleep(self.delay)
                            print(f'{player[0]} learned from its moves.')
                            last_state, last_move = player[-1][3][-1]
                            self.ai.learn(last_state, last_move, 1, 0, self.bet_ratio)
                
                #splitting one card error...
                else:
                    time.sleep(self.delay)
                    print('You need two of the same card to split.')
                    if player[2]:
                        time.sleep(self.delay)
                        print(f'{player[0]} learned from its moves.')
                        last_state, last_move = player[-1][3][-1]
                        self.ai.learn(last_state, last_move, 1, 0, self.bet_ratio)

            #stand: keep current score and move on to showdown...
            elif move == 's':
                self.showdown_queue.append(player)
                break
            
            #invalid move error...
            else:
                time.sleep(self.delay)
                print('Enter "h" for hit or "s" for stand.')


    def split_round(self, player = [], split_card = ()):
            
            '''for double the earnings...'''

            #split card is brought back and score is reset...
            time.sleep(self.delay)
            print(f'\nThe {split_card[0]} is back on the table.')
            player[-1][0].append(split_card[1])
            player[-1][1] = 0
            player[-1][2] = [split_card]
            if split_card[1] == 11:
                player[-1][1] += 1
            
            #reset split queue and end of game boolean...
            self.split = []
            
            #round part...
            self.h_or_s(self.deck, player)

            #if split again, start another split queue in split queue...
            for sc in self.split:
                self.split_round(player, sc)


    def showdown(self, deck = [], showdown_queue = []):

        '''house always wins...'''

        #skip if queue empty...
        if len(showdown_queue) == 0:
            return

        #find highest scorer in queue...
        scores = []
        for player in showdown_queue:
            scores += player[-1][0]
        highest_score = max(scores)
        
        #house reveals its hand...
        time.sleep(self.delay)
        print(f'\nHouse totals {self.house[1][0][-1]}.')
        
        #house hits until it...
        while True:

            #busts...
            if self.check_for_bust(self.house):
                break
            
            #hits blackjack...
            elif self.house[1][0][-1] == 21:
                time.sleep(self.delay)
                print('Blackjack! House wins.')
                break
            
            #or reaches 17, 18, 19, or 20...
            elif self.house[1][0][-1] >= 17 or self.house[1][0][-1] > highest_score:
                break

            else:
                self.deal(deck, self.house, announce = True)

        #final challenge...
        print('')
        no_dupes = []
        for player in showdown_queue:
            if player[0] not in no_dupes:
                for score in player[-1][0]:
                    time.sleep(self.delay)
                    if self.house[1][0][-1] > score:
                        print(f'House ({self.house[1][0][-1]}) beats {player[0]} ({score}).')
                        player[-1][-1] += ['L']
                    elif self.house[1][0][-1] < score:
                        print(f'{player[0]} ({score}) beats House ({self.house[1][0][-1]}).')
                        player[-1][-1] += ['W']
                    else:
                        print(f'{player[0]} ({score}) matches House ({self.house[1][0][-1]}).')
                        player[-1][-1] += ['T']
                no_dupes.append(player[0])


    def round(self, players = []):

        '''a single round of blackjack; updates player results & ai trajectory...'''

        #reset game stats...
        self.house = ['House', [[0], 0, []]]
        self.split = []
        self.showdown_queue = []
        for player in players:
            player[-1] = [[0], 0, [], [], []]

        #and check deck...
        if len(self.deck) < (5 + 5*len(players)):
            time.sleep(self.delay)
            print('Deck running low. Reshuffling cards.\n')
            self.deck = self.build_deck(self.n_decks)
            self.running_count = 0

        #deal two cards to house with one hidden...
        self.deal(self.deck, self.house, announce = True)
        self.deal(self.deck, self.house, announce = False)
        self.check_for_bust(self.house)

        #if blackjack, game over. house rules...
        if self.house[1][0][-1] == 21:
            time.sleep(self.delay * 3)
            print(f'Unfortunately, the second card was a {self.house[1][2][-1][0]}.')
            time.sleep(self.delay * 3)
            for player in players:
                player[-1][-1] += ['L']
            return
        
        #deal two cards to each player...
        for player in players:
            print('')
            self.deal(self.deck, player, announce = True)
            self.deal(self.deck, player, announce = True)
            self.check_for_bust(player)

            #hit or stand loop...
            self.h_or_s(self.deck, player)

            #if split, start split rounds...
            for sc in self.split:
                self.split_round(player, sc)

        #finish off the survivors...
        self.showdown(self.deck, self.showdown_queue)

        #return results from the round...
        time.sleep(self.delay * 3)
    

    def blackjack(self, bank_balance = 100.00, delay = True):

        '''full game with betting and ai...'''

        #ui stuff...
        top_border = f'\n\n\n{"-"*49}\n{"/♠️/♥️/♣️/♦️"*6}/\n'
        bottom_border = f'\n{"/♠️/♥️/♣️/♦️"*6}/\n{"-"*49}\n\n\n'
        blackjack_border = f'\n\n\n{"-"*49}\n{"/♠️/♥️/♣️/♦️"*2} ===BLACKJACK=== {"♠️/♥️/♣️/♦️/"*2}'

        #get number of players...
        print(top_border)
        while True:
            try:
                self.n_players = input(f'How many players? ')
                self.n_players = int(self.n_players)
                break
            except ValueError:
                print("Invalid input. Please enter a number.")
        print(f'Setting up game for {self.n_players} players...')
        print(bottom_border)

        #set up players...
        self.players = []
        self.losers = []
        self.names = []
        self.ai = None
        for i in range(self.n_players):

            #get player name, starting balance, and species...
            print(top_border)
            player_name = str(input(f'Player {i + 1} Name: '))
            player_name = f'Player {i + 1}' if player_name == '' else player_name
            final_player_name = f'{player_name}{random.randint(100, 999)}' if player_name in self.names else player_name
            self.names.append(player_name)
            print(f'{final_player_name} was given ${bank_balance:.2f} to play.')
            ai_boolean = True if final_player_name[:2].lower() == 'ai' else False

            #players:  [name, 
            #           bank balance, 
            #           AI?,
            #           bank record, 
            #           game record [W, L], 
            #           round statistics [scores, ace count, card history, trajectory, results]]
            self.players.append([final_player_name, 
                                 bank_balance, 
                                 ai_boolean, 
                                 [bank_balance], 
                                 [0, 0], 
                                 [[0], 0, [], [], []]])
            
            #if ai is playing, initialize model and try to load weights...
            if ai_boolean:
                self.ai = blackjack_model()
                try:
                    self.ai.load_state_dict(torch.load('blackjack_model.pth'))
                except (FileNotFoundError, RuntimeError):
                    print(f'{final_player_name} knows nothing of blackjack.')
            print(bottom_border)

        #game loop...
        while True:

            #set game settings...
            self.n_decks = 5
            self.deck = self.build_deck(self.n_decks)
            self.running_count = 0
            self.true_count = 0
            self.min_bet = 0.01
            self.bet_ratio = 0.2
            self.quit = False
            if delay:
                self.delay = 0.5
            else:
                self.delay = 0

            #round loop...
            self.round_n = 1
            while True:
                
                #betting loop...
                print(blackjack_border)
                print(f'{"-"*19} Round {self.round_n:>3} {"-"*19}\n\n')
                self.bets = []
                for player in self.players[:]:
                    
                    #get bets from...
                    time.sleep(self.delay)
                    print(f'\n{player[0]} has ${player[1]:.2f}.')

                    #ai players...
                    if player[2]:
                        time.sleep(self.delay)
                        bet = round(player[1] * self.bet_ratio, 2)
                        bet = bet if bet >= self.min_bet else self.min_bet
                        print(f'{player[0]} bet ${bet:.2f}.')
                        self.bets.append(bet)

                    #or real players...
                    else:
                        while True:
                            bet_input = input(f'Enter a bet: ')
                            if bet_input.lower() == 'q':
                                print(f'{player[0]} cashed out at ${player[1]:.2f}.')
                                self.players.remove(player)
                                self.losers.append(player)
                                break
                            try:
                                bet = round(float(bet_input), 2)
                                if bet < 0.01 or bet > player[1]:
                                    print("Invalid bet amount. Must be > 0 and ≤ your balance.")
                                    continue
                                print(f'{player[0]} bet ${bet:.2f}.')
                                self.bets.append(bet)
                                break
                            except ValueError:
                                print("Invalid input. Please enter a number.")

                #one round of blackjack card game...
                time.sleep(self.delay)
                print(blackjack_border)
                print(f'{"-"*19} Round {self.round_n:>3} {"-"*19}\n\n\n')
                self.round(self.players)
                    
                #deal out wins or losses based on the result of the round...
                print(blackjack_border)
                print(f'{"-"*19} Round {self.round_n:>3} {"-"*19}\n\n\n')
                for i in range(len(self.players)):
                    time.sleep(self.delay)
                    ds = 0
                    player_i_balance = self.players[i][1]
                    for result in self.players[i][-1][-1]:
                        if result == 'D':
                            ds += 1
                        elif result == 'W':
                            mult = 2 if ds > 0 else 1
                            self.players[i][1] += self.bets[i] * mult
                            print(f'{self.players[i][0]} recieves ${self.bets[i] * mult:.2f}.')
                            ds -= 1
                        elif result == 'L':
                            mult = 2 if ds > 0 else 1
                            self.players[i][1] -= self.bets[i] * mult
                            print(f'${self.bets[i] * mult:.2f} is collected from {self.players[i][0]}.')
                            ds -= 1
                        elif result == 'T':
                            mult = 2 if ds > 0 else 1
                            print(f'Push! {self.players[i][0]} keeps ${self.bets[i] * mult:.2f}.')
                            ds -= 1
                    
                    #ai learns from its first move...
                    if self.players[i][2]:
                        if len(self.players[i][-1][3]) > 0:
                            first_state, first_move = self.players[i][-1][3][0]
                            self.ai.learn(first_state, 
                                          first_move, 
                                          player_i_balance, 
                                          self.players[i][1], 
                                          self.bet_ratio)

                    #record statistics:
                    self.players[i][3].append(self.players[i][1])
                    if player_i_balance > self.players[i][1]:
                        self.players[i][4][1] += 1
                    else:
                        self.players[i][4][0] += 1
                        
                #update round number and loop to steal more money...
                self.round_n += 1
                for player in self.players[:]:
                    if player[1] <= self.min_bet:
                        time.sleep(self.delay)
                        print(f'{player[0]} was forced off the table.')
                        self.players.remove(player)
                        self.losers.append(player)
                time.sleep(self.delay)
                if len(self.players) == 0:
                    break

            #plot game...
            print(top_border)
            l_names = []
            for player in self.losers:
                l_names.append(player[0])
            pad = max(len(name) for name in l_names) + 2
            for player in self.losers:
                print(f'{player[0]:<{pad}} W: {player[4][0]:>3}   L: {player[4][1]:>3}   ', end = '')
                print(f'{player[4][0]/((player[4][1]+player[4][0] + 1e-10)) * 100:.2f}%')
                print(f'{"-"*49}')
                plt.plot(player[3], 
                         linestyle = '-', 
                         label = f'{player[0]:<{pad}}: {player[4][0]/((player[4][1]+player[4][0] + 1e-10)) * 100:.2f}%')
            print(bottom_border)
            
            plt.title(f'Blackjack Game Statistics')
            plt.xlabel('Rounds')
            plt.ylabel('Bank Balance ($)')
            plt.grid(True)
            plt.legend()
            plt.show()
            
            #ask users if they'd like to lose more money...
            rib = input('Play again? (ENTER/q) ')
            if rib == 'q':
                break
            else:
                for player in self.losers[:]:
                    player[1] = bank_balance
                    player[3] = [bank_balance]
                    player[4] = [0, 0]
                    self.players.append(player)
                    self.losers.remove(player)               
        print(f'\n\n\n\n')