# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 23:04:20 2020

@author: Kais
"""

from Checker.AI.AISystems import MiniMaxAlphaBetaSystem
from Checker.AI.AISystems import FeaturesBasedSystem
from Checker.AI.Agent import Agent
from Checker.Engine import play, train, test_agents
import random


def start_game():

    print('#############################')
    print('#Welcome to the Checker Game#')
    print('#############################')
    print('please note that white is always plays first!')

    print('colours available for you are: white, black')
    while True:
        colour = input('enter the colour you want to play with: ')
        colour = colour.lower()
        if colour in ['white', 'black']:
            break
        else:
            print('please enter a valid colour!')
    while True:
        try:
            level = int(input('enter the difficulty level [1-5]: '))
        except Exception:
            print('please enter an integer value!')
            continue
        if level > 5 or level < 1:
            print('please enter a value in range [1, 5]')
            continue
        break
    agent_colour = 'white' if colour == 'black' else 'black'
    print(f'agent colour: {agent_colour}')
    print(f'agent difficulty: {level}')
    system5 = MiniMaxAlphaBetaSystem(level)
    agent = Agent(agent_colour, system5)
    play(agent)


def test():
    level = 5
    agent_colour = 'white'
    print(f'agent colour: {agent_colour}')
    print(f'agent difficulty: {level}')
    system5 = MiniMaxAlphaBetaSystem(level)
    agent = Agent(agent_colour, system5)
    play(agent)


if __name__ == '__main__':
    start_game()
    # test()
# =============================================================================
#     def get_agent(name, learning_rate = 0.0001):
#         system = FeaturesBasedSystem(name, learning_rate, True)
#         agent = Agent('white', system)
#         return agent
# 
#     learning_rate = 0.1
#     ag2 = get_agent('tom_agent')
#     ag3 = get_agent('tom_agent_test1')
# # =============================================================================
# #     for i in range(5):
# #         print(f'iteration {i+1}:')
# #         ag1 = get_agent('tom_agent_test', learning_rate)
# #         train(ag1, 200, explore_probability=5)
# #         learning_rate /= 2
# # =============================================================================
# 
# # =============================================================================
# #     ag1 = get_agent('tom_agent_test', learning_rate)
# #     train(ag1, 100, explore_probability=10)
# # =============================================================================
# 
#     # test_agents(ag1, ag2, 10)
#     # test_agents(ag1, ag3, 10)
#     print(f'{ag1.get_name()} parameters')
#     print(ag1.get_system()._parameters)
#     print(f'{ag2.get_name()} parameters')
#     print(ag2.get_system()._parameters)
# =============================================================================
