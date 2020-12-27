# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 23:04:20 2020

@author: Kais
"""

from Checker.AI.AISystems import MiniMaxAlphaBetaSystem
from Checker.AI.AISystems import FeaturesBasedSystem
from Checker.AI.Agent import Agent
from Checker.Engine import play, train, test_agents
from Checker.AI.BoardGenerators import random_game_generator
from UI import CLI


def start_game():

    print('Welcome to the Checker Game')
    print('please note that white is always plays first')

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
            level = int(input('enter the difficulty level [1-6]: '))
        except Exception:
            print('please enter an integer value!')
            continue
        if level > 6 or level < 1:
            print('please enter a value in range [1, 6]')
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
    # start_game()
    test()
# =============================================================================
#     def get_agent(name, learning_rate):
#         system = FeaturesBasedSystem(name, learning_rate, True)
#         agent = Agent('white', system)
#         return agent
#     ag1 = get_agent('tom_agent', 0.01)
# =============================================================================
# =============================================================================
#     # train(ag1, 100)
#     results = []
#     for i in range(3000):
#         initial_game = random_game_generator()
#         _, res = train(ag1, 1, initial_game, plots=False,
#                        results_output=False)
#         results.extend(res)
#         if (i+1) % 100 == 0:
#             print(f'{i+1}-th game finished!')
#     wins = results.count(1)
#     loses = results.count(-1)
#     draws = results.count(0)
#     print(f'training results of agent {ag1.get_name()}')
#     print(f'wins: {wins}')
#     print(f'loses: {loses}')
#     print(f'draws: {draws}')
#     print('##################################')
# =============================================================================
    # ag2 = get_agent('tom_agent', 0.01)
    # test_agents(ag1, ag2, 1, True)
