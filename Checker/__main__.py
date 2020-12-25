# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 23:04:20 2020

@author: Kais
"""

from Checker.AI.AISystems import MiniMaxAlphaBetaSystem
from Checker.AI.Agent import Agent
from Checker.Engine import play


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


if __name__ == '__main__':
    start_game()
