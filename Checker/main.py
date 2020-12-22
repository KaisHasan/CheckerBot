# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 23:04:20 2020

@author: Kais
"""

from Checker.AI.AISystems import FeaturesBasedSystem
from Checker.AI.Agent import Agent
from Checker.Engine import train, play, play_with_other_agent
from Checker.Game import Board


def num_on_the_left_white(board: Board) -> float:
    num = 0
    for i in range(0, 7 + 1, 2):
        disk = board.get_disk_at((i, 0))
        if disk is not None and disk.get_colour() == 'white':
            num += 1
    return num


def num_on_the_left_black(board: Board) -> float:
    num = 0
    for i in range(0, 7 + 1, 2):
        disk = board.get_disk_at((i, 0))
        if disk is not None and disk.get_colour() == 'black':
            num += 1
    return num


def num_on_the_right_white(board: Board) -> float:
    num = 0
    for i in range(1, 7 + 1, 2):
        disk = board.get_disk_at((i, 7))
        if disk is not None and disk.get_colour() == 'white':
            num += 1
    return num


def num_on_the_right_black(board: Board) -> float:
    num = 0
    for i in range(1, 7 + 1, 2):
        disk = board.get_disk_at((i, 7))
        if disk is not None and disk.get_colour() == 'black':
            num += 1
    return num


def num_on_the_down_white(board: Board) -> float:
    num = 0
    for j in range(0, 7 + 1, 2):
        disk = board.get_disk_at((0, j))
        if disk is not None and disk.get_colour() == 'white':
            num += 1
    return num


def num_on_the_down_black(board: Board) -> float:
    num = 0
    for j in range(0, 7 + 1, 2):
        disk = board.get_disk_at((0, j))
        if disk is not None and disk.get_colour() == 'black':
            num += 1
    return num


def num_on_the_up_white(board: Board) -> float:
    num = 0
    for j in range(1, 7 + 1, 2):
        disk = board.get_disk_at((7, j))
        if disk is not None and disk.get_colour() == 'white':
            num += 1
    return num


def num_on_the_up_black(board: Board) -> float:
    num = 0
    for j in range(1, 7 + 1, 2):
        disk = board.get_disk_at((7, j))
        if disk is not None and disk.get_colour() == 'black':
            num += 1
    return num


features = [
        num_on_the_left_white,
        num_on_the_left_black,
        num_on_the_right_white,
        num_on_the_right_black,
        num_on_the_down_white,
        num_on_the_down_black,
        num_on_the_up_white,
        num_on_the_up_black
    ]


if __name__ == '__main__':

    def get_agent(name: str, learning_rate: float,
                  features: list,
                  use_saved_parameters: bool = True) -> Agent:
        system = FeaturesBasedSystem(name, learning_rate,
                                     use_saved_parameters, *features)
        agent = Agent(colour='white', system=system)
        return agent

    print('Welcome to the Checker Game')
    print('please note that white is always plays first')
    print('agents available are:')
    print('agent1')
    print('tom_agent')
    while True:
        agent_name = input('enter the name of the agent to play with it:')
        agent_name = agent_name.lower()
        if agent_name in ['agent1', 'tom_agent']:
            break
        else:
            print('please enter a valid agent name!')
    print('colours available for you are: white, black')
    while True:
        colour = input('enter the colour you want to play with:')
        colour = colour.lower()
        if colour in ['white', 'black']:
            break
        else:
            print('please enter a valid colour!')
    agent_colour = 'white' if colour == 'black' else 'black'
    print(f'agent colour: {agent_colour}')
    if agent_name == 'agent1':
        agent1_features = features.copy()
        agent1 = get_agent('agent1', 0.0001, agent1_features)
        agent1.set_colour(agent_colour)
        play(agent1)
    else:
        tom_agent = get_agent('tom_agent', 0.001, [])
        tom_agent.set_colour(agent_colour)
        play(tom_agent)
