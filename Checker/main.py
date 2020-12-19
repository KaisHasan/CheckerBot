# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 23:04:20 2020

@author: Kais
"""

from Checker.AI.AISystems import FeaturesBasedSystem
from Checker.AI.Agent import Agent
from Checker.Engine import Train, Play, Train_with_other_agent, play_with_other_agent
from Checker.Game import Board
import matplotlib.pyplot as plt
import numpy as np


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

    def train_agent(agent: Agent, num_of_games: int):

        costs, results = Train(agent=agent,
                               num_of_games=num_of_games,
                               output=False)
        # Play(agent)
        # print(f'{i+1}-th training:')
        wins = results.count(1)
        loses = results.count(-1)
        draws = results.count(0)
        print(f'training results of agent {agent.get_name()}')
        print(f'wins: {wins}')
        print(f'loses: {loses}')
        print(f'draws: {draws}')
        print('##################################')
        print(costs[-1])
        print_iter = 1
        plt.plot(np.arange(0, num_of_games, print_iter),
                 costs[::print_iter], 'k')
        plt.show()

    def get_agent(name: str, learning_rate: float,
                  features: list,
                  use_saved_parameters: bool = True) -> Agent:
        system = FeaturesBasedSystem(name, learning_rate,
                                     use_saved_parameters, *features)
        agent = Agent(colour='white', system=system)
        return agent

    my_agent_features = features.copy()
    my_agent = get_agent('my_agent', 0.03, my_agent_features)

    my_agent1_features = features.copy()
    my_agent1 = get_agent('my_agent1', 0.03, my_agent1_features)

    train_agent(my_agent1, 10)

    num_of_games = 10
    d = {'win': 1, 'lose': -1, 'draw': 0}
    print('######################################################')
    print('status when: my_agent is "white", my_agent1 is "black"')
    print('######################################################')
    my_agent1.set_colour('black')
    results = []
    for i in range(num_of_games):
        result = play_with_other_agent(my_agent, my_agent1, False)
        results.append(d[result])
    wins = results.count(1)
    loses = results.count(-1)
    draws = results.count(0)
    print(f'wins: {wins}')
    print(f'loses: {loses}')
    print(f'draws: {draws}')
    print('######################################################')
    print('status when: my_agent is "black", my_agent1 is "white"')
    print('######################################################')
    my_agent.set_colour('black')
    my_agent1.set_colour('white')
    results = []
    for i in range(num_of_games):
        result = play_with_other_agent(my_agent, my_agent1)
        results.append(d[result])
    wins = results.count(1)
    loses = results.count(-1)
    draws = results.count(0)
    print(f'wins: {wins}')
    print(f'loses: {loses}')
    print(f'draws: {draws}')
# =============================================================================
#     my_agent = get_agent('my_agent', 0.03)
#     my_agent.set_colour('black')
#     Play(my_agent)
# =============================================================================
