# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 23:04:20 2020

@author: Kais
"""

from Checker.AI.AISystems import FeaturesBasedSystem
from Checker.AI.Agent import Agent
from Checker.Engine import train, play, play_with_other_agent
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

    def train_agent(agent: Agent, num_of_games: int, output: bool= False):

        costs, results = Train(agent,
                               num_of_games,
                               output)
        wins = results.count(1)
        loses = results.count(-1)
        draws = results.count(0)
        print(f'training results of agent {agent.get_name()}')
        print(f'wins: {wins}')
        print(f'loses: {loses}')
        print(f'draws: {draws}')
        print('##################################')
        print(f'cost: {costs[-1]}')
        print_iter = num_of_games // 10
        plt.plot(np.arange(0, num_of_games, print_iter + 1),
                 costs[::print_iter+1], 'k')
        plt.show()

    def get_agent(name: str, learning_rate: float,
                  features: list,
                  use_saved_parameters: bool = True) -> Agent:
        system = FeaturesBasedSystem(name, learning_rate,
                                     use_saved_parameters, *features)
        agent = Agent(colour='white', system=system)
        return agent

    def test_agents(agent1: Agent, agent2: Agent, num_of_games: int):
        d = {'win': 1, 'lose': -1, 'draw': 0}
        print('######################################################')
        print(f'status for {agent1.get_name()}')
        print(f'{agent1.get_name()}: "white", {agent2.get_name()}: "black"')
        print('######################################################')
        agent1.set_colour('white')
        agent2.set_colour('black')
        results = []
        for i in range(num_of_games):
            result = play_with_other_agent(agent1, agent2, False)
            results.append(d[result])
        wins = results.count(1)
        loses = results.count(-1)
        draws = results.count(0)
        print(f'wins: {wins}')
        print(f'loses: {loses}')
        print(f'draws: {draws}')
        print('######################################################')
        print(f'status for {agent1.get_name()}')
        print(f'{agent1.get_name()}: "black", {agent2.get_name()}: "white"')
        print('######################################################')
        agent1.set_colour('black')
        agent2.set_colour('white')
        results = []
        for i in range(num_of_games):
            result = play_with_other_agent(agent1, agent2)
            results.append(d[result])
        wins = results.count(1)
        loses = results.count(-1)
        draws = results.count(0)
        print(f'wins: {wins}')
        print(f'loses: {loses}')
        print(f'draws: {draws}')

    agent1_features = features.copy()
    agent1 = get_agent('agent1', 0.0001, agent1_features)

    # train_agent(agent1, 1000)
    # print(agent1.get_system()._parameters)
    tom_agent = get_agent('tom_agent', 0.001, [])

    # train_agent(tom_agent, 100)

    test_agents(agent1, tom_agent, 100)

    # Play(tom_agent)
