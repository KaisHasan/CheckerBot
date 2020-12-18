# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 23:04:20 2020

@author: Kais
"""

from Checker.AI.AISystems import FeaturesBasedSystem
from Checker.AI.Agent import Agent
from Checker.Engine import Train, Play
from Checker.Game import Board
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
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
    name = 'test'
    learning_rate = 0.001
    use_saved_parameters = False
    system = FeaturesBasedSystem(name, learning_rate,
                                 use_saved_parameters, *features)
    agent = Agent(colour='white', system=system)
    num_of_games = 1
    costs, results = Train(agent=agent, num_of_games=num_of_games,
                           output=False)
    # Play(agent)
    wins = results.count(1)
    loses = results.count(-1)
    draws = results.count(0)
    print(f'wins: {wins}')
    print(f'loses: {loses}')
    print(f'draws: {draws}')
    print(costs[-1])
    plt.plot(np.arange(1, num_of_games + 1), costs, 'xk-')
    plt.show()
