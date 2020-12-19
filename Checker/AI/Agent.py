# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 20:25:53 2020

@author: Kais
"""

from Checker.AI.AISystems import AISystem, FeaturesBasedSystem
from Checker.Game import Board, Disk, Moves
import numpy as np


class Agent:
    def __init__(self, colour: str, system: AISystem) -> None:
        self.set_colour(colour)
        self._system =  system

    def copy(self):
        a = Agent(self._colour, self._system.copy())
        return a

    def get_colour(self) -> str:
        return self._colour

    def set_colour(self, colour: str) -> None:
        if colour not in ['white', 'black']:
            raise ValueError("colour must be either 'white', or 'black'!")
        self._colour = colour

    def get_fitness(self, board: Board) -> float:
        return self._system.predict(board)

    def choose_board(self, board: Board) -> Board:
        boards = Moves.get_all_next_boards(board, self._colour)
        # get the fitness of every board
        values = list(map(self.get_fitness, boards))
        # get the id of the best board
        i = np.argmax(values)
        return boards[i]

    def learn(self, boards: list, final_status: str) -> None:
        self._system.update_parameters(boards, self._colour, final_status)

    def get_system(self):
        return self._system

    def get_name(self) -> str:
        return self._system.get_name()

if __name__ == '__main__':
    system = FeaturesBasedSystem('test', learning_rate=0.01,
                                 useSavedParameters=True)
    a = Agent('white', system)
    assert(a.get_colour() == 'white')

    white_disks = [(0, 4), (0, 6), (1, 5), (1, 7),
                   (2, 0), (2, 4), (3, 5)]
    black_disks = [(3, 1), (4, 4), (5, 5), (6, 0),
                   (6, 2), (6, 4), (6, 6)]
    for i, loc in enumerate(white_disks.copy()):
        white_disks[i] = Disk(location=loc, colour='white')
    for i, loc in enumerate(black_disks.copy()):
        black_disks[i] = Disk(location=loc, colour='black')
    b1 = Board(set(white_disks), set(black_disks))
    _ = a.get_fitness(b1)

    b = a.choose_board(b1)
    print('Everything work.')
