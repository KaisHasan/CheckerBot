# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 20:25:53 2020

@author: Kais
"""

from Checker.AI.AISystems import AISystem, FeaturesBasedSystem
from Checker.Game import Board, Disk, Moves
import numpy as np


class Agent:
    """AI Agent.

    This agent use one of the AISystems.
    you can let the agent choose between a list of boards using the method
    choose_board.
    """

    def __init__(self, colour: str, system: AISystem) -> None:
        """Initialize the Agent.

        Parameters
        ----------
        colour : str
            the colour of the pieces controlled by the agent.
            must be either 'black', or 'white'.
        system : AISystem
            the system used by the agent to play and learn.

        Returns
        -------
        None

        """
        self.set_colour(colour)
        self._system = system

    def copy(self):
        """Use it to copy the agent.

        this will copy the system and the colour of the agent.
        for information about what happened when a system is copied see
        documentation of the copy method in AISystems.

        Returns
        -------
        Agent
            a copy of calling agent.

        """
        a = Agent(self._colour, self._system.copy())
        return a

    def get_colour(self) -> str:
        """Get the colour of the agent.

        Returns
        -------
        str
            the colour of the agent (i.e 'black', or 'white').

        """
        return self._colour

    def set_colour(self, colour: str) -> None:
        """Set the colour of the agent.

        Parameters
        ----------
        colour : str
            the colour of the pieces controlled by the agent.
            must be either 'black', or 'white'.

        Raises
        ------
        ValueError
            if the colour is not valid (i.e it's neither 'black', nor 'white').

        Returns
        -------
        None

        """
        if colour not in ['white', 'black']:
            raise ValueError("colour must be either 'white', or 'black'!")
        self._colour = colour

    def get_fitness(self, boards: list, turn: int, draw_counter: int) -> float:
        """Get the fitness value of a given board.

        this function will calculate for each board how good it is, using
        agent' system specific prediction.

        Parameters
        ----------
        boards : list
            list of boards we want to calculate their fitnesses.
        turn : int
            turn's number.
        draw_counter : int
            counter of non-attack moves.

        Returns
        -------
        float
            how good the board is (i.e the fitness value of the board).

        """
        return self._system.predict(boards, turn, draw_counter)

    def choose_board(self, board: Board, turn: int,
                     draw_counter: int) -> Board:
        """Choose the best board by applying the best move on the given board.

        Parameters
        ----------
        board : Board
            the current board.
        turn : int
            turn's number.
        draw_counter : int
            counter of non-attack moves.

        Returns
        -------
        Board
            the new board, which is the result of applying
            the best valid move on the given board.

        """
        # generate all possible boards
        boards = Moves.get_all_next_boards(board, self._colour)
        # get the fitness of every board
        values = self.get_fitness(boards, turn + 1, draw_counter)
        # get the id of the best board
        # because our evaluation function is predict how good
        # a board is for white, then when we play as 'black'
        # we must takes the minimum to put the 'white' in the worst
        # possible scenario.
        if self._colour == 'black':
            i = np.argmin(values)
        else:
            i = np.argmax(values)
        return boards[i]

    def learn(self, boards: list, final_status: str) -> None:
        """Let the agent learn using the hisory of a game.

        the agent will use its system to learn.

        Parameters
        ----------
        boards : list
            the list of all boards through the game
            where it's white turn.
        final_status : str
            the final status of the white player.
            must be 'win', 'lose', or 'draw'.

        Returns
        -------
        None

        """
        self._system.update_parameters(boards, final_status)

    def get_system(self) -> AISystem:
        """Get the system used by the agent.

        Returns
        -------
        AISystem
            the system used by the agent.

        """
        return self._system

    def get_name(self) -> str:
        """Get the name of the system used by the agent.

        Returns
        -------
        str
            the name of the system used by the agent.

        """
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
    _ = a.get_fitness([b1], 0, 0)

    b = a.choose_board(b1, 0, 0)
    print('Everything work.')
