# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 22:49:48 2020

@author: Kais
"""

from Checker.AI.Agent import Agent
from Checker.AI.BoardGenerators import initial_board_generator
from Checker.UI import CLI
from Checker.Game import Moves, Board

def Train(agent: Agent, num_of_games: int, output: bool) -> list:
    """Train the agent by making it plays games agiant its self.

    Parameters
    ----------
    agent : Agent
        The agent to train.
    num_of_games : int
        number of training games.
    output: bool
        indicates if you want to print the game or not.

    Returns
    -------
    list
        cost for each game

    """
    costs = []
    cli = CLI()
    # -1 is lose
    #  1 is win
    #  0 is draw
    d = {'win': 1, 'lose': -1, 'draw': 0}
    results = []
    for i in range(num_of_games):
        boards = []  # list of Agent board positions through the game.
        current_board = initial_board_generator()
        boards.append(current_board)
        current_colour = 'white'
        final_status = None
        for turn in range(1, 250 + 1):
            status = current_board.get_status(current_colour, turn)
            if output is True:
                print(f'turn: {turn}')
                cli.show(current_board)
            if status is not None:
                final_status = status
                break
            agent.set_colour(current_colour)
            current_board = agent.choose_board(current_board)
            if current_colour == 'white':
                boards.append(current_board)
            current_colour = 'white' if current_colour == 'black' else 'black'
        agent.set_colour('white')
        agent.learn(boards, final_status)
        costs.append(agent.get_system().compute_error(boards, 'white',
                                                      final_status))
        results.append(d[final_status])
    return costs, results

def Play(agent: Agent):
    def get_board(board, loc1, loc2):
        boards = Moves.get_next_boards(board, loc1)
        for b in boards:
            if b.get_disk_at(loc2) is not None:
                return b
        return None
    cli = CLI()
    current_board = initial_board_generator()
    # colour = {1:'white', 0:'black'}
    final_status = None
    for turn in range(1, 200):
        status = current_board.get_status('white', turn)
        print(f'turn: {turn}')
        cli.show(current_board)
        if turn % 2 == 1:
            print('computer move')
        else:
            print('human move')
        if status is not None:
            final_status = status
            break
        if turn % 2 == 1:
            current_board = agent.choose_board(current_board)
        else:
            while True:
                loc1 = cli.get_location()
                if current_board.get_disk_at(loc1) is not None:
                    break
            while True:
                loc2 = cli.get_location()
                if get_board(current_board, loc1, loc2) is not None:
                    break
            current_board = get_board(current_board, loc1, loc2)
    print(f'game is {final_status} for white!')
