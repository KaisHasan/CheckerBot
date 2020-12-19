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
    colour = {1:'white', 0:'black'}
    original_colour = agent.get_colour()
    results = []
    for i in range(num_of_games):
        boards = []  # list of Agent board positions through the game.
        current_board = initial_board_generator()
        final_status = None
        for turn in range(1, Board.draw_turn_number + 1):
            agent.set_colour(colour[turn % 2])
            if colour[turn % 2] == original_colour:
                boards.append(current_board.copy())
            status = current_board.get_status(original_colour, turn)
            if output is True:
                print(f'turn: {turn}')
                cli.show(current_board)
            if status is not None:
                final_status = status
                break
            current_board = agent.choose_board(current_board)
        agent.learn(boards.copy(), final_status)
        costs.append(agent.get_system().compute_error(boards,
                                                      agent.get_colour(),
                                                      final_status))
        results.append(d[final_status])
    return costs, results


def Train_with_other_agent(agent: Agent, num_of_games: int, output: bool,
          other_agent = None) -> list:
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
    if other_agent is None:
        other_agent = agent.copy()
        enemy_colour = 'black'
        if enemy_colour == agent.get_colour():
            enemy_colour = 'white'
        other_agent.set_colour(enemy_colour)

    costs = []
    cli = CLI()
    # -1 is lose
    #  1 is win
    #  0 is draw
    d = {'win': 1, 'lose': -1, 'draw': 0}
    colour = {1:'white', 0:'black'}
    results = []
    for i in range(num_of_games):
        boards = []  # list of Agent board positions through the game.
        current_board = initial_board_generator()
        final_status = None
        for turn in range(1, Board.draw_turn_number + 1):
            if colour[turn%2] == agent.get_colour():
                boards.append(current_board.copy())
            status = current_board.get_status(agent.get_colour(), turn)
            if output is True:
                print(f'turn: {turn}')
                cli.show(current_board)
            if status is not None:
                final_status = status
                break
            if colour[turn%2] == agent.get_colour():
                current_board = agent.choose_board(current_board)
            else:
                current_board = other_agent.choose_board(current_board)
        agent.learn(boards.copy(), final_status)
        costs.append(agent.get_system().compute_error(boards,
                                                      agent.get_colour(),
                                                      final_status))
        results.append(d[final_status])
    return costs, results, other_agent

def Play(agent: Agent):
    def get_board(board, loc1, loc2):
        boards = Moves.get_next_boards(board, loc1)
        for b in boards:
            if b.get_disk_at(loc2) is not None:
                return b
        return None
    cli = CLI()
    current_board = initial_board_generator()
    colour = {0:'white', 1:'black'}
    final_status = None
    for turn in range(1, 200):
        status = current_board.get_status(agent.get_colour(), turn)
        print(f'turn: {turn}')
        cli.show(current_board)
        if colour[turn % 2] == agent.get_colour():
            print('computer move')
        else:
            print('human move')
        if status is not None:
            final_status = status
            break
        if colour[turn % 2] == agent.get_colour():
            current_board = agent.choose_board(current_board)
        else:
            while True:
                try:
                    print('source:')
                    loc1 = cli.get_location()
                    if current_board.get_disk_at(loc1) is not None:
                        break
                except ValueError as err:
                    print(err)
            while True:
                try:
                    print('destination:')
                    loc2 = cli.get_location()
                    if get_board(current_board, loc1, loc2) is not None:
                        break
                except ValueError as err:
                    print(err)
            current_board = get_board(current_board, loc1, loc2)
    print(f'game is {final_status} for white!')


def play_with_other_agent(agent1: Agent, agent2: Agent,
                          output: bool = False) -> str:
    if agent1.get_colour() == agent2.get_colour():
        return "the two agents have the same disk's colour"
    final_status = None
    colour = {1:'white', 0:'black'}
    cli = CLI()
    current_board = initial_board_generator()
    for turn in range(1, Board.draw_turn_number + 1):
        status = current_board.get_status(agent1.get_colour(), turn)
        if output is True:
            print(f'turn: {turn}')
            cli.show(current_board)
        if status is not None:
            final_status = status
            break
        if colour[turn % 2] == agent1.get_colour():
            current_board = agent1.choose_board(current_board)
        else:
            current_board = agent2.choose_board(current_board)
    return final_status