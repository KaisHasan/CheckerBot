# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 22:49:48 2020

@author: Kais
"""

from Checker.AI.Agent import Agent
from Checker.AI.BoardGenerators import initial_board_generator
from Checker.UI import CLI
from Checker.Game import Moves, update_draw_counter
import matplotlib.pyplot as plt
import numpy as np
import random


def train(agent: Agent, num_of_games: int,
          initial_game: tuple = None,
          explore_probability: float = 10,
          output: bool = False,
          results_output: bool = True,
          plots: bool = True) -> tuple:
    """Train the agent by making it plays games against its self.

    Parameters
    ----------
    agent : Agent
        The agent to train.
    num_of_games : int
        number of training games.
    initial_game : tuple
        a starting game
            first element is an initial board.
            second element is turn's number.
            third element is draw counter.
        the default is None.
    explore_probability : int
        the probability that the agent choose to explore a random
        move to explore instead of playing the optimal move he learnt so far.
        if it's None then no exploration will happened.
        the probability will be 1/explore_probability.
        the default is 0.1.
    output: bool, optional
        indicates if you want to print the game or not.
        the default is False.
    results_output: bool, optional
        indicates if you want to print the game's results or not.
        the default is True.
    plots: bool, optional
        indicates if you want to print the plots for errors
        in prediction or not.
        the default is True.

    Returns
    -------
    tuple
        first element, list:
            i-th element is the error after updating parameters using i-th game
        second element, list:
            i-th element is the result for on the i-th game.

    """
    costs = []
    cli = CLI()  # object to use the console interface if output is True.
    # dictionary for convert status into numbers.
    d = {'win': 1, 'lose': -1, 'draw': 0}
    # dictionary for knowing the colour of the current player
    # 1 is the first player
    # 0 is the second player
    colour = {1: 'white', 0: 'black'}
    # used to restore the agent colour after training end
    # thats because we are updating evaluation based on white position.
    previous_colour = agent.get_colour()
    results = []  # store the results of the games.
    # the probability that the agent will choose random move.
    prob = explore_probability
    data_set = {'board': [], 'f_status': []}
    for i in range(num_of_games):
        boards = []  # list of white board positions through the game.
        if initial_game is None:
            # generate the initial board
            current_board = initial_board_generator()
            start_turn = 1
            draw_counter = 0
        else:
            current_board, start_turn, draw_counter = initial_game
        # 2-tuple
        # first element is the status (i.e 'win', 'lose', or draw).
        # second element is the colour of the player with the final status.
        final_status = (None, None)

        for turn in range(start_turn, 10000):
            # alternate the roles of the same agent by setting its colour
            # to the colour of the player who should move in the current turn
            agent.set_colour(colour[turn % 2])
            # store the boards when its white turn.
            if colour[turn % 2] == 'white':
                boards.append(current_board)
            # get status of the game, 'None' indicates that game is still going
            status = current_board.get_status(colour[turn % 2], draw_counter)

            # print the turn number and the board if we need them.
            if output is True:
                print(f'turn: {turn}')
                cli.show(current_board)

            # if the game ends set the right values to final_status
            # then exit the game
            if status is not None:
                final_status = (status, colour[turn % 2])
                break

            # number of disks before the move:
            size_before = current_board.get_number_of_disks(None)

            # get the next board
            # this done by letting the agent choose the move.
            if prob is not None:
                r = random.randint(1, prob)
            if prob is not None and r == 1:
                next_boards = []
                Moves.get_all_next_boards(current_board, colour[turn % 2],
                                          next_boards)
                current_board = random.choice(next_boards)
            else:
                current_board = agent.choose_board(current_board,
                                                   turn, draw_counter)

            # number of disks before the move:
            size_after = current_board.get_number_of_disks(None)

            draw_counter = update_draw_counter(draw_counter, size_before,
                                               size_after)

        # get the final status for white
        if final_status[1] == 'black' and final_status[0] != 'draw':
            final_status = ('win' if final_status[0] == 'lose' else 'lose',
                            final_status[1])
        # add the cost before updating.
        costs.append(agent.get_system().compute_error(boards, final_status[0]))
        # let the agent learn using the boards positions through the game.
        # agent.learn(boards, final_status[0])
        # add the result of the game for white player
        results.append(d[final_status[0]])
        data_set['board'].append(boards)
        data_set['f_status'].append(final_status[0])

        if (i+1) % 100 == 0:
            print(f'{i+1} games finished unitl now!')
    # reset the colour of the agent
    agent.set_colour(previous_colour)
    # let the agent learn from the previous games
    m = min(results.count(-1), results.count(1))
    lose_m = m
    win_m  = m
    for i in range(num_of_games):
        bs = data_set['board'][i]
        fs = data_set['f_status'][i]
        if fs == 'win' and win_m > 0:
            agent.learn(bs, fs)
            win_m -= 1
        elif fs == 'lose' and lose_m > 0:
            agent.learn(bs, fs)
            lose_m -= 1
        elif fs == 'draw':
            agent.learn(bs, fs)
    if results_output is True:
        wins = results.count(1)
        loses = results.count(-1)
        draws = results.count(0)
        print(f'training results of agent {agent.get_name()}')
        print(f'wins: {wins}')
        print(f'loses: {loses}')
        print(f'draws: {draws}')
        print('##################################')
        print(f'cost: {costs[-1]}')
    if plots is True:
        print_iter = num_of_games // num_of_games
        plt.plot(np.arange(0, num_of_games, print_iter + 1),
                 costs[::print_iter+1], 'k')
        plt.show()
    return costs, results


def play(agent: Agent, train: bool = False) -> None:
    """Start a game between a human user and the agent.

    Note: white is always the first player but the agent can play with
    any valid colour (i.e 'white', or 'black') and the human use will
    take the other one.

    Parameters
    ----------
    agent : Agent
        the agent you want to play with.
    train : bool, optional
        indicates if you want to train the agent using the game you played.
        The default is False.

    Returns
    -------
    None

    """
    def get_board(board, loc1, loc2):
        boards = []
        Moves.get_next_boards(board, loc1, boards)
        for b in boards:
            if b.get_disk_at(loc2) is not None:
                return b
        return None
    cli = CLI()
    current_board = initial_board_generator()
    colour = {1: 'white', 0: 'black'}
    final_status = (None, None)
    boards = []
    # counter of non-attack moves.
    draw_counter = 0
    for turn in range(1, 10000):
        status = current_board.get_status(colour[turn % 2], draw_counter)
        print(f'turn: {turn}')
        cli.show(current_board)
        if colour[turn % 2] == 'white':
            boards.append(current_board)
        if colour[turn % 2] == agent.get_colour():
            print('computer move')
        else:
            print('human move')
        if status is not None:
            final_status = (status, colour[turn % 2])
            break

        # number of disks before the move:
        size_before = current_board.get_number_of_disks(None)

        if colour[turn % 2] == agent.get_colour():
            current_board = agent.choose_board(current_board,
                                               turn, draw_counter)
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

            # number of disks before the move:
            size_after = current_board.get_number_of_disks(None)

            draw_counter = update_draw_counter(draw_counter, size_before,
                                               size_after)
    temp_status = final_status[0]
    if final_status[1] != 'white':
        temp_status = 'win' if temp_status == 'lose' else 'lose'
    if train is True:
        agent.learn(boards, temp_status)
        print('cost:', agent.get_system().compute_error(boards,
                                                        final_status[0]))
    print(f'game is {final_status[0]} for {final_status[1]}!')


def play_with_other_agent(agent1: Agent, agent2: Agent,
                          output: bool = False) -> str:
    """Let two agents play together.

    Parameters
    ----------
    agent1 : Agent
        the first agent.
    agent2 : Agent
        the seocnd agent.
    output : bool, optional
        indicates if you want to print the game or not. The default is False.

    Returns
    -------
    str
        the status of the first agent on the game.
        or "the two agents have the same disk's colour"
                            if the colours are the same.

    """
    if agent1.get_colour() == agent2.get_colour():
        return "the two agents have the same disk's colour"
    # 2-tuple
    # first element is the status (i.e 'win', 'lose', or draw).
    # second element is the colour of the player with the final status.
    final_status = (None, None)
    colour = {1: 'white', 0: 'black'}
    cli = CLI()
    # counter of non-attack moves.
    draw_counter = 0
    current_board = initial_board_generator()
    for turn in range(1, 10000):
        status = current_board.get_status(colour[turn % 2], draw_counter)
        if output is True:
            print(f'turn: {turn}')
            cli.show(current_board)

        # if the game ends set the right values to final_status
        # then exit the game
        if status is not None:
            final_status = (status, colour[turn % 2])
            break

        # number of disks before the move:
        size_before = current_board.get_number_of_disks(None)

        if colour[turn % 2] == agent1.get_colour():
            current_board = agent1.choose_board(current_board,
                                                turn, draw_counter)
        else:
            current_board = agent2.choose_board(current_board,
                                                turn, draw_counter)

        # number of disks before the move:
        size_after = current_board.get_number_of_disks(None)

        draw_counter = update_draw_counter(draw_counter, size_before,
                                           size_after)

    # get the final status for the first player
    if final_status[1] != agent1.get_colour() and final_status[0] != 'draw':
        final_status = ('win' if final_status[0] == 'lose' else 'lose',
                        final_status[1])
    return final_status[0]


def test_agents(agent1: Agent, agent2: Agent, num_of_games: int,
                output: bool = False):
    """Use this function to let two agents play together.

    this function will print the status of the games.
    the agent1 whill take 'white' first, then 'black'.
    so the total number of games will be 2*num_of_games.

    Parameters
    ----------
    agent1 : Agent
        the first agent.
    agent2 : Agent
        the second.
    num_of_games : int
        number of test games.
    output : bool, optional
        indicates if you want to print the game or not. The default is False.

    Returns
    -------
    None.

    """
    d = {'win': 1, 'lose': -1, 'draw': 0}
    print('######################################################')
    print(f'status for {agent1.get_name()}')
    print(f'{agent1.get_name()}: "white", {agent2.get_name()}: "black"')
    print('######################################################')
    agent1.set_colour('white')
    agent2.set_colour('black')
    results = []
    for i in range(num_of_games):
        result = play_with_other_agent(agent1, agent2, output)
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
        result = play_with_other_agent(agent1, agent2, False)
        results.append(d[result])
    wins = results.count(1)
    loses = results.count(-1)
    draws = results.count(0)
    print(f'wins: {wins}')
    print(f'loses: {loses}')
    print(f'draws: {draws}')
