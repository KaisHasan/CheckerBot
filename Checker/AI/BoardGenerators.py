# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 15:02:40 2020

@author: Kais
"""

from Checker.Game import Board, Disk, Moves, update_draw_counter
import random


def initial_board_generator() -> Board:
    """Generate an initial board.

    Returns
    -------
    Board

    """
    black_disks = [(0, 0), (0, 2), (0, 4), (0, 6),
                   (1, 1), (1, 3), (1, 5), (1, 7),
                   (2, 0), (2, 2), (2, 4), (2, 6)]
    white_disks = []
    for location in black_disks:
        new_location = (7-location[0], 7-location[1])
        white_disks.append(new_location)
    for i, loc in enumerate(white_disks.copy()):
        white_disks[i] = Disk(location=loc, colour='white')
    for i, loc in enumerate(black_disks.copy()):
        black_disks[i] = Disk(location=loc, colour='black')
    b = Board(set(white_disks), set(black_disks))
    return b


def random_game_generator() -> tuple:
    """Generate random valid board obtained by some moves from initial board.

    the maximim number of moves in the board generated is 20

    Returns
    -------
    tuple
        a random game
            first element is a random valid board.
            second element is turn's number.
            third element is draw counter.

    """
    current_board = initial_board_generator()
    num_of_turns = random.randint(0, 10)

    draw_counter = 0
    colour = {1: 'white', 0: 'black'}
    for turn in range(1, num_of_turns):
        # number of disks before the move:
        size_before = current_board.get_number_of_disks(None)

        boards = []
        Moves.get_all_next_boards(current_board, colour[turn % 2], boards)
        # get the next board
        current_board = random.choice(boards)

        # number of disks before the move:
        size_after = current_board.get_number_of_disks(None)

        draw_counter = update_draw_counter(draw_counter, size_before,
                                           size_after)

    return current_board, num_of_turns, draw_counter


def generate_random_training_set() -> tuple:
    colour = {1: 'white', 0: 'black'}
    current_board = initial_board_generator()
    boards = []  # list of white board positions through the game.
    # 2-tuple
    # first element is the status (i.e 'win', 'lose', or draw).
    # second element is the colour of the player with the final status.
    final_status = (None, None)
    start_turn = 1
    draw_counter = 0
    for turn in range(start_turn, 10000):
        # store the boards where its white turn.
        if colour[turn % 2] == 'white':
            boards.append(current_board)
        # get status of the game, 'None' indicates that game is still going
        status = current_board.get_status(colour[turn % 2], draw_counter)

        # if the game ends set the right values to final_status
        # then exit the game
        if status is not None:
            final_status = (status, colour[turn % 2])
            break

        # number of disks before the move:
        size_before = current_board.get_number_of_disks(None)

        # get the next board
        # this done by letting the agent choose the move.
        next_boards = []
        Moves.get_all_next_boards(current_board, colour[turn % 2],
                                  next_boards)
        current_board = random.choice(next_boards)

        # number of disks before the move:
        size_after = current_board.get_number_of_disks(None)

        draw_counter = update_draw_counter(draw_counter, size_before,
                                           size_after)

    # get the final status for white
    if final_status[1] == 'black' and final_status[0] != 'draw':
        final_status = ('win' if final_status[0] == 'lose' else 'lose',
                        final_status[1])
    return boards, final_status[0]
