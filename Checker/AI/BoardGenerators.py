# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 15:02:40 2020

@author: Kais
"""

from Checker.Game import Board, Disk


def initial_board_generator() -> Board:
    """Generate an initial board.

    Returns
    -------
    Board

    """
    white_disks = [(0, 0), (0, 2), (0, 4), (0, 6),
                   (1, 1), (1, 3), (1, 5), (1, 7),
                   (2, 0), (2, 2), (2, 4), (2, 6)]
    black_disks = []
    for location in white_disks:
        new_location = (7-location[0], 7-location[1])
        black_disks.append(new_location)
    for i, loc in enumerate(white_disks.copy()):
        white_disks[i] = Disk(location=loc, colour='white')
    for i, loc in enumerate(black_disks.copy()):
        black_disks[i] = Disk(location=loc, colour='black')
    b = Board(set(white_disks), set(black_disks))
    return b
