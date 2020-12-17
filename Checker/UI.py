# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 13:59:46 2020

@author: Kais
"""
from Checker.Game import Board


class UI:
    """User interface Abstract class, inherit it to build an interface."""

    def get_location(self) -> tuple:
        """Get a location from the user.

        Returns
        -------
        tuple
            the location entered by the user.

        """
        pass

    def show(self, board: Board) -> None:
        """Show the given board.

        Parameters
        ----------
        board : Board

        Returns
        -------
        None

        """
        pass

    def show_result(self, board: Board) -> None:
        """Show the result of the current board.

        result must be 'win', 'lose', 'draw', or None if non-final board given.

        Parameters
        ----------
        board : Board

        Returns
        -------
        None

        """
        pass


class CLI(UI):
    """Command line interface for CheckerGame."""

    def get_location(self) -> tuple:
        """Get a location from the user.

        Returns
        -------
        tuple
            the location entered by the user.

        """
        loc = input('please enter the row and column numbers (0-index): ')
        if len(loc.split()) != 2:
            raise ValueError('you must enter exactly two numbers!')
        location = tuple(loc.split())
        if max(location) > 7 or min(location) < 0:
            raise ValueError('row, and column numbers must be in range[0, 7]')
        return location

    def show(self, board: Board) -> None:
        """Show the given board.

        Parameters
        ----------
        board : Board

        Returns
        -------
        None

        """
        for i in range(8):
            for j in range(8):
                disk = board.get_disk_at((i, j))
                if disk is None:
                    print(' . ', end='')
                else:
                    out = ''
                    if disk.is_king() is True:
                        out += 'k'
                    if disk.get_colour() == 'white':
                        out += 'w'
                    else:
                        out += 'b'
                    if len(out) < 3:
                        out += ' '
                    if len(out) < 3:
                        out = ' ' + out
                    print(out, end='')
            print()
        print()

    def show_result(self, board: Board, turn: int) -> None:
        """Show the result of the current board.

        result must be 'win', 'lose', 'draw', or None if non-final board given.

        Parameters
        ----------
        board : Board
        turn: int

        Returns
        -------
        None

        """
        result = board.get_status('white', turn)
        if result is None:
            print('#########Game does not end yet!##########')
        if result != 'draw':
            result += ' for white!'
        result += 'The game is '
        print(f'######### {result} ############')
