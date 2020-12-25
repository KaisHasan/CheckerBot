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
        location = tuple(int(x) for x in loc.split())
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
        for i in reversed(range(8)):
            print(f'{i} ', end='')
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
        print('  ', end='')
        for i in range(8):
            print(f' {i} ', end='')
        print()
        print()
