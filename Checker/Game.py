# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 20:28:10 2020

@author: Kais
"""


class Disk:
    """Class represents Disks on the CheckerBoard."""

    white_directions = ((1, 1), (1, -1))
    black_directions = ((-1, -1), (-1, 1))

    def __init__(self, *, location: tuple, colour: str):
        """Constructor for Disk object.

        Parameters
        ----------
        tuple : location, optional
            Location of the Disk in the Board. The default is None.
        str : colour, optional
            Colour of the Disk ('white' or 'black'). The default is None.

        Returns
        -------
        None.

        """
        self._location = location
        self._king = False  # True if the Disk have promoted to King.
        self.set_colour(colour)
        if self._colour == 'white':
            self._directions = self.white_directions
        elif self._colour == 'black':
            self._directions = self.black_directions

    def set_colour(self, colour: str) -> None:
        if type(colour) != type(' '):
            raise TypeError('colour must be of type str!')
        if colour.lower() in ['white', 'black']:
            self._colour = colour.lower()
        else:
            raise ValueError('Colour must be either "white" or "black"!')

    def get_colour(self) -> str:
        return self._colour

    def get_directions(self) -> tuple:
        return self._directions

    def get_location(self) -> tuple:
        return self._location

    def is_king(self) -> bool:
        return self._king

    def set_location(self, location: tuple) -> None:
        self._location = location

    def promote_to_king(self):
        self._king = True
        self._directions = *self.white_directions, *self.black_directions


