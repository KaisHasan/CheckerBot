# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 20:28:10 2020

@author: Kais
"""


class Disk:
    """Represents Disks on the CheckerBoard."""

    # Valid Directions of a non-king white Disk.
    _white_directions = ((1, 1), (1, -1))
    # Valid Directions of a non-king black Disk.
    _black_directions = ((-1, -1), (-1, 1))

    def __init__(self, *, location: tuple, colour: str) -> None:
        """

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
        self._king = False  # True if the Disk is King.
        self.set_colour(colour)
        # Initialize directions corresponding to the Disk's colour.
        if self._colour == 'white':
            self._directions = self._white_directions
        elif self._colour == 'black':
            self._directions = self._black_directions

    def set_colour(self, colour: str) -> None:
        """Set the colour of the disk.

        Parameters
        ----------
        colour : str
            colour must be either 'black' or 'white'.

        Raises
        ------
        TypeError
            if colour is not string.
        ValueError
            if colour is neither 'black' nor 'white'.

        Returns
        -------
        None

        """
        if type(colour) != type(' '):
            raise TypeError('colour must be of type str!')
        if colour.lower() in ['white', 'black']:
            self._colour = colour.lower()
        else:
            raise ValueError('Colour must be either "white" or "black"!')

    def get_colour(self) -> str:
        """Use it to get the colour of the disk.

        Returns
        -------
        str
            The colour of the disk

        """
        return self._colour

    def get_directions(self) -> tuple:
        """Use it to get the valid move's directions of the disk.

        Note that this will not take the board borders into account.

        Returns
        -------
        tuple
            the valid move's directions of the disk.

        """
        return self._directions

    def get_location(self) -> tuple:
        """Use it to get the current location of the Disk on the board.

        Returns
        -------
        tuple
            current location of the disk.

        """
        return self._location

    def is_king(self) -> bool:
        """Use it to know if the disk is a king.

        Returns
        -------
        bool
            True if the disk is a king, False otherwise.

        """
        return self._king

    def set_location(self, location: tuple) -> None:
        """Change the current location of the disk.

        Parameters
        ----------
        location : tuple
            the new location.

        Returns
        -------
        None

        """
        self._location = location

    def promote_to_king(self):
        """Promote the disk to be a king.

        Returns
        -------
        None.

        """
        self._king = True
        self._directions = *self._white_directions, *self._black_directions


class Board:
    """Represents the CheckerBoard."""

    def __init__(self, white_disks: list, black_disks: list) -> None:
        """

        Parameters
        ----------
        white_disks : list
            contains all white disks.
            type of each element is Disk.
        black_disks : list
            contains all black disks.
            type of each element is Disk.

        Returns
        -------
        None

        """
        # Initialize _disks dictionary.
        self._disks = dict()
        self._disks['white'] = white_disks
        self._disks['black'] = black_disks

        # Initialize _kings which contains the number of kings in each colour
        self._kings = {'white': self.get_number_of_kings('white'),
                       'black': self.get_number_of_kings('black')}

        # Initialize dictionary with:
        # location as key and disk at that location as value.
        self._disks_at = self.build_disks_at()

    def get_number_of_kings(self, colour: str) -> int:
        """Use it to get the number of kings with a given colour.

        Parameters
        ----------
        colour : str
            the colour of the kings.

        Returns
        -------
        int
            number of kings with the given colour.

        """
        num = 0
        for disk in self._disks[colour]:
            num += disk.colour == colour

        return num

    def get_number_of_disks(self, colour: str = None) -> int:
        """Use it to get the number of disks with a given colour.

        if colour = None is used then it will return the number of all disks.

        Parameters
        ----------
        colour : str, optional
            the colour of the disks. The default is None.

        Raises
        ------
        ValueError
            if the colour is nor None, nor 'white', nor 'black'.

        Returns
        -------
        int
            number of disks with the specified colour, or all disks if None.

        """
        if colour is None:
            return self.get_number_of_disks() + self.get_number_of_disks('black')
        elif colour == 'white':
            return len(self._disks['white'])
        elif colour == 'black':
            return len(self._disks['black'])
        else:
            raise ValueError("""colour must be
                             "black", or "white", or "None"! """)

    def build_disks_at(self) -> dict:
        """Build a dictionary.

        Returns
        -------
        dict
            location as key, and disk at that location as value.

        """
        d = dict()
        # Get the white disks.
        for disk in self._disks['white']:
            loc = disk.get_location()
            d[loc] = disk

        # Get the black disks.
        for disk in self._disks['black']:
            loc = disk.get_location()
            d[loc] = disk

        return d

    def get_disk_at(self, location: tuple) -> Disk:
        """Use it to get the disk in some location.

        Parameters
        ----------
        location : tuple
            some location on the board.

        Returns
        -------
        Disk
            The disk at the given location if there is any, None otherwise.

        """
        if location in self._disks:
            return self._disks[location]
        return None

    def get_disks(self, colour: str = None) -> list:
        """Get all disks of a given colour, or all disks if colour is None.

        Parameters
        ----------
        colour : str, optional
            must be None, or 'white', or 'black'. The default is None.

        Raises
        ------
        ValueError
            if the colour is nor None, nor 'white', nor 'black'.

        Returns
        -------
        list

        """
        if colour is None:
            return self.get_disks('white') + self.get_disks('black')
        elif colour == 'white':
            return self._disks['white']
        elif colour == 'black':
            return self._disks['black']
        else:
            raise ValueError("""colour must be
                             "black", or "white", or "None"! """)

    def is_empty(self, colour: str = None) -> bool:
        """Use it to know if there is no disk of the given colour.

        if colour is None, then return true if 'white' & 'black' are empty

        Parameters
        ----------
        colour : str, optional
            colour must be None or 'white' or 'black'. The default is None.

        Raises
        ------
        ValueError
            if colour is nor None, nor 'white', nor 'black'.

        Returns
        -------
        bool

        """
        if colour is None or colour in ['white', 'black']:
            return self.get_number_of_disks(colour)
        else:
            raise ValueError("""colour must be
                             "black", or "white", or "None"! """)


if __name__ == '__main__':
    pass
