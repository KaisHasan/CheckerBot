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

    def __init__(self, *, location: tuple = None, colour: str = None) -> None:
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
        if self._colour == 'white' and self._location[0] == 7:
            self.promote_to_king()
        if self._colour == 'black' and self._location[0] == 0:
            self.promote_to_king()

    def promote_to_king(self) -> None:
        """Promote the disk to be a king.

        Returns
        -------
        None.

        """
        if self._king:
            return
        self._king = True
        self._directions = *self._white_directions, *self._black_directions

    def is_enemy(self, other) -> bool:
        """Use it to find if a given disk is enemy with the current disk.

        Parameters
        ----------
        other : Disk

        Raises
        ------
        TypeError
            if the type of other is not Disk.

        Returns
        -------
        bool
            True if other have same colour as current disk, False otherwise.

        """
        if not isinstance(other, type(self)):
            raise TypeError('Argument must be of type Disk')
        return self._colour != other.get_colour()

    def __hash__(self):
        return hash((self._location, self._king, self._colour))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return self._colour == other.get_colour() and \
            self._king == other.is_king() and \
            self._location == other.get_location()


class Board:
    """Represents the CheckerBoard."""

    def __init__(self, white_disks: set, black_disks: set) -> None:
        """

        Parameters
        ----------
        white_disks : set
            contains all white disks.
            type of each element is Disk.
        black_disks : set
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

        # Initialize dictionary with:
        # location as key and disk at that location as value.
        self._disks_at = self.build_disks_at()

    def get_number_of_kings(self, colour: str) -> int:
        """Use it to get the number of kings with a given colour.

        Parameters
        ----------
        colour : str
            the colour of the kings.

        Raises
        ------
        ValueError
            if the colour is neither 'white', nor 'black'.

        Returns
        -------
        int
            number of kings with the given colour.

        """
        if colour not in ['white', 'black']:
            raise ValueError("colour must be either 'white' or 'black'!")
        num = 0
        for disk in self._disks[colour]:
            if disk.is_king() and disk.colour == colour:
                num += 1

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
            return self.get_number_of_disks('white') + self.get_number_of_disks('black')
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
        """Use it to get the disk given the location.

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

    def get_disks(self, colour: str = None) -> set:
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
        set

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
            return self.get_number_of_disks(colour) == 0
        else:
            raise ValueError("""colour must be
                             "black", or "white", or "None"! """)

    def remove_disk_at(self, location: tuple) -> None:
        """Use it to remove a disk with a given location from the board.

        Parameters
        ----------
        location : tuple
            location at the board of disk which we want to remove.

        Raises
        ------
        ValueError
            if location does not contain any Disk.

        Returns
        -------
        None

        """
        if location not in self._disks_at:
            raise KeyError('location is not found!')
        d = self._disks_at[location]
        self._disks[d.get_colour()].discard(d)


if __name__ == '__main__':
    d1 = Disk(location=(0, 0), colour='white')
    d2 = Disk(location=(0, 0), colour='white')
    assert(d1 == d2)
    assert(hash(d1) == hash(d2))
    d1.set_location((7, 1))
    assert(d1 != d2)
    assert(hash(d1) != hash(d2))
    assert(d1.is_king() is True)
    d2.set_colour('black')
    assert(d1.is_enemy(d2) is True)
    d1 = Disk(location=(1, 1), colour='white')
    d2 = Disk(location=(2, 1), colour='white')
    d3 = Disk(location=(3, 1), colour='black')
    d4 = Disk(location=(3, 3), colour='black')
    white_disks = set([d1, d2])
    black_disks = set([d3, d4])
    b = Board(white_disks, black_disks)
    assert(b.get_number_of_disks() == 4)
    assert(b.get_number_of_disks('white') == 2)
    assert(b.get_number_of_disks('black') == 2)
    assert(b.get_number_of_kings('white') == 0)
    assert(b.get_number_of_kings('black') == 0)
    b.remove_disk_at((1, 1))
    assert(b.get_number_of_disks('white') == 1)
    b.remove_disk_at((3, 3))
    assert(b.get_number_of_disks('black') == 1)
    assert(b.is_empty('white') is False)
    b.remove_disk_at((2, 1))
    assert(b.get_number_of_disks('white') == 0)
    assert(b.is_empty('white') is True)
    print('Everything work.')
