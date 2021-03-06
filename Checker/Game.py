# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 20:28:10 2020

@author: Kais
"""


class Disk:
    """Represents Disks on the CheckerBoard.

    contains location, colour of the disk, and wheather it's a king,
        and the directions to which it can move.
    promotion to king is automatically done when the location has changed,
        that's includes update directions to which it can move.
    hashing, and equality test are supported.
    copy method is supported.

    """

    # Valid Directions of a non-king black Disk.
    _black_directions = ((1, 1), (1, -1))
    # Valid Directions of a non-king white Disk.
    _white_directions = ((-1, -1), (-1, 1))

    def __init__(self, *, location: tuple = None, colour: str = None) -> None:
        """Initialize the Disk.

        Parameters
        ----------
        location : tuple, optional
            Location of the Disk in the Board. The default is None.
        colour : str, optional
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

    def copy(self):
        """Use it to copy the calling disk.

        Returns
        -------
        Disk
            a copy of the calling disk.

        """
        d = Disk(location=self._location, colour=self._colour)
        d._king = self._king
        d._directions = self._directions
        return d

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
        if not isinstance(colour, str):
            raise TypeError('colour must be of type str!')
        if colour.lower() in ['white', 'black']:
            self._colour = colour.lower()
        else:
            raise ValueError('Colour must be either "white" or "black"!')

    def get_colour(self) -> str:
        """Get the colour of the disk.

        Returns
        -------
        str
            The colour of the disk.

        """
        return self._colour

    def get_directions(self) -> tuple:
        """Get the valid move's directions of the disk.

        Note that this will not take the board borders into account.

        Returns
        -------
        tuple
            the valid move's directions of the disk.

        """
        return self._directions

    def get_location(self) -> tuple:
        """Get the current location of the Disk on the board.

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

        Note: this will promote the disk to a king automatically
                if the new location indicates that the disk must promoted.

        Parameters
        ----------
        location : tuple
            the new location.

        Returns
        -------
        None

        """
        self._location = location
        if self._colour == 'white' and self._location[0] == 0:
            self.promote_to_king()
        if self._colour == 'black' and self._location[0] == 7:
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
        """Find if the given disk is enemy with the calling disk.

        Parameters
        ----------
        other : Disk

        Raises
        ------
        TypeError
            if the type of the parameter is not Disk.

        Returns
        -------
        bool
            True if the given disk have the same colour as the
            calling disk, False otherwise.

        """
        if not isinstance(other, Disk):
            raise TypeError('Argument must be of type Disk')
        return self._colour != other.get_colour()

    def __hash__(self):
        return hash((self._location, self._king, self._colour))

    def __eq__(self, other):
        if not isinstance(other, Disk):
            return False
        return self._colour == other.get_colour() and \
            self._king == other.is_king() and \
            self._location == other.get_location()


class Board:
    """Represents the CheckerBoard.

    represent the CheckerBoard using two sets:
        one for 'white', and another for 'black' pieces.
    you can reach to every piece of a given colour.
    you can reach to a piece at a given location.
    remove and add pieces are supported.

    """

    # max. number of game's turns without attack-move, before
    # declare draw.
    max_draw_counter = 55

    def __init__(self, white_disks: set, black_disks: set) -> None:
        """Initialize the Board.

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
        self._disks_at = self._build_disks_at()

    def copy(self):
        """Use it to copy the calling board.

        Returns
        -------
        Board
            a copy of the calling board.

        """
        return Board(self._disks['white'].copy(), self._disks['black'].copy())

    def get_number_of_kings(self, colour: str) -> int:
        """Get the number of kings with a given colour.

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
            if disk.is_king() and disk._colour == colour:
                num += 1

        return num

    def get_number_of_disks(self, colour: str = None) -> int:
        """Get the number of disks with a given colour.

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
            return (self.get_number_of_disks('white') +
                    self.get_number_of_disks('black'))
        elif colour == 'white':
            return len(self._disks['white'])
        elif colour == 'black':
            return len(self._disks['black'])
        else:
            raise ValueError("""colour must be
                             "black", or "white", or "None"! """)

    def _build_disks_at(self) -> dict:
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
        """Get the disk at the given location.

        Parameters
        ----------
        location : tuple
            some location on the board.

        Returns
        -------
        Disk
            The disk at the given location if there is any, None otherwise.

        """
        if location in self._disks_at:
            return self._disks_at[location]
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
        """Remove a disk with a given location from the board.

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
        del self._disks_at[location]

    def add_disk_at(self, disk: Disk, location: tuple) -> None:
        """Add a disk in the given location.

        Parameters
        ----------
        disk : Disk
            a disk that we want to add.
        location : tuple
            location on the board where you want to add the given disk.

        Raises
        ------
        ValueError
            if the location not empty (i.e there is disk on it).

        Returns
        -------
        None

        """
        if location in self._disks_at:
            raise ValueError('location is already have a Disk!')
        self._disks[disk.get_colour()].add(disk)
        self._disks_at[location] = disk

    def get_status(self, colour: str, draw_counter: int) -> str:
        """Get the status of the game for the player with the given colour.

        the status is 'win', 'lose', or 'draw' if the game ends,
        otherwise None is returned.

        Parameters
        ----------
        colour : str
            the colour of the pieces controlled by the player that we want
            to know if he is at 'win', 'lose', 'draw' status, or if the
            game is not end yet.
        draw_counter : int
            counter of non-attack moves.

        Returns
        -------
        str
            the status is 'win', 'lose', or 'draw' if the game ends,
            otherwise None is returned..

        """
        boards_me = []
        Moves.get_all_next_boards(self, colour, boards_me)
        n_me = len(boards_me)
        if n_me == 0:
            return 'lose'
        enemy_colour = 'white' if colour == 'black' else 'black'
        boards_enemy = []
        Moves.get_all_next_boards(self, enemy_colour, boards_enemy)
        n_enemy = len(boards_enemy)
        if n_enemy == 0:
            return 'win'
        n_me = len(self.get_disks(colour))
        n_enemy = len(self.get_disks(enemy_colour))
        if draw_counter >= Board.max_draw_counter:
            return 'draw'
        return None  # the game is still going


class Moves:
    """Contains the moves functionality of the checker game."""

    def is_valid_position(x: int, y: int) -> bool:
        """Test if a given location is inside the board or not.

        Parameters
        ----------
        x : int
            row number.
        y : int
            column number.

        Returns
        -------
        bool
            True if the given location is inside the board, False otherwise.

        """
        return x >= 0 and x < 8 and y >= 0 and y < 8

    def get_next_boards(board: Board, location: tuple, next_boards: list, *,
                        next_locations: list = None,
                        threatened: dict = None) -> None:
        """Get all possible boards after the Disk at the given location move.

        Parameters
        ----------
        board : Board
            the current board (i.e before move).

        location : tuple
            location of the disk which we want to move it.

        next_boards : list
            list of all next boards.
            this an output parameter i.e we use it to store the results
            of the method.

        next_locations : list, optional, keyword-only
            used to store all locations that the disk at
            the given location can move to.
            The default is None.

        threatened : dict, optional, keyword-only
            used to store the location of every threatened disk for the enemy.
            locations are the keys.
            The default is None.

        Returns
        -------
        None

        """
        # frontier is a list of tuples.
        # tuple[0] is the board reached so far.
        # tuple[1] is the current location
        # tuple[2] is a flag that indicates if a non-eat move is allowed
        frontier = []
        frontier.append((board, location))
        non_eat_move = True

        while frontier:
            # retrive information if the current state
            current_board, current_location = frontier.pop()
            # extract the disk which we want to move
            current_disk = current_board.get_disk_at(current_location)

            # eat moves:
            for dx, dy in current_disk.get_directions():
                # calculate the next location row, and colomn number.
                next_x = current_location[0] + 2*dx
                next_y = current_location[1] + 2*dy
                # test if the next location is inside the board
                # skip if not
                if not Moves.is_valid_position(next_x, next_y):
                    continue
                next_location = (next_x, next_y)
                # get calculate the location of the enemy disk we want to eat.
                enemy_location = (current_location[0] + dx,
                                  current_location[1] + dy)

                # test if the next location is empty so we can move to it
                # skip if not
                if current_board.get_disk_at(next_location) is not None:
                    continue
                # test if the enemy location is contain a disk
                # skip if not
                if current_board.get_disk_at(enemy_location) is None:
                    continue
                # enemy disk to be eaten
                enemy_disk = current_board.get_disk_at(enemy_location)
                # check if the disk at enemy location is really an enemy
                # i.e not one of your pieces
                if current_disk.is_enemy(enemy_disk) is False:
                    continue
                # add the enemy disk to the threatened disks if we need that
                if threatened is not None:
                    threatened[enemy_disk.get_location()] = 1
                # very bad time complexity, need future improvement
                # build a new board
                next_board = current_board.copy()

                # remove the disks from the old locations
                next_board.remove_disk_at(current_location)
                next_board.remove_disk_at(enemy_location)
                # create a new disk to update the information in it
                next_disk = current_disk.copy()
                next_disk.set_location(next_location)  # update the location.
                # add the disk to the new location
                next_board.add_disk_at(next_disk, next_location)
                # add the board to the next_boards
                next_boards.append(next_board)
                # add the next_location if we need it
                if next_locations is not None:
                    next_locations.append(next_location)
                # don't add to frontier if the disk have just promoted
                # because a disk cannot use king moves until next turn.
                if not current_disk.is_king() and next_disk.is_king():
                    continue
                frontier.append((next_board, next_location))
            if non_eat_move is True:
                for dx, dy in current_disk.get_directions():
                    # calculate the next location row, and colomn number
                    next_x = current_location[0] + dx
                    next_y = current_location[1] + dy
                    # test if the next location is inside the board
                    # skip if not
                    if not Moves.is_valid_position(next_x, next_y):
                        continue
                    next_location = (next_x, next_y)

                    # test if the next location is empty so we can move to it
                    # skip if not
                    if current_board.get_disk_at(next_location) is None:
                        # very bad time complexity, need future improvement
                        # build a new board
                        next_board = current_board.copy()
                        # remove the disk from the old location
                        next_board.remove_disk_at(current_location)
                        # create a new disk to update the information in it
                        next_disk = current_disk.copy()
                        # update the location.
                        next_disk.set_location(next_location)
                        # add the disk to the new location
                        next_board.add_disk_at(next_disk, next_location)
                        # add the board to the next_boards
                        next_boards.append(next_board)
                        # add the next_location if we need it
                        if next_locations is not None:
                            next_locations.append(next_location)

            non_eat_move = False  # only first time a non-eat move allowed.

    def get_all_next_boards(board: Board, colour: str,
                            next_boards: list,
                            threatened: dict = None) -> None:
        """Get all possible boards after moving a disk.

        Parameters
        ----------
        board : Board
            the current board (i.e before move).
        colour : str
            the colour of the current player.
        next_boards : list
            list of all next boards.
            this an output parameter i.e we use it to store the results
            of the method.
        threatened : dict, optional
            used to store the location of every threatened disk for the enemy.
            locations are the keys.
            The default is None.

        Raises
        ------
        ValueError
            if the colour is neither 'white', nor 'black'.

        Returns
        -------
        None

        """
        if colour not in ['white', 'black']:
            raise ValueError("colour must be either 'white', or 'black'!")

        # try to move each disk of the given player one at a time
        for disk in board.get_disks(colour):
            Moves.get_next_boards(board, disk.get_location(),
                                  next_boards,
                                  threatened=threatened)


def update_draw_counter(draw_counter: int, size_before: int,
                        size_after: int) -> None:
    """Update counter of non-attack moves, which is a condition for draw.

    Parameters
    ----------
    draw_counter : int
        counter of non-attack moves.
    size_before : int
        the size of the board before the move.
    size_after : int
        the size of the board after the draw.

    Returns
    -------
    int
        counter of non-attack moves after update process.

    """
    if size_before != size_after:
        draw_counter = 0
    else:
        draw_counter += 1
    return draw_counter


if __name__ == '__main__':
    def disk_and_board_test():
        d1 = Disk(location=(7, 7), colour='white')
        d2 = Disk(location=(7, 7), colour='white')
        assert(d1 == d2)
        assert(hash(d1) == hash(d2))
        d1.set_location((0, 6))
        assert(d1 != d2)
        assert(hash(d1) != hash(d2))
        assert(d1.is_king() is True)
        d2.set_colour('black')
        assert(d1.is_enemy(d2) is True)
        d1 = Disk(location=(6, 6), colour='white')
        d2 = Disk(location=(5, 6), colour='white')
        d3 = Disk(location=(4, 6), colour='black')
        d4 = Disk(location=(4, 4), colour='black')
        white_disks = set([d1, d2])
        black_disks = set([d3, d4])
        b = Board(white_disks, black_disks)
        assert(b.get_number_of_disks() == 4)
        assert(b.get_number_of_disks('white') == 2)
        assert(b.get_number_of_disks('black') == 2)
        assert(b.get_number_of_kings('white') == 0)
        assert(b.get_number_of_kings('black') == 0)
        b2 = Board(white_disks=b.get_disks('white').copy(),
                   black_disks=b.get_disks('black').copy())
        b.remove_disk_at((6, 6))
        assert(b2 is not b)
        assert(b.get_number_of_disks('white') == 1)
        b.remove_disk_at((4, 4))
        assert(b.get_number_of_disks('black') == 1)
        assert(b.is_empty('white') is False)
        b.remove_disk_at((5, 6))
        assert(b.get_number_of_disks('white') == 0)
        assert(b.is_empty('white') is True)

    def moves_tests():
        white_disks = [(0, 4), (0, 6), (1, 5), (1, 7),
                       (2, 0), (2, 4), (3, 5)]
        black_disks = [(3, 1), (4, 4), (5, 5), (6, 0),
                       (6, 2), (6, 4), (6, 6)]
        for i, loc in enumerate(white_disks.copy()):
            white_disks[i] = (7 - loc[0], 7 - loc[1])
        for i, loc in enumerate(black_disks.copy()):
            black_disks[i] = (7 - loc[0], 7 - loc[1])
        for i, loc in enumerate(white_disks.copy()):
            white_disks[i] = Disk(location=loc, colour='white')
        for i, loc in enumerate(black_disks.copy()):
            black_disks[i] = Disk(location=loc, colour='black')
        b = Board(set(white_disks), set(black_disks))
        d_moves = {
            (5, 7): [(3, 5)],
            (5, 3): [(4, 4)],
            (7, 1): [],
            (4, 2): [(3, 1), (2, 4), (0, 6), (0, 2)],
            (4, 6): [(5, 5)],
            (3, 3): [(4, 4), (5, 1)]
        }
        for k, v in d_moves.items():
            next_locations = []
            next_boards = []
            Moves.get_next_boards(b, k, next_boards,
                                  next_locations=next_locations)
            code_results = sorted(next_locations)
            correct_results = sorted(v)
            assert(code_results == correct_results)
        b.remove_disk_at((4, 2))
        d = Disk(location=(4, 2), colour='white')
        d.promote_to_king()
        b.add_disk_at(d, (4, 2))
        next_locations = []
        next_boards = []
        Moves.get_next_boards(b, (4, 2), next_boards,
                              next_locations=next_locations)
        assert(len(next_boards) > 0)
        code_results = sorted(next_locations)
        correct_results = sorted([(3, 1), (2, 4), (0, 6),
                                  (0, 2), (2, 0), (5, 1)])
        assert(code_results == correct_results)

    def threat_test():
        white_disks = [(0, 4), (0, 6), (1, 5), (1, 7),
                       (2, 0), (2, 4), (3, 5)]
        black_disks = [(3, 1), (4, 4), (5, 5), (6, 0),
                       (6, 2), (6, 4), (6, 6)]
        for i, loc in enumerate(white_disks.copy()):
            white_disks[i] = (7 - loc[0], 7 - loc[1])
        for i, loc in enumerate(black_disks.copy()):
            black_disks[i] = (7 - loc[0], 7 - loc[1])
        for i, loc in enumerate(white_disks.copy()):
            white_disks[i] = Disk(location=loc, colour='white')
        for i, loc in enumerate(black_disks.copy()):
            black_disks[i] = Disk(location=loc, colour='black')
        b = Board(set(white_disks), set(black_disks))
        threatened = dict()
        next_boards = []
        Moves.get_all_next_boards(b, 'white', next_boards, threatened)
        correct_results = [
                (4, 6), (3, 3), (1, 3), (1, 5)
            ]
        assert(sorted(correct_results) == sorted(threatened.keys()))
        assert(len(threatened.keys()) == 4)
        threatened = dict()
        next_boards = []
        Moves.get_all_next_boards(b, 'black', next_boards, threatened)
        correct_results = [
                (4, 2)
            ]
        assert(sorted(correct_results) == sorted(threatened.keys()))
        assert(len(threatened.keys()) == 1)

    disk_and_board_test()
    moves_tests()
    threat_test()

    print('Everything work.')
