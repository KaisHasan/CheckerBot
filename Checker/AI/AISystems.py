# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 22:24:06 2020

@author: Kais
"""

from Checker.Game import Board, Moves, Disk
import numpy as np


class AISystem:
    """Abstract class for an AI system.

    Note: you must follow the documentation when implementing a system
    that inherit this class to use it with an agent.

    """

    def update_parameters(self, boards: list, final_status: str) -> None:
        """Use it to update the parameters of the system.

        Parameters
        ----------
        boards : list
            a list of boards represent the board positions through the game.
            type of each element must be of type Board.
        final_status: str
            the final status of the game, i.e 'win', 'lose', 'draw'.

        Returns
        -------
        None

        """
        pass

    def predict(self, board: Board) -> float:
        """Use it to predict the fitness value for a given board.

        Parameters
        ----------
        board : Board
            the board we want to predict its fitness.

        Returns
        -------
        float
            the fitness value of the given board.

        """
        pass

    def copy(self):
        """Use it to copy the system.

        Note: you must give the new system a name different from the name
        of the original system, otherwise logical errors will happen.

        Returns
        -------
        AISystem
            a copy of the calling system.

        """
        pass

    def get_name(self) -> str:
        """Get the name of the system.

        Returns
        -------
        str
            the name of the system.

        """
        pass

    def compute_error(self, boards: list, final_status: str) -> float:
        """Compute the error in the prediction of our system.

        Parameters
        ----------
        boards : list
            the list of all boards through the game
            where it's white turn.
        final_status : str
            the final status of the white player.
            must be 'win', 'lose', or 'draw'.

        Returns
        -------
        float
            the error in the prediction of our system.

        """
        pass


class FeaturesBasedSystem(AISystem):
    """An AISystem that extract features and use them to predict."""

    def __init__(self, name: str, learning_rate: float,
                 useSavedParameters: bool, *features):
        """Initialize the System.

        Parameters
        ----------
        name : str
            the name of the system.
        learning_rate : float
            used when we want to update the parameters.
            must be a small value << 1.
        useSavedParameters : bool
            indicates if you want to use the parameters from previous
            training processes of the system with the given name.
            if its value is False, then the parameters will given a random
            values.
            when no previous parameters found for the system the behaviour
            is similar to where the value is False.
        *features : functions
            a functions that extracts features.
            the functions must take a board and return a numeric value.

        Returns
        -------
        None.

        """
        # add features:
        self._features = []
        self._features.append(self._f0_b)
        self._features.append(self._f1_number_of_black_pieces)
        self._features.append(self._f2_number_of_white_pieces)
        self._features.append(self._f3_number_of_black_kings)
        self._features.append(self._f4_number_of_white_kings)
        self._features.append(self._f5_number_of_pieces_threatened_by_white)
        self._features.append(self._f6_number_of_pieces_threatened_by_black)
        self._features.extend(features)
        self._add_features = list(features)  # used when we want to copy

        # set name
        self._name = name

        self._useSavedParameters = useSavedParameters

        # set parameters { shape = (n, 1) }:
        self._parameters = np.random.randn(len(self._features), 1) * 0.01
        if self._useSavedParameters is True:
            try:
                with open(self._name + '_' + 'parameters.npy', 'rb') as f:
                    self._parameters = np.load(f, allow_pickle=True)
            except IOError:
                pass  # if no file exist just use the default values first

        # set learning rate
        self._learning_rate = learning_rate

    def copy(self):
        """Use it to copy the system.

        Returns
        -------
        AISystem
            a copy of the calling system.

        """
        s = FeaturesBasedSystem('copy_'+self._name, self._learning_rate,
                                self._useSavedParameters,
                                *self._add_features.copy())
        return s

    def get_name(self) -> str:
        """Get the name of the system.

        Returns
        -------
        str
            the name of the system.

        """
        return self._name

    def generate_training_set(self, boards: list, final_status: str) -> tuple:
        """Generate the training set from the given history of some game.

        Parameters
        ----------
        boards : list
            the list of all boards through the game
            where it's white turn.
        final_status : str
            the final status of the white player.
            must be 'win', 'lose', or 'draw'.

        Raises
        ------
        ValueError
            if the given list of boards represents a game that not end yet.

        Returns
        -------
        tuple
            first element is a list of the training examples.
                it's a list of 2-tuple.
                so the first column contains the boards.
                the second column contains the training values of each board.
            second element is a list of the predicted values of the boards.

        """
        boards.reverse()
        m = len(boards)
        # shape = (m, 2)
        # first column is board
        # second column is V
        training_set = []
        # list to store the predictions values
        v_hat = []

        if final_status == 'win':
            next_exmaple = (boards[0], 1)
            training_set.append(next_exmaple)
        elif final_status == 'lose':
            next_exmaple = (boards[0], -1)
            training_set.append(next_exmaple)
        elif final_status == 'draw':
            next_exmaple = (boards[0], 0)
            training_set.append(next_exmaple)
        else:
            raise ValueError("The game doesn't end yet to train on it")

        for i in range(1, m):
            v_hat.append(self.predict(boards[i-1]))
            v_i = v_hat[i-1]
            next_exmaple = (boards[i], v_i)
            training_set.append(next_exmaple)
        v_hat.append(self.predict(boards[-1]))
        training_set.reverse()
        v_hat.reverse()
        boards.reverse()
        return training_set, v_hat

    def compute_error(self, boards: list, final_status: str) -> float:
        """Compute the error in the prediction of our system.

        Parameters
        ----------
        boards : list
            the list of all boards through the game
            where it's white turn.
        final_status : str
            the final status of the white player.
            must be 'win', 'lose', or 'draw'.

        Returns
        -------
        float
            the error in the prediction of our system.

        """
        training_set, pred = self.generate_training_set(boards, final_status)
        v_tr = np.array([x[1] for x in training_set])
        v_hat = np.array(pred)
        diff = v_tr - v_hat
        m = v_tr.shape[0]
        return (1./(2.*m))*np.dot(diff.T, diff)

    def update_parameters(self, boards: list, final_status: str) -> None:
        """Use it to update the parameters of the system.

        Parameters
        ----------
        boards : list
            a list of boards represent the board positions through the game.
            type of each element must be of type Board.
        final_status: str
            the final status of the game, i.e 'win', 'lose', 'draw'.

        Returns
        -------
        None

        """
        training_set, pred = self.generate_training_set(boards, final_status)
        # v_tr.shape = (m, )
        v_tr = np.array([x[1] for x in training_set])
        # v_hat.shape = (m, )
        v_hat = np.array(pred)

        # features matrix of shape (m, n)
        # m examples
        # n features for each example
        X = self.get_all_features(boards)
        assert(X.shape == (len(training_set), len(self._features)))

        m = X.shape[0]

        # vectorized version:
        self._parameters = self._parameters + (self._learning_rate/m)*(
                                                    np.dot(X.T, v_tr - v_hat)
                                                )

        # save the parameters into a file to use them later.
        if self._useSavedParameters is True:
            with open(self._name + '_' + 'parameters.npy', 'wb') as f:
                np.save(f, self._parameters)

    def get_features(self, board: Board) -> np.array:
        """Get the features of the given board.

        Parameters
        ----------
        board : Board

        Returns
        -------
        x : np.array
            an array with shape: (number of features, 1).
            contains the features values for the given board.

        """
        n = len(self._features)
        x = np.zeros((n, 1))
        for i in range(n):
            x[i] = self._features[i](board)
        return x

    def get_all_features(self, boards: list) -> np.array:
        """Get the features of all boards.

        Parameters
        ----------
        boards : list
            a list of boards.

        Returns
        -------
        x : np.array
            an array with shape: (number of boards, number of features).
            i-th row contains the features values for the i-th board.

        """
        m = len(boards)
        n = len(self._features)
        x = np.zeros((m, n))
        for i, board in enumerate(boards):
            x[i, :] = self.get_features(board).ravel()
        return x

    def predict(self, board: Board) -> float:
        """Use it to predict the fitness value for a given board.

        the predict value is a linear combination of the
        features values and the parameters.

        Parameters
        ----------
        board : Board
            the board we want to predict its fitness.

        Returns
        -------
        float
            the fitness value of the given board.

        """
        # features vector { shape=(n, 1) }:
        x = self.get_features(board)
        return np.sum(np.dot(self._parameters.T, x))

    def _f0_b(self, board: Board) -> float:
        return 1

    def _f1_number_of_black_pieces(self, board: Board) -> float:
        return board.get_number_of_disks('black')

    def _f2_number_of_white_pieces(self, board: Board) -> float:
        return board.get_number_of_disks('white')

    def _f3_number_of_black_kings(self, board: Board) -> float:
        return board.get_number_of_kings('black')

    def _f4_number_of_white_kings(self, board: Board) -> float:
        return board.get_number_of_kings('white')

    def _f5_number_of_pieces_threatened_by_white(self, board: Board) -> float:
        threatened = dict()
        _ = Moves.get_all_next_boards(board, 'white', threatened)
        return len(threatened.keys())

    def _f6_number_of_pieces_threatened_by_black(self, board: Board) -> float:
        threatened = dict()
        _ = Moves.get_all_next_boards(board, 'black', threatened)
        return len(threatened.keys())


if __name__ == '__main__':
    f_system = FeaturesBasedSystem('test', 0.01, True)
    white_disks = [(0, 4), (0, 6), (1, 5), (1, 7),
                   (2, 0), (2, 4), (3, 5)]
    black_disks = [(3, 1), (4, 4), (5, 5), (6, 0),
                   (6, 2), (6, 4), (6, 6)]
    for i, loc in enumerate(white_disks.copy()):
        white_disks[i] = Disk(location=loc, colour='white')
    for i, loc in enumerate(black_disks.copy()):
        black_disks[i] = Disk(location=loc, colour='black')
    b = Board(set(white_disks), set(black_disks))
    print(f_system.predict(b))
    print('Everything work.')
