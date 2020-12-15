# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 22:24:06 2020

@author: Kais
"""

from Checker.Game import Board, Moves, Disk
import numpy as np

class AISystem:
    def update_parameters(self, boards: list) -> None:
        pass
    def predict(self, board: Board) -> float:
        pass


class FeaturesBasedSystem(AISystem):
    def __init__(self, name: str, learning_rate: int,
                 useSavedParameters: bool, *features):
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

        # set name
        self._name = name

        self._useSavedParameters = useSavedParameters

        # set parameters { shape = (n, 1) }:
        self._parameters = np.random.rand(len(self._features), 1)
        if self._useSavedParameters is True:
            try:
                with open(self._name + '_' + 'parameters.npy', 'rb') as f:
                    self._parameters = np.load(f)
            except IOError:
                pass  # if no file exist just use the default values first

        # set learning rate
        self._learning_rate = learning_rate

    def generate_training_set(self, boards: list, colour: str) -> tuple:
        boards.reverse()
        m = len(boards)
        # shape = (m, 2)
        # first column is board
        # second column is V
        training_set = []
        turn = len(boards)
        status1 = boards[0].get_status(colour, turn)
        enemy_colour = 'white' if colour == 'black' else 'black'
        status2 = boards[0].get_status(enemy_colour, turn)
        status = None
        if status1 == 'lose':
            status = 'lose'
        elif status2 == 'lose':
            status = 'win'
        # if we reach here then no one is a winner
        # first case is that it's a draw then status 1&2 must be 'draw'
        # second case is that the game is still going then None is the value
        status = status1
        if status == 'win':
            next_exmaple = (boards[0], 100)
            training_set.append(next_exmaple)
        elif status == 'lose':
            next_exmaple = (boards[0], -100)
            training_set.append(next_exmaple)
        elif status == 'draw':
            next_exmaple = (boards[0], 0)
            training_set.append(next_exmaple)
        else:
            raise ValueError("The game doesn't end yet to train on it")

        # list to store the predictions values
        v_hat = []
        for i in range(1, m):
            v_hat[i-1] = self.predict(boards[i-1])
            v_i = v_hat[i-1]
            next_exmaple = (boards[i], v_i)
            training_set.append(next_exmaple)
        v_hat[-1] = self.predict(boards[-1])
        training_set.reverse()
        v_hat.reverse()
        return training_set, v_hat

    def compute_error(self, boards: list) -> float:
        training_set, pred = self.generate_training_set(boards)
        v_tr = np.array(training_set[:, 1])
        v_hat = np.array(pred)
        diff = v_tr - v_hat
        return np.dot(diff.T, diff)

    def update_parameters(self, boards: list, colour: str) -> None:
        training_set, pred = self.generate_training_set(boards, colour)
        # v_tr.shape = (m, )
        v_tr = np.array(training_set[:, 1])
        # v_hat.shape = (m, )
        v_hat = np.array(pred)

        # features matrix of shape (m, n)
        # m examples
        # n features for each example
        X = self.get_all_features(boards)
        self._parameters = self._parameters + self._learning_rate * (
                                np.dot(X.T, v_tr - v_hat)
                            )
        with open(self._name + '_' + 'parameters.npy', 'wb') as f:
            np.save(self._parameters, f)

    def get_features(self, board: Board) -> np.array:
        n = len(self._features)
        x = np.zeros((n, 1))
        for i in range(n):
            x[i] = self._features[i](board)
        return x

    def get_all_features(self, boards: list) -> np.array:
        m = len(boards)
        n = len(self._features)
        x = np.zeros((m, n))
        for i, board in enumerate(boards):
            x[i, :] = self.get_features(board)
        return x

    def predict(self, board: Board) -> float:
        # features vector { shape=(n, 1) }:
        x = self.get_features(board)
        return np.squeeze(np.dot(self._parameters.T, x))

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
        _ = Moves.get_all_next_boards(board, 'white', threatened=threatened)
        return len(threatened.keys())

    def _f6_number_of_pieces_threatened_by_black(self, board: Board) -> float:
        threatened = dict()
        _ = Moves.get_all_next_boards(board, 'black', threatened=threatened)
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
