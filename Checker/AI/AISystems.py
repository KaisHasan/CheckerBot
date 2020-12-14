# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 22:24:06 2020

@author: Kais
"""

from Checker import Board
import numpy as np


class AISystem:
    def update_parameters(self, boards: list) -> None:
        pass
    def predict(self, board: Board) -> float:
        pass


class FeaturesBasedSystem(AISystem):
    def __init__(self, learning_rate: int,
                 useSavedParameters: bool, *features):
        # add features:
        self._features = []
        self._features.append(self._f1_number_of_black_pieces)
        self._features.append(self._f2_number_of_white_pieces)
        self._features.append(self._f3_number_of_black_kings)
        self._features.append(self._f4_number_of_white_kings)
        self._features.append(self._f5_number_of_pieces_threatened_by_white)
        self._features.append(self._f6_number_of_pieces_threatened_by_black)
        self._fetaures.extend(features)

        self._useSavedParameters = useSavedParameters

        # set parameters:
        self._parameters = np.zeros((len(self._features)), 0)
        if self._useSavedParameters is True:
            try:
                with open('parameters.npy', 'rb') as f:
                    self._parameters = np.load(f)
            except IOError:
                pass  # if no file exist just use the default values first

        # set learning rate
        self._learning_rate = learning_rate

    def update_parameters(self, boards: list) -> None:
        pass

    def predict(self, board: Board) -> float:
        pass

    def _f1_number_of_black_pieces(self, board: Board) -> float:
        return board.get_number_of_disks('black')

    def _f2_number_of_white_pieces(self, board: Board) -> float:
        return board.get_number_of_disks('white')

    def _f3_number_of_black_kings(self, board: Board) -> float:
        return board.get_number_of_kings('black')

    def _f4_number_of_white_kings(self, board: Board) -> float:
        return board.get_number_of_kings('white')

    def _f5_number_of_pieces_threatened_by_white(self, board: Board) -> float:
        pass

    def _f6_number_of_pieces_threatened_by_black(self, board: Board) -> float:
        pass
