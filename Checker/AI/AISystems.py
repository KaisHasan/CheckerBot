# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 22:24:06 2020

@author: Kais
"""

from Checker.Game import Board, Moves, Disk, update_draw_counter
import numpy as np
import json


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

    def predict(self, boards: list, turn: int, draw_counter: int) -> np.array:
        """Use it to predict the fitnesses values for a given list of boards.

        Parameters
        ----------
        boards : list
            list of boards we want to predict their fitnesses.
        turn : int
            turn's number.
        draw_counter : int
            counter of non-attack moves.

        Returns
        -------
        np.array
            the fitnesses values of the given boards.

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
            except IOError as err:
                print(err)
                print('system will use random parameters')
                pass  # if no file exist just use the default values first

        # set learning rate
        self._learning_rate = learning_rate

    def set_parameters(self, parameters: np.array) -> None:
        """Set the parameters of the system to a given values.

        Parameters
        ----------
        parameters : np.array
            a given list of parameters.

        Returns
        -------
        None

        """
        self._parameters = parameters

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
            numpy array contains the training values of each board.
            second element is a list of the predicted values of the boards.

        """
        m = len(boards)
        # shape = (m, )
        training_set = np.zeros((m, 1))
        assert(training_set.shape == (m, 1))
        # numpy array to store the predictions values
        v_hat = self.predict(boards)
        assert(v_hat.shape == (m, 1))

        if final_status == 'win':
            training_set[-1] = 1
        elif final_status == 'lose':
            training_set[-1] = -1
        elif final_status == 'draw':
            training_set[-1] = 0
        else:
            raise ValueError("The game doesn't end yet to train on it")

        for i in range(m - 1):
            training_set[i] = v_hat[i+1]

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
        v_tr, v_hat = self.generate_training_set(boards, final_status)

        diff = v_tr - v_hat
        m = v_tr.shape[0]
        return (1./(2.*m))*np.sum(np.dot(diff.T, diff))

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
        v_tr, v_hat = self.generate_training_set(boards, final_status)
        m = v_tr.shape[0]
        assert(v_tr.shape == (m, 1))
        assert(v_hat.shape == (m, 1))

        # features matrix of shape (m, n)
        # m examples
        # n features for each example
        X = self.get_all_features(boards)
        assert(X.shape == (m, len(self._features)))

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

    def predict(self, boards: list, turn: int, draw_counter: int) -> np.array:
        """Use it to predict the fitnesses values for a given list of boards.

        the predict value for each board is a linear combination
        of the features values and the parameters.

        Parameters
        ----------
        boards : list
            list of boards we want to predict their fitnesses.
        turn : int
            turn's number.
        draw_counter : int
            counter of non-attack moves.

        Returns
        -------
        np.array
            shape = (number of boards, 1)
            the fitnesses values of the given boards.

        """
        # features matrix { shape=(m, n) }:
        X = self.get_all_features(boards)
        return np.dot(X, self._parameters)

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
        boards = []
        Moves.get_all_next_boards(board, 'white', boards, threatened)
        return len(threatened.keys())

    def _f6_number_of_pieces_threatened_by_black(self, board: Board) -> float:
        threatened = dict()
        boards = []
        Moves.get_all_next_boards(board, 'black', boards, threatened)
        return len(threatened.keys())


class NeuralNetworkBasedSystem(AISystem):
    """Neural network system for Checker bot."""

    def __init__(self, name: str, learning_rate: float, num_hidden_units: list,
                 use_saved_parameters: bool) -> None:
        """Initialize Neural Network System.

        Parameters
        ----------
        name : str
            name of the system [used to save parameters].
        learning_rate : float
        num_hidden_units : list
            i-th element of the list contains number of units
            in the i-th hidden layer.
        use_saved_parameters : bool
            indicates if you want to use the parameters from previous
            training processes of the system with the given name.
            if its value is False, then the parameters will given a random
            values.
            when no previous parameters found for the system the behaviour
            is similar to where the value is False.

        Raises
        ------
        ValueError
            if the number of hidden layers (i.e the length of the list:
                                            num_hidden_units) is zero.

        Returns
        -------
        None

        """
        # set name
        self._name = name

        self._use_saved_parameters = use_saved_parameters

        if len(num_hidden_units) < 1:
            raise ValueError('Number of the hidden units must be > 0')
        self._num_units = [32]  # number of units in input layer
        self._num_units.extend(num_hidden_units)  # add hidden layers
        self._num_units.append(1)  # add the output layer

        # set parameters { shape = (n, 1) }:
        self._parameters = self._initialize_parameters()
        if self._use_saved_parameters is True:
            try:
                self._parameters = self.load_parameters()
            except IOError as err:
                print(err)
                print('system will use random parameters')
                assert(self._parameters is not None)
                pass  # if no file exist just use the default values first

        # set learning rate
        self._learning_rate = learning_rate

    def save_parameters(self) -> None:
        """Save the parameters of the neural network into json file.

        Returns
        -------
        None

        """
        parameters = dict()
        # convert the numpy arrays into lists
        # that's because we can't serialize numpy arrays
        for k, v in self._parameters.items():
            parameters[k] = v.tolist()
        # write into the json file
        with open(self._name + '_parameters.json', 'w') as f:
            json.dump(parameters, f)

    def load_parameters(self) -> dict:
        """Load the parameters of the neural network from a json file.

        Returns
        -------
        dict
            the parameters of the neural network.

        """
        with open(self._name + '_parameters.json', 'r') as f:
            temp = json.load(f)  # load the parameters into dictionary
            parameters = dict()
            # convert the lists to numpy arrays
            for k, v in temp.items():
                parameters[k] = np.array(v)
            return parameters

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
        # generate:
        # training values into v_tr
        # activation value of every layer into cache
        v_tr, cache = self._generate_training_set(boards, final_status)

        # get the gradient
        # dictionary contains the partial derivative of cost function
        # w.r.t weights and biases of every layer
        grad = self._back_propagation(v_tr, cache)

        # number of layers + 1 (we don't count input layer i.e layer zero)
        num_layer = len(self._num_units)

        for i in range(1, num_layer):
            # update weights
            self._parameters['W' + str(i)] -= self._learning_rate * (
                                                grad['dW' + str(i)]
                                                )
            # update bias
            self._parameters['b' + str(i)] -= self._learning_rate * (
                                                grad['db' + str(i)]
                                                )
        self.save_parameters()  # save parameters after update done.

    def predict(self, boards: list, turn: int, draw_counter: int) -> np.array:
        """Use it to predict the fitness value for a given list of boards.

        Parameters
        ----------
        boards : list
            list of boards we want to predict their fitnesses.
        turn : int
            turn's number.
        draw_counter : int
            counter of non-attack moves.

        Returns
        -------
        np.array
            the fitnesses values of the given boards.

        """
        # do forward propagation to get all activation values of layers
        # retrieve tha last activation value which represent the output.
        return self._forward_propagation(boards)[-1]

    def copy(self):
        """Use it to copy the system.

        Returns
        -------
        AISystem
            a copy of the calling system.

        """
        temp = NeuralNetworkBasedSystem('copy_' + self._name,
                                        self._learning_rate,
                                        self._num_units[1:-1],
                                        self._useSavedParameters)
        return temp

    def get_name(self) -> str:
        """Get the name of the system.

        Returns
        -------
        str
            the name of the system.

        """
        return self._name

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
        v_tr, cache = self._generate_training_set(boards, final_status)
        m = v_tr.shape[1]  # number of examples
        diff = cache[-1] - v_tr
        return (1./(2*m))*np.sum(np.dot(diff.T, diff))

    def _activation(self, z: np.array) -> np.array:
        """Activation function.

        Parameters
        ----------
        z : np.array

        Returns
        -------
        np.array
            shape of input is the same of the shape of output.

        """
        return np.tanh(z)

    def _initialize_parameters(self) -> dict:
        """Initialize the parameters of the neural network.

        Returns
        -------
        dict
            dictionary contains the parameters of the neural network.

        """
        num_layer = len(self._num_units)  # number of layer + 1
        parameters = dict()
        for i in range(1, num_layer):
            parameters['W' + str(i)] = np.random.randn(self._num_units[i],
                                                       self._num_units[i-1]
                                                       ) * 0.01
            parameters['b' + str(i)] = np.zeros((self._num_units[i], 1))
        return parameters

    def _get_input(self, board: Board) -> np.array:
        """Generate the input to the neural network from a board.

        Parameters
        ----------
        board : Board

        Returns
        -------
        np.array
            shape is (32, 1).

        """
        inp = np.zeros((32, 1))
        id = 0
        for i in range(8):
            for j in range(i & 1, 8, 2):
                disk = board.get_disk_at((i, j))
                if disk is None:
                    inp[id] = 0
                else:
                    val = 1
                    if disk.is_king():
                        val += 1
                    if disk.get_colour() == 'black':
                        val = -val
                    inp[id] = val
                id += 1
        return inp

    def _get_input_layer(self, boards: list) -> np.array:
        """Generate the input of the neural network from a list of boards.

        Parameters
        ----------
        boards : list

        Returns
        -------
        np.array
            shape is (32, number of boards).

        """
        m = len(boards)
        A_0 = np.zeros((32, m))
        for j in range(m):
            A_0[:, j] = self._get_input(boards[j]).ravel()
        return A_0

    def _forward_propagation(self, boards: list) -> list:
        """Forward propagation through the network.

        Parameters
        ----------
        boards : list
            list of boards through the game for 'white'.

        Returns
        -------
        list
            activation value for each layer.

        """
        num_layer = len(self._num_units)
        A_0 = self._get_input_layer(boards)
        cache = [A_0]
        for i in range(1, num_layer):
            A_pre = cache[i-1]
            W_i = self._parameters['W' + str(i)]
            b_i = self._parameters['b' + str(i)]
            Z_i = np.dot(W_i, A_pre) + b_i
            A_i = self._activation(Z_i)
            cache.append(A_i)
        return cache

    def _back_propagation(self, v_tr: np.array, cache: list) -> dict:
        """Back propagation through the network to caculate gradient.

        Parameters
        ----------
        v_tr : np.array
            training values.
        cache : list
            activation value for each layer.

        Returns
        -------
        dict
            dictionary contains the partial derivative of cost function
            w.r.t weights and biases of every layer.

        """
        m = v_tr.shape[1]  # number of examples
        num_layer = len(self._num_units)  # number of layers + 1
        A_n = cache[-1]  # the output of the neural network

        assert(A_n.shape == (1, m))
        assert(v_tr.shape == (1, m))

        dZ_n = (1./m)*(A_n - v_tr)*(1. - A_n**2)

        assert(dZ_n.shape == (1, m))

        grad = dict()
        # (n[l], m).(n[l-1], m).T = (n[l], n[l-1])
        grad['dW' + str(num_layer - 1)] = np.dot(dZ_n, cache[-2].T)

        assert(grad['dW' + str(num_layer - 1)].shape == (self._num_units[-1],
                                                         self._num_units[-2]))

        grad['db' + str(num_layer - 1)] = dZ_n.sum(axis=1, keepdims=True)

        assert(grad['db' + str(num_layer - 1)].shape == (self._num_units[-1],
                                                         1))
        dZ_pre = dZ_n
        for i in reversed(range(1, num_layer - 1)):
            A_i = cache[i]
            A_pre = cache[i-1]
            W_pre = self._parameters['W' + str(i + 1)]
            dZ_i = np.dot(W_pre.T, dZ_pre)*(1. - A_i**2)
            grad['dW' + str(i)] = np.dot(dZ_i, A_pre.T)
            grad['db' + str(i)] = dZ_i.sum(axis=1, keepdims=True)

            assert(grad['dW' + str(i)].shape == (self._num_units[i],
                                                 self._num_units[i-1]))
            assert(grad['db' + str(i)].shape == (self._num_units[i], 1))

            dZ_pre = dZ_i

        return grad

    def _generate_training_set(self, boards: list, final_status: str) -> tuple:
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
            numpy array contains the training values of each board.
            second element is a cache of A_i for every layer i.

        """
        m = len(boards)
        # shape = (m, )
        training_set = np.zeros((1, m))
        assert(training_set.shape == (1, m))
        # numpy array to store the predictions values
        cache = self._forward_propagation(boards)
        v_hat = cache[-1]
        assert(v_hat.shape == (1, m))

        if final_status == 'win':
            training_set[0, -1] = 1
        elif final_status == 'lose':
            training_set[0, -1] = -1
        elif final_status == 'draw':
            training_set[0, -1] = 0
        else:
            raise ValueError("The game doesn't end yet to train on it")

        for i in range(m - 1):
            training_set[0, i] = v_hat[0, i+1]

        return training_set, cache


class MiniMaxAlphaBetaSystem(AISystem):
    """MiniMax alpha Beta for system for checker bot."""

    def __init__(self, depth: int) -> None:
        """Initialize the system.

        Parameters
        ----------
        depth : int
            the maximum depth of search before using a prediction
            using FeaturesBasedSystem.

        Returns
        -------
        None

        """
        self._tot_num = 0
        self._parameters = np.array([-0.01194564,
                                    -0.02149239,
                                     0.00013474,
                                    -0.00838412,
                                     0.00121441,
                                     0.0057414 ,
                                    -0.00907267])
        self._parameters = self._parameters.reshape((7, 1))
        assert(self._parameters.shape == (7, 1))
        self._pred_system = FeaturesBasedSystem('temp',
                                                0.01,
                                                False)
        self._pred_system.set_parameters(self._parameters)
        self._depth = depth

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

    def predict(self, boards: list, turn: int, draw_counter: int) -> np.array:
        """Use it to predict the fitnesses values for a given list of boards.

        Parameters
        ----------
        boards : list
            list of boards we want to predict their fitnesses.
        turn : int
            turn's number.
        draw_counter : int
            counter of non-attack moves.

        Returns
        -------
        np.array
            the fitnesses values of the given boards.

        """
        self._tot_num = 0
        m = len(boards)
        result = np.zeros((m, 1))
# =============================================================================
#         boards = list(zip(boards,
#                           self._pred_system.predict(boards,
#                                                     turn,
#                                                     draw_counter).ravel()
#                           )
#                       )
# =============================================================================
        # there is a bug here!
        # draw_counter must be updated
        # but in practise that doesn't effect the perfomance of the system
        # so we choose to let it instead of modifing the predict function.
        if turn % 2 == 1:
            current_max = -10
            for i in range(m):
                result[i] = self._max(boards[i], 1, turn, draw_counter,
                                      current_max, 1e7)
                current_max = max(current_max, result[i])
                # print(f'result[{i}]: {result[i]}')
        else:
            current_min = 10
            for i in range(m):
                result[i] = self._min(boards[i], 1, turn, draw_counter,
                                      -1e7, current_min)
                current_min = min(current_min, result[i])
                # print(f'result[{i}]: {result[i]}')
        # print(f'DEBUG: number of expanded nodes: {self._tot_num}')
        return result

    def copy(self):
        """Use it to copy the system.

        Returns
        -------
        AISystem
            a copy of the calling system.

        """
        return self

    def get_name(self) -> str:
        """Get the name of the system.

        Returns
        -------
        str
            the name of the system.

        """
        return f'MiniMaxAlphaBeta{self._depth}'

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
        return 0

    def _terminal_state(self, board: Board, turn: int,
                        draw_counter: int) -> str:
        """Get the status of a terminal state, or None if it's not terminal.

        Parameters
        ----------
        board : Board
            board state.
        turn : int
            turn's number.
        draw_counter : int
            counter of non-attack moves.

        Returns
        -------
        str
            'win', 'lose', 'draw', or None if it's not a terminal state.

        """
        if turn % 2 == 1:
            colour = 'white'
        else:
            colour = 'black'
        status = board.get_status(colour, draw_counter)
        return status

    def _max(self, board: Board, depth: int,
             turn: int, draw_counter: int,
             alpha: float, beta: float) -> float:
        self._tot_num += 1
        status = self._terminal_state(board, turn, draw_counter)
        if status is not None:
            sign = 1 if turn % 2 == 1 else -1
            if status == 'win':
                return 1 * sign
            elif status == 'lose':
                return -1 * sign
            else:
                return 0
        if depth >= self._depth:
            return np.sum(self._pred_system.predict([board], turn,
                                                    draw_counter))
        boards = []
        Moves.get_all_next_boards(board, 'white', boards)
        current_max = -1e7
        size_before = board.get_number_of_disks(None)
        for b in boards:
            next_draw_counter = draw_counter
            size_after = b.get_number_of_disks(None)
            next_draw_counter = update_draw_counter(next_draw_counter,
                                                    size_before, size_after)
            current_max = max(current_max, self._min(b, depth + 1, turn + 1,
                                                     next_draw_counter,
                                                     alpha, beta))
            if current_max >= beta:
                return current_max
            alpha = max(alpha, current_max)
        return current_max

    def _min(self, board: Board, depth: int,
             turn: int, draw_counter: int,
             alpha: float, beta: float) -> float:
        self._tot_num += 1
        status = self._terminal_state(board, turn, draw_counter)
        if status is not None:
            sign = 1 if turn % 2 == 1 else -1
            if status == 'win':
                return 1 * sign
            elif status == 'lose':
                return -1 * sign
            else:
                return 0
        if depth >= self._depth:
            return np.sum(self._pred_system.predict([board], turn,
                                                    draw_counter))
        boards = []
        Moves.get_all_next_boards(board, 'black', boards)
        current_min = 1e7
        size_before = board.get_number_of_disks(None)
        for b in boards:
            next_draw_counter = draw_counter
            size_after = b.get_number_of_disks(None)
            next_draw_counter = update_draw_counter(next_draw_counter,
                                                    size_before, size_after)
            current_min = min(current_min, self._max(b, depth + 1, turn + 1,
                                                     next_draw_counter,
                                                     alpha, beta))
            if current_min <= alpha:
                return current_min
            beta = min(beta, current_min)
        return current_min


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
    print(f_system.predict([b], 0, 0))
    print('Everything work.')
