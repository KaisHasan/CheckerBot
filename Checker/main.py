# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 23:04:20 2020

@author: Kais
"""

from Checker.AI.AISystems import FeaturesBasedSystem
from Checker.AI.Agent import Agent
from Checker.Engine import Train, Play

if __name__ == '__main__':
    system = FeaturesBasedSystem(name='test_1', learning_rate=0.01,
                                 useSavedParameters=True)
    agent = Agent(colour='white', system=system)
    costs, results = Train(agent=agent, num_of_games=10, output=False)
    # Play(agent)
    tot = 10
    wins = results.count(1)
    loses = results.count(-1)
    draws = results.count(0)
    print(f'wins: {wins}')
    print(f'loses: {loses}')
    print(f'draws: {draws}')
    print(costs)
