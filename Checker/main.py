# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 23:04:20 2020

@author: Kais
"""

from Checker.AI.AISystems import FeaturesBasedSystem
from Checker.AI.Agent import Agent
from Checker.Engine import train, play, test_agents


if __name__ == '__main__':

    def get_agent(name: str, learning_rate: float,
                  features: list,
                  use_saved_parameters: bool = True) -> Agent:
        system = FeaturesBasedSystem(name, learning_rate,
                                     use_saved_parameters, *features)
        agent = Agent(colour='white', system=system)
        return agent

# =============================================================================
#     ag1 = get_agent('tom_agent_test', 0.001, [])
#     # train(ag1, 1000)
#     # ag2 = get_agent('tom_agent', 0.001, [])
#     # test_agents(ag1, ag2, 100)
# 
#     import sys
#     sys.exit()
# =============================================================================

    print('Welcome to the Checker Game')
    print('please note that white is always plays first')

    print('colours available for you are: white, black')
    while True:
        colour = input('enter the colour you want to play with: ')
        colour = colour.lower()
        if colour in ['white', 'black']:
            break
        else:
            print('please enter a valid colour!')
    agent_colour = 'white' if colour == 'black' else 'black'
    print(f'agent colour: {agent_colour}')
    tom_agent = get_agent('tom_agent', 0.001, [])
    tom_agent.set_colour(agent_colour)
    play(tom_agent)
