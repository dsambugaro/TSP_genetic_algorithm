#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 21:33:23 2018

@author: dsambugaro
"""

import math
import random
import functools
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


def createRoute(routeNodes):
    '''
    Shuffles randomly the given array
    '''
    route = random.sample(routeNodes, len(routeNodes))
    return route

def initialPopulation(routeNodes, size=100):
    '''
    Generate initial population by randomly shuffling the route nodes.
    Default population size: 100
    '''
    population = []
    for i in range(0, size):
        population.append(createRoute(routeNodes))
    return population

def distances_between_points(p1, p2):
    '''
    Returns the square distance between two given points
    '''
    return np.dot((p1-p2),(p1-p2))

def fitness(route, cities):
    '''
    Calculates and return route fitness with base in the Euclidean distance.
    
    The fitness is given by:
        1 / log10(Euclidean distance)

    '''
    distances_between_cities = []
    
    for i in range(1, len(route)):
        if i != len(route): # distance between cities along route
            p1 = np.array(cities['COORD'][route[i-1]],cities['SECTION'][route[i-1]])
            p2 = np.array(cities['COORD'][route[i]],cities['SECTION'][route[i]])
            distances_between_cities.append(distances_between_points(p1,p2))
        else: # distance between first and last citie
            p1 = np.array(cities['COORD'][route[0]],cities['SECTION'][route[0]])
            p2 = np.array(cities['COORD'][route[i-1]],cities['SECTION'][route[i-1]])
            distances_between_cities.append(distances_between_points(p1,p2))
        
    euclidan_distance = math.sqrt(np.sum(distances_between_cities))
    
    fitness = 1 / np.log10(euclidan_distance)
    
    return fitness
    
    

def main():
    cities = pd.read_csv('data/a280.csv', ';')
    pop = initialPopulation(list(cities['NODE']), 15)
    pop = np.subtract(pop, 1)
    for i in range(len(pop)-1):
        ft = fitness(pop[i], cities)
        print(ft)
    

if __name__ == '__main__':
    main()