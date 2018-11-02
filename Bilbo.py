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
    Shuffles randomly the given route
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

def orderedCrossover_v1(parent_1, parent_2):
    '''
    A version of ordered crossover that returns a valid child for TSP
    This implementation works that way:
    
    Consider the parents:
        parent_1 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        parent_2 = [1, 3, 5, 7, 9, 2, 4, 6, 8]
        
    Randomly selects a subset of parent_1:
        parent_1 = [1, 2, | 3, 4, 5, 6, | 7, 8, 9]
    Then creates a 'offspring':
        offspring = [x, x, 3, 4, 5, 6, x, x, x ]
    And fill the Xs with cities of parent_2, in order they appear and without
    duplicating any city:
        offspring = [1, 7, 3, 4, 5, 6, 9, 2, 8]
    '''
    
    gene_1 = 0
    gene_2 = 0
    
    # Takes at least 40% and at most 70% of cities in the first parent
    while abs(gene_1-gene_2) < int(len(parent_1)*0.4):
        gene_1 = np.random.randint(0, int(len(parent_1)*0.7)+1)
        gene_2 = np.random.randint(0, int(len(parent_1)*0.7)+1)
    
    # Selects first and last positions of subset
    first_position = min(gene_1, gene_2)
    last_position = max(gene_1, gene_2)
    
    # Creates a offspring without repetition
    child = []
    for i in range(len(parent_2)):
        if parent_2[i] not in parent_1[first_position:last_position+1]:
            child.append(parent_2[i])
        if i in range(first_position, last_position+1):
            child.insert(i,parent_1[i])

    return child

def orderedCrossover_v2(parent_1, parent_2, random_qnt):
    '''
    A version of ordered crossover that returns a valid child for TSP.
        
    '''
    random_positions = random.sample(range(len(parent_1)), random_qnt)
    random_positions.sort()
    
    subset = []
    for i in range(len(random_positions)):
        subset.append(parent_1[random_positions[i]])
    
    positions_ordered = []
    for node in parent_2:
        if node in subset:
            positions_ordered.append(parent_2.index(node))
    
    child = parent_1
    for i in range(len(subset)):
        child[random_positions[i]] = subset[positions_ordered[i]]
    
    return child

def main():
    cities = pd.read_csv('data/a280.csv', ';')
    pop = initialPopulation(list(cities['NODE']), 15)
    pop = np.subtract(pop, 1)
    #for i in range(len(pop)-1):
    #    fits = fitness(pop[i], cities)
    p1 = [1,2,3,4,5,6,7,8,9]
    p2 = [1,3,5,7,9,2,4,6,8]
    print('Parent 1:', end=' ')
    print(*p1)
    print('Parent 2:', end=' ')
    print(*p2)
    orderedCrossover_v1(p1,p2)

if __name__ == '__main__':
    main()