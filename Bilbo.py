#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 21:33:23 2018

@author: dsambugaro
"""

import math
import random
from sys import argv

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

def euclidian_distance_calc(route, cities):
    '''
    Return euclidian distance between the cities of given route
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
        
    euclidian_distance = math.sqrt(np.sum(distances_between_cities))
    return euclidian_distance

def fitness(route, cities):
    '''
    Calculates and return route fitness with base in the Euclidean distance.
    
    The fitness is given by:
        1 / log10(Euclidean distance)

    '''
    
    euclidian_distance = euclidian_distance_calc(route, cities)
    fitness = 1 / np.log10(euclidian_distance)
    
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
            positions_ordered.append(subset.index(node))
    
    child = parent_1
    for i in range(len(subset)):
        child[random_positions[i]] = subset[positions_ordered[i]]
    
    return child

def alternativeCrossover(parent_1, parent_2):
    '''
    Returns a child created with an alternative crossover
    '''
      
    child = list(parent_1)
    cut = random.randrange(0,len(parent_1)-1)
    
    for i in range(cut, len(child)):
        if parent_1[i] not in parent_2[cut:] and parent_2[i] not in child[:cut]:
            child[child.index(parent_2[i])] = child[i]
            child[i] = parent_2[i]
    
    return child

def mutate_v1(child):
    '''
    Swap mutation: swap positions of two genes
    '''
    index_1 = random.randrange(0,len(child)) # Get a random position
    index_2 = random.randrange(0,len(child)) # Get another random position

    
    if index_1 != index_2:
        aux = child[index_1]
        child[index_1] = child[index_2]
        child[index_2] = aux
    else:                                        # Equal positions
        mutate_v1(child)
    return child

def mutate_v2(child):
    '''
    Scramble mutation: A variable set of genes is chosen, shuffled, and
    inserted back into the individual.
    '''

    index_1 = random.randrange(0,len(child)) # Random initial position
    index_2 = random.randrange(0,len(child)) # Random final position

    if (index_1 > index_2) or ((index_2 - index_1) < 1):
        mutate_v2(child)
    
    else:
        aux = list(child[index_1:index_2+1])
        aux1 = aux
        aux = random.sample(aux,len(aux))
        if aux == aux1:
            while aux == aux1:
                aux = random.sample(aux,len(aux))                

        j = 0
        for i in range(index_1,(index_2+1)):
            child[i] = aux[j]
            j = j+1
   
    
    return child

def acumulate(v): 
    '''
    Creates the accumulated distribution table
    '''
    acum = 0
    r = []
    for i in v:
        acum += i
        r.append(acum)
    return r   


def pick_parent(pop, cities, fitness_fn):
    '''
    Selects an individual to be a parent, based on roulette technique
    '''
    
    fits = []
    for i in range(len(pop)):
        fits.append(fitness_fn(pop[i],cities))

    fits_sum = sum(fits)
    norm = map(lambda x: x / fits_sum, fits)
    acum = acumulate(norm)
    r = np.random.uniform()
    
    for i in range(len(acum)):
        if r < acum[i]:
            return pop[i]

def argmax(vetor):
    '''
    Returns the index of the bigger element of a given array
    '''
    return np.argsort(vetor)[-1]

def genect(cities, pop_size, crossover, mutation, chance_multation=0.10, elitsm=0, fitness_fn=fitness, k_gen=10):
    pop = initialPopulation(list(cities['NODE']), pop_size)
    
    # Subtract necessary because this will be used as indices in arrays
    pop = np.subtract(pop, 1)
    
    fits = []
    for i in range(len(pop)):
        fits.append(fitness(pop[i],cities))
    
    best_solution = pop[argmax(fits)]
    max_solution = []
    avg_solution = []
    number_generations = 0
    
    last_improvement = 0
    
    while last_improvement < k_gen:
        number_generations += 1
        print('Generation: {}'.format(number_generations))
        new_pop = []
        for individual in pop:
            parent_1 = pick_parent(pop, cities, fitness_fn)
            parent_2 = pick_parent(pop, cities, fitness_fn)
            if crossover == 'ordered_v1':
                child = orderedCrossover_v1(parent_1, parent_2)
            elif crossover == 'ordered_v2':
                child = orderedCrossover_v2(parent_1, parent_2, 3)
            elif crossover == 'alternative':
                child = alternativeCrossover(parent_1, parent_2)
            else:
                raise Exception('Crossover ' + crossover + ' not implemented yet')
            if np.random.uniform() < chance_multation:
                if mutation == 'swap':
                    child = mutate_v1(child)
                elif mutation == 'scramble':
                    child = mutate_v2(child)
                else:
                    raise Exception('Mutation ' + mutation + ' not implemented yet')
            
            new_pop.append(child)
            
        if elitsm:
            fits = []
            for i in range(len(pop)):
                fits.append(fitness(pop[i],cities))
            best_individuals = np.argsort(fits)
            best_individuals.reverse()
            
            for i in range(elitsm):
                pop[i] = pop[best_individuals[i]]
            
            for i in range(elitsm+1, len(new_pop)):
                pop[i] = new_pop[i]
                    
        else:
            pop = new_pop
        
        fits = []
        for i in range(len(pop)):
            fits.append(fitness_fn(pop[i],cities))
            
        new_best_solution = pop[argmax(fits)]
        print('Generation Fitness: {}'.format(fitness_fn(new_best_solution, cities)))
        
        if fitness_fn(new_best_solution, cities) > fitness_fn(best_solution, cities):
            best_solution = new_best_solution
            last_improvement = 0
        else:
            last_improvement += 1
        
        max_solution.append(fitness_fn(new_best_solution, cities))        
        
        fits = []
        for i in range(len(pop)):
            fits.append(fitness_fn(pop[i],cities))
        fits_sum = sum(fits)        
        
        avg_solution.append(fits_sum / len(pop))
        
    plt.figure()
    plt.plot(range(number_generations), max_solution, label='Max fitness')
    plt.plot(range(number_generations), avg_solution, label='Avg fitness')
    plt.legend()
    plt.ylabel('Fitness')
    plt.xlabel('Generation')
    plt.show()
    plt.savefig('results/popSize_{}_crossover_{}_mutation_{}_estag_{}_elitms_{}.pdf'.format(pop_size, crossover, mutation, k_gen, elitsm))
    
    return best_solution
            

def main():
    cities = pd.read_csv('data/a280.csv', ';')
    pop_size = 15
    estag = 100
    
    best_way = genect(cities, pop_size, 'alternative', 'swap', k_gen=estag)
    print('pop inicial 100 | Crossosever alternativo | mutação swap | estagnação 20 | sem elitismo')
    print('Distance: ')
    print(euclidian_distance_calc(best_way, cities))
    print('\n\n')
    
    best_way = genect(cities, pop_size, 'alternative', 'scramble', k_gen=estag)
    print('pop inicial 100 | Crossosever alternativo | mutação scramble | estagnação 20 | sem elitismo')
    print('Distance: ')
    print(euclidian_distance_calc(best_way, cities))
    print('\n\n')
    
    best_way = genect(cities, pop_size, 'ordered_v2', 'swap', k_gen=estag)
    print('pop inicial 100 | Crossosever ordered_v2 | mutação swap | estagnação 20 | sem elitismo')
    print('Distance: ')
    print(euclidian_distance_calc(best_way, cities))
    print('\n\n')
    
    best_way = genect(cities, pop_size, 'ordered_v2', 'scramble', k_gen=estag)
    print('pop inicial 100 | Crossosever ordered_v2 | mutação scramble | estagnação 20 | sem elitismo')
    print('Distance: ')
    print(euclidian_distance_calc(best_way, cities))
    print('\n\n')
    
if __name__ == '__main__':
    main()