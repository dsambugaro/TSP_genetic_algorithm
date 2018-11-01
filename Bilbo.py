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


def createRoute(cityList):
    route = random.sample(cityList, len(cityList))
    return route


def initialPopulation(k_size, cityList):
    population = []
    for i in range(0, k_size):
        population.append(createRoute(cityList))
    return population

def getDistance(cities):
    

def main():
    cities = pd.read_csv('data/a280.csv', ';')
    initialPopulation(1, list(cities['NODE']))
    

if __name__ == '__main__':
    main()