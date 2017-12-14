''' Ant Colony Optimization Algorithm '''


''' importing the libs & pkgs   '''
import numpy as np
#import pandas as pd
import random
import sys
import time


''' main flow   '''
''' finds best path from all cities given the boundary conditions    '''
def run(distanceMap, maxIterations, antNumber, pheromoneRegulator, visibilityRegulator, updateConst, pheromoneDecay):

    cityNumber = np.shape(distanceMap)[0] - 1
    cities = np.arange(1, cityNumber + 1, 1)

    # to record best
    bestPath = None
    bestPathLength = np.inf

    # need to check tours for all startingCities
    for startingCity in cities:
        #print ('startingCity:', startingCity)

        # contains pheromone level of each path between cities
        pheromoneMap = np.ones_like(distanceMap)

        # for each iteration of all ants' complete tours
        for it in range(maxIterations):
            #print ('It:', it)

            # store paths to update pheromoneMap after each iteration
            paths = []

            # for each ant's complete tour
            for ant in range(antNumber):
                #print ('ant:', ant)

                # new ant
                a = Ant(cityNumber, startingCity, pheromoneRegulator, visibilityRegulator)

                # ants make a tour
                while len(a.path) < cityNumber:

                    # update path following next chosen city
                    a.path = np.append(a.path, a.getNextCity(pheromoneMap, distanceMap, a.path))

                paths.append(a.path)

                # update best tour
                pathLength = getPathLenth(a.path, distanceMap)
                if pathLength < bestPathLength:
                    bestPathLength = pathLength
                    bestPath = a.path
                    #print (bestPath, bestPathLength)

                #print (bestPath, bestPathLength)
                #print ()

            # update pheromone trails after all ants have toured
            pheromoneMap = updatePheromoneMap(paths, pheromoneMap, updateConst, distanceMap, pheromoneDecay)

            #print (pd.DataFrame(pheromoneMap))
            #print (bestPath, bestPathLength)

    return bestPath, bestPathLength


# updates the record of the pheromone trial for each complete path of an ant
def updatePheromoneMap(paths, pMap, updateConst, distanceMap, pheromoneDecay):

    pheromoneMap = pMap * (1 - pheromoneDecay)

    # for each path
    for path in paths:

        pathLength = float(getPathLenth(path, distanceMap))

        # iterate through each step between cities
        for i in range (1, len(path)):

            start = path[i - 1]                # starts from
            end = path[i]                      # travels to

            # set pheromone level for each path between cities
            pheromoneMap[start, end] += updateConst / pathLength
            pheromoneMap[end, start] += updateConst / pathLength

    return pheromoneMap


# returns the path length of a path
def getPathLenth(path, distanceMap):
    # var to store distance
    distanceTravelled = 0

    # iterate through each step between cities
    for i in range (1, len(path)):
        start = path[i - 1]                             # starts from
        end = path[i]                                   # travels to
        distanceTravelled += distanceMap[start, end]    # cumulative distance over path

    return distanceTravelled



# represents an ant with ability to choose next path based on factors
# and ability to remember tour of current state
class Ant:

    # each ant must have the same starting city
    # each iterate must be same, gives best tour from given starting position
    def __init__(self, cityNumber, startingCity, pheromoneRegulator, visibilityRegulator):

        self.path = np.array([startingCity])
        self.cityNumber = cityNumber
        self.pheromoneRegulator = pheromoneRegulator
        self.visibilityRegulator = visibilityRegulator


    # returns probality of travelling to next based on the maps and current path of the ant
    def probabilityTravel(self, pheromoneMap, distanceMap, path, next):

        cities = np.arange(1, self.cityNumber + 1, 1)
        allowedCities = np.array([city for city in cities if not city in path])
        currentCity = path[-1]

        # check if next has been visited yet
        if not next in allowedCities:
            return 0

        pheromoneWeight = lambda nextCity: (pheromoneMap[currentCity, nextCity] ** self.pheromoneRegulator)
        visibilityWeight = lambda nextCity: (distanceMap[currentCity, nextCity] ** np.negative(self.visibilityRegulator))

        weightedProduct = lambda nextCity: pheromoneWeight(nextCity) * visibilityWeight(nextCity)
        sumOfWeightedProducts = np.sum([weightedProduct(possibleNextCity) for possibleNextCity in allowedCities])

        return weightedProduct(next) / float(sumOfWeightedProducts)


    # get the next city to travel to based on the probalities of travelling to them
    # roulette wheel implementation
    def getNextCity(self, pheromoneMap, distanceMap, path):

        cities = np.arange(1, self.cityNumber + 1, 1)
        allowedCities = np.array([city for city in cities if not city in path])

        probalities = [self.probabilityTravel(pheromoneMap, distanceMap, path, possibleNextCity) for possibleNextCity in allowedCities]

        index, rand = 0, random.random()

        # move up the stack the amount of the random number
        while rand > 0:

            # get index corresponding to fitness at fitness height in stack
            rand -= probalities[index]
            index += 1

        # when random number reaches 0, we are the height
        index -= 1

        return allowedCities[index]
