''' Genetic Algorithm '''


''' importing the libs & pkgs   '''
import copy
import numpy as np
import random
import rec
import sys
import time


''' main flow   '''
def run(map, totalPopulation):

    # set up vars for iteration
    totalCities = np.shape(map)[0] - 1
    bestPathLength = np.inf
    bestPath = None

    # initalise a new population
    pop = Population(totalPopulation, totalCities, map)
    pop.makeNewPopulation()

    while True:

        children = pop.getChildren()
        pop.setNewPopulation(children)

        # if there is a shorter path, then update it
        bestPopPathLength, bestPopPath = pop.shortest()
        if bestPathLength > bestPopPathLength:
            bestPathLength = bestPopPathLength
            bestPath = bestPopPath
            #print (bestPath, bestPathLength)

            # save results
            totalTime = pop.newPopTime[-1]
            rec.save('gen', bestPath, bestPathLength, totalTime, totalCities)



############################################################################


''' Represents one person in the population.
    A prepopulated random path representing a
    path travelled between all cities is set     '''
class Person:


    # makes a new person with a random path
    def __init__(self, totalCities):
        self.totalCities = totalCities

        # fitness of the person, relative to the inverse of total distance
        # of his path and the rest of the population. ie. it is normalised
        self.fitness = None


    def makePath(self):
        # base path to shuffle to get a random path
        basePath = np.arange(1, self.totalCities + 1, 1)

        # path travelled by a person is the base path shuffled 20 times
        # now is a random path
        self.path = self.mutatePath(basePath, 20)


    def setPath(self, path):
        self.path = path
        # reset fitness, needs to be recalculated bases on totalFitness
        self.fitness = None


    # returns a path shuffled so many times
    def mutatePath(self, path, times):
        shuffledPath = path.copy()

        for i in range(times):
            # get two random indexes
            index1 = random.randint(0, len(path) - 1)
            index2 = random.randint(0, len(path) - 1)

            # swap indexes
            temp = shuffledPath[index1]
            shuffledPath[index1] = shuffledPath[index2]
            shuffledPath[index2] = temp

        return shuffledPath


    # return a mixture of two paths
    # ie is child DNA based from two parents
    def crossPaths(self, pathA, pathB):

        # make a new path of some slice of a parents path
        startIndex = random.randint(0, len(pathA))
        endIndex = random.randint(startIndex, len(pathA) + 1)
        newPath = pathA[startIndex: endIndex]

        # add remaining cities to the path in the order of the others parent's
        for city in pathB:
            if not city in newPath:
                newPath = np.append(newPath, city)

        return newPath


    # fitness can be equated to the inverse of distance travelled along each person's path
    # computes distance by iterating through path on the map
    def getDistanceTravelled(self, map):

        # var to store distance
        distanceTravelled = 0

        # iterate through each step between cities
        for i in range (1, self.totalCities):
            start = self.path[i - 1]                # starts from
            end = self.path[i]                      # travels to
            distanceTravelled += map[start, end]    # cumulative distance over path

        return distanceTravelled


############################################################################


''' Represents the entire population of people.
    Each person with a randomised path              '''
class Population:


    # total number of people in population
    # total number of cities given in search problem
    # catalogue is an array of all persons in the population
    def __init__(self, totalSize, totalCities, map):

        self.totalSize = totalSize
        self.totalCities = totalCities
        self.map = map
        self.catalogue = None
        self.newPopTime = np.array([])


    # populate poplation with people of random paths
    def makeNewPopulation(self):
        # array of totalSize many people
        catalogue = np.array([])

        for i in range(self.totalSize):
            person = Person(self.totalCities)
            person.makePath()
            catalogue = np.append(catalogue, person)

        self.setNewPopulation(catalogue)


    # set poplation with predefined children
    def setNewPopulation(self, catalogue):
        self.catalogue = catalogue
        self.setPopFitness()


    # returns new population based on fittest of old population
    def getChildren(self):

        startTime = time.time()

        # catalogue for new children
        nexGen = np.array([])

        # replace each member of the population
        while len(nexGen) < len(self.catalogue):

            # new child
            child = Person(self.totalCities)

            # two paths of fit parents
            fitA = self.getFitPerson()
            fitB = self.getFitPerson()

            # mixture of two paths
            child.path = child.crossPaths(fitA.path, fitB.path)

            # only keep child if as parent
            parentsPathLengths = [fitA.getDistanceTravelled(self.map), fitB.getDistanceTravelled(self.map)]

            if child.getDistanceTravelled(self.map) < min(parentsPathLengths):
                nexGen = np.append(nexGen, child)
            else:
                # if not as fit, mutate and try again
                child.path = child.mutatePath(child.path, 1)
                if child.getDistanceTravelled(self.map) < min(parentsPathLengths):
                    nexGen = np.append(nexGen, child)

        self.newPopTime = np.append(self.newPopTime, time.time() - startTime)

        return nexGen


    # ROULETTE WHEEL IMPLEMENTATION
    # picks with higher probability of getting person with higher fitness
    # returns a fit person
    def getFitPerson(self):

        totalPopulation = len(self.catalogue)

        # list of all fitnesses of poplation
        fitnesses = [person.fitness for person in self.catalogue]

        index, rand = 0, random.random()

        # move up the stack the amount of the random number
        while rand > 0:

            # get index corresponding to fitness at fitness height in stack
            rand -= fitnesses[index]
            index += 1

        # when random number reaches 0, we are the height
        index -= 1

        # get fitness with more chance of getting higher fitness
        selectedFitness = fitnesses[index]

        # index of the selected person
        selectedIndex = [i for i, person in enumerate(self.catalogue) if person.fitness == selectedFitness]

        # a copy of the selected person
        return copy.copy(self.catalogue[selectedIndex[0]])


    # GAUSSIAN BIAS TOWARDS FIT PEOPLE - NOT IN WORKING SOLUTION
    # if fitnesses are sorted in descending order, a gaussian function with mean = 0
    # is more likely to pick someone closer to the start of the list
    # returns a fit person
    def getFitPersonGaussian(self, variance):

        # list of all fitnesses of poplation
        fitnesses = [person.fitness for person in self.catalogue]

        # gets index of poplation based on gaussian dist.
        # ie. if sorted by fitness in desc. more likely to pick a 0 or lower numbers
        randIndex = int(abs(random.gauss(0, variance)))

        while randIndex >= len(fitnesses):
            # get randomIndex less than length of fitness
            randIndex = int(abs(random.gauss(0, variance)))

        # sort list in descending order
        sortedFitnesses = sorted(fitnesses, reverse=True)

        # get fitness with more chance of getting higher fitness
        selectedFitness = sortedFitnesses[randIndex]

        # index of the selected person
        selectedIndex = [i for i, person in enumerate(self.catalogue) if person.fitness == selectedFitness]

        # a copy of the selected person
        return copy.copy(self.catalogue[selectedIndex[0]])


    # sets normalised fitness of each person based on total fitness of population
    def setPopFitness(self):

        totalFitness = np.sum([1. / person.getDistanceTravelled(self.map) for person in self.catalogue])

        for person in self.catalogue:
            # ie. normalised fitness = unnormalised fitness / totalFitness
            person.fitness = 1. / (totalFitness * person.getDistanceTravelled(self.map))


    def shortest(self):

        # list of all distances travelled by each person
        pathLengthList = [person.getDistanceTravelled(self.map) for person in self.catalogue]

        # smallest path length
        bestPathLength = min(pathLengthList)

        # index of smallest path length
        minPathIndex = [i for i in range(len(pathLengthList)) if pathLengthList[i] == bestPathLength][0]

        # best path
        bestPath = [person.path for person in self.catalogue][minPathIndex]

        return bestPathLength, bestPath
