'''	AI Search Main  '''
'''	Python 3.5.2    '''


''' importing the libs & pkgs '''
import numpy as np
import preprocessor
import rec
import sys
import time
import urllib


''' get map from preprocessor '''
#testcase#012#017#021#026#042#048#058#175#180#535
sf = str(sys.argv[1])
map = preprocessor.getMap('searchfiles/AISearchfile' + sf + '.txt')
#print (map)
#print ('~~~~~~~~map~~~~~~~~~')


''' Genetic '''
def runGen():

    cities = np.shape(map)[0] - 1
    populationSize = int(50 * (np.log(cities) + 1))

    bestPath, shortestPathLength = gen.run(map, populationSize)



''' ACO '''
def runAnt():

    totalDistances = preprocessor.getTotalDistance(map)
    cities = np.shape(map)[0] - 1

    maxIterations = 10
    antNumber = 10 * cities
    updateConst = 0.05 * totalDistances / (cities - 1.0)
    pheromoneRegulator = 1
    visibilityRegulator = 1
    pheromoneDecay = 0.1


    startTime = time.time()

    bestPath, shortestPathLength = ant.run(
        map, maxIterations, antNumber, pheromoneRegulator, visibilityRegulator, updateConst, pheromoneDecay)

    totalTime = time.time() - startTime

    #print (totalTime, shortestPathLength)
    #print (bestPath)

    rec.save('ant', bestPath, shortestPathLength, totalTime, cities)



''' running algorithms '''
alg = str(sys.argv[2])

if alg == 'ant':
    import ant
    runAnt()

elif alg == 'gen':
    import gen
    runGen()
