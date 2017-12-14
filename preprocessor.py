''' File Processor '''
''' Works with txt search files, maps and odds	'''


''' importing the libs & pkgs '''
import numpy as np
import re


''' returns a map matrix after loading and checking the txt input file '''
def getMap (fileName):


	''' reading the file '''
	with open(fileName, 'r') as searchFile:
		data = searchFile.read().replace('\t', '').replace('\n', '')
	parsedData = ''.join(data.split()).split(',')
	name = str(parsedData[0].split('=')[1])
	size = int(parsedData[1].split('=')[1])


	''' checking the file structure 	'''
	# setup a list of triangular numbers up to size
	listOfTriangles = [1]
	for i in range (2, size):
		last = listOfTriangles[-1]
		next = last + i
		listOfTriangles.append(next)

	# check if list-of-integers is triagular shaped
	listOfDistances = parsedData[2 : ]
	expectedTriangleNumber = listOfTriangles[-1]
	if len(listOfDistances) != expectedTriangleNumber:
		raise ValueError('The list-of-integers in the read file is not complete!')


	''' the distance matrix '''
	map = np.zeros((size + 1, size + 1))

	# populate matrix row by row from list-of-integers
	difference = size - 1
	startIndex = 0
	endIndex = startIndex + difference

	numbExpression = re.compile(r'[^\d.]+')

	for y in range (0, size):
		for x in range (0, len(listOfDistances[startIndex : endIndex])):

			# remove non numerical chars from list-of-integers
			dirt = listOfDistances[startIndex : endIndex][len(listOfDistances[startIndex : endIndex]) - 1 - x]
			clean = numbExpression.sub('', dirt)

			map[y + 1][size - x] = float(clean)

		startIndex = endIndex
		difference -= 1
		endIndex = startIndex + difference

	# adding labels to left and top most axes and making symetrical
	map[:, 0] = np.arange(0, size + 1)
	map = np.transpose(map) + map

	return map


''' returns the sum of all the roads in the map	'''
def getTotalDistance(map):
	return np.sum(map[1:, 1:]) / 2.0
