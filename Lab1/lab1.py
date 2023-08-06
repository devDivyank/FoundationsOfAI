"""
File: Lab1 - CSCI-630: Intro to AI
Author: Divyank Kulshrestha, dk9924@rit.edu
Description: generating optimal paths for orienteering
Language: Python3

NOTE: Using graph, heap and BFS implementation from the course CSCI603
        and then modifying it as required.
"""

import math
import sys
from PIL import Image
from graph import Graph
from vertex import Vertex
from heap import Heap

def generateOptimalPath(terrainImage, elevationFile, pathPointsFile, outputImageFileName):
    """
        Creates a graph using elevation and terrain and finds the optimal
        path to be printed on the terrain image

        :param terrainImage: image of the terrain
        :param elevationFile: word given to start with
        :param pathPointsFile: the final word to reach
        :param outputImageFileName: the final word to reach

        :return: list of vertices in the shortest path
    """
    terrainTypes = {
        (248, 148, 18): "Open Land",
        (255, 192, 0): "Rough Meadow",
        (255, 255, 255): "Easy Movement Forest",
        (2, 208, 60): "Slow Run Forest",
        (2, 136, 40): "Walk Forest",
        (5, 73, 24): "Impassible Vegetation",
        (0, 0, 255): "Lake/Swamp/Marsh",
        (71, 51, 3): "Paved Road",
        (0, 0, 0): "Footpath",
        (205, 0, 101): "Out Of Bounds"
    }
    terrainSpeeds = {
        'Open Land': 7,
        'Rough Meadow': 4,
        'Easy Movement Forest': 6,
        'Slow Run Forest': 5,
        'Walk Forest': 3,
        'Impassible Vegetation': 2,
        'Lake/Swamp/Marsh': 1,
        'Paved Road': 8,
        'Footpath': 9,
        'Out Of Bounds': 0
    }

    # opening the terrain image
    with Image.open(terrainImage) as terrainImage:
        terrainPixels = terrainImage.load()
    # opening the elevation data file and converting to 2D array
    with open(elevationFile) as elevationFile:
        elevationPoints = [list(map(float, line.strip().split())) for line in elevationFile]
    # opening the path data file and converting to 2D array
    with open(pathPointsFile) as pathFile:
        checkpoints = [list(map(int, line.strip().split())) for line in pathFile]
        # ignoring the last 5 values in each row
        for i in range(len(elevationPoints)):
            elevationPoints[i] = elevationPoints[i][:-5]

    # creating a graph, with each vertex representing a pixel (each having elevation and terrain type)
    terrainGraph = Graph()
    for i in range(len(elevationPoints)):
        for j in range(len(elevationPoints[i])):
            terrainGraph.addVertex((j, i), elevationPoints[i][j], terrainTypes[terrainPixels[j, i][0:3]])

    # adding the edges from each pixel (vertex) to 8 neighbouring pixels only if src and dest are not "Out Of Bounds"
    for src in terrainGraph:
        x = src.key[0]
        y = src.key[1]
        if src.terrainType != "Out Of Bounds":
            for i in range(-1,2):
                for j in range(-1,2):
                    if terrainGraph.getVertex((x+i,y+j)) and  terrainGraph.getVertex((x+i,y+j)) != src and \
                            terrainGraph.getTerrainType((x+i,y+j)) != "Out Of Bounds":
                        dest = terrainGraph.getVertex((x+i,y+j))
                        cost = getDistance3D(src, dest) / terrainSpeeds[dest.terrainType]
                        terrainGraph.addEdge(src.key, dest.key, cost)

    # finding and appending each optimal path between two checkpoints to the total path
    totalPath = []
    for i in range(len(checkpoints) - 1):
        start = terrainGraph.getVertex((checkpoints[i][0], checkpoints[i][1]))
        end = terrainGraph.getVertex((checkpoints[i+1][0], checkpoints[i+1][1]))
        currentPath = findShortestPath(start, end)
        totalPath.extend(currentPath)

    # calculating the total length of the path
    totalPathLength = 0
    for i in range(len(totalPath)-1):
        totalPathLength += getDistance3D(totalPath[i], totalPath[i+1])

    # changing the colour of pixels in the total path on the output image
    for point in totalPath:
        x = point.key[0]
        y = point.key[1]
        terrainPixels[x, y] = (255, 0, 0, 255)

    # changing the colour of the checkpoints for better visibility on the output image
    for point in checkpoints:
        x = point[0]
        y = point[1]
        for i in range(-1,2):
            for j in range(-1,2):
                try:
                    terrainPixels[x+i, y+j] = (0, 255, 255, 255)
                except:
                    continue

    # saving the final output image
    terrainImage.save(outputImageFileName)
    print("Image saved!")
    print("Total Path Length = " + str(totalPathLength))


def getDistance3D(src: Vertex, dest: Vertex):
    """
        Calculates the straight line distance between two points with different co-ordinates and elevations

        :param src: the first point (a vertex in the graph)
        :param dest: the second point (a vertex in the graph)

        :return: the straight line distance between the points
    """
    elevationDifference = abs(src.elevation - dest.elevation)
    # getting real world size of one pixel which is one third of an arc-second
    # equivalent to 10.29 m in longitude (X) and 7.55 m in latitude (Y))
    a = (dest.key[0] - src.key[0]) * 10.29
    b = (dest.key[1] - src.key[1]) * 7.55
    distance = math.sqrt((a ** 2) + (b ** 2) + (elevationDifference ** 2))
    return distance


def calculateHeuristic(src: Vertex, dest: Vertex):
    """
        Calculates the heuristic value (straight line distance between point and goal by maximum speed possible)

        :param src: the first point (a vertex in the graph)
        :param dest: the second point (will always be the final destination/vertex)

        :return: the straight line distance between the points
    """
    distance = getDistance3D(src, dest)
    heuristicValue = distance / 9              # maximum speed possible = 9 (on footpath)
    return heuristicValue


def findShortestPath(start: Vertex, end: Vertex):
    """
    Find the optimal path by simulating A* algo, between a start and end vertex
    :param start: the start vertex
    :param end: the destination vertex

    :return: A list of Vertex objects from start to end, if a path exists,
        otherwise None
    """
    # to store g and f values for each vertex we encounter
    gValues = {start: 0}
    fValues = {start: calculateHeuristic(start, end)}

    # Using a heap as the dispenser type will result in a priority queue implementation
    # function to compare values before inserting into the heap to maintain its order
    def lessfn(v1, v2):
        return fValues[v1] < fValues[v2]
    queue = Heap(lessfn)
    queue.insert(start)

    # The predecessor dictionary maps the current Vertex object to its
    # immediate predecessor.  This collection serves as both a visited
    # construct and a way to find the path
    predecessors = {}
    predecessors[start] = None  # add the start vertex with no predecessor

    # Loop until either the queue is empty, or the end vertex is encountered
    while len(queue) > 0:
        # popping from heap gives us the vertex with lowest fValue
        current = queue.pop()
        if current == end:
            break
        for neighbor in current.getConnections():
            # if we visit a vertex first time or find a lower fValue for an earlier vertex,
            # we update the gvalue and fValue, store new predecessor for the vertex,
            # and insert it back in the heap
            if neighbor not in predecessors or gValues[neighbor] > gValues[current] + current.getWeight(neighbor):
                predecessors[neighbor] = current
                gValues[neighbor] = gValues[current] + current.getWeight(neighbor)
                fValues[neighbor] = gValues[neighbor] + calculateHeuristic(neighbor, end)
                queue.insert(neighbor)

    # If the end vertex is in predecessors, a path was found
    # tracing the path and the total path length
    if end in predecessors:
        path = []
        current = end
        while current != start:                     # loop backwards from end to start
            path.insert(0, current)                 # prepend current to the path list
            current = predecessors[current]         # move to the predecessor
        path.insert(0, start)
        return path
    else:
        return None, 0

if __name__ == '__main__':
    terrainImage = sys.argv[1]
    elevationFile = sys.argv[2]
    pathFile = sys.argv[3]
    outputFileName = sys.argv[4]
    generateOptimalPath(terrainImage, elevationFile, pathFile, outputFileName)
