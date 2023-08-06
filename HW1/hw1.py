"""
File: HW1 - CSCI-630: Intro to AI
Author: Divyank Kulshrestha, dk9924@rit.edu
Description: find the shortest list of words that connects the two given words.
Language: Python3

NOTE: Using graph and BFS implementation from the course CSCI603.
"""

import sys
from graph import Graph
from searchAlgos import findShortestPath

def findPath(fileName, startWord, targetWord):
    """
        Creates a graph of words on the dictionary and finds the shortest path connecting two words

        :param fileName: name/path of the dictionary file
        :param startWord: word given to start with
        :param targetWord: the final word to reach

        :return: list of vertices in the shortest path
    """
    with open(fileName) as f:
        allWords = f.read().splitlines()
    wordGraph = Graph()
    for word in allWords:
        wordGraph.addVertex(word)
        for vertex in wordGraph.getVertices():
            if len(word) == len(vertex) and connectWords(word, vertex):
                wordGraph.addEdge(word, vertex)
                wordGraph.addEdge(vertex, word)

    return findShortestPath(wordGraph.getVertex(startWord), wordGraph.getVertex(targetWord))

def connectWords(wordOne, wordTwo):
    """
        Tells us whether two words should be connected on the graph.
        Criteria for connecting: same length and ONLY one character difference

        :param wordOne: first word to connect
        :param wordTwo: second word to connect

        :return: boolean value to decide if connection btw words should exist
    """
    connect = False
    for a, b in zip(wordOne, wordTwo):
        if a != b:
            if connect == False:
                connect = True
            else:
                return False
    return connect

if __name__ == '__main__':
    fileName = sys.argv[1]
    startWord = sys.argv[2]
    targetWord = sys.argv[3]
    path = findPath(fileName, startWord, targetWord)
    if path == None:
        print("No solution")
    else:
        for v in path:
            print(v.id)
