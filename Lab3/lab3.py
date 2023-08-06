"""
    File: Lab3 - CSCI-630: Intro to AI
    Author: Divyank Kulshrestha, dk9924@rit.edu
    Description: training a Decision Tree model and classifying based on the langauge
    Language: Python3

    Use the following commands to train the model:
    lab3.py train train.dat dt.model dt
    lab3.py train train.dat ada.model ada

    Use the following command to test the model
    lab3.py predict dt.model unlabeledTest.dat
    lab3.py predict ada.model unlabeledTest.dat
    lab3.py predict best.model unlabeledTest.dat
"""
import pickle
import re
import sys
from AdaBoost import AdaBoost
import numpy
import pandas


def getTrainingDataDT(dataFile):
    """
        Creates a Pandas dataframe for the features of the data the model will be trained with (in .dat file)

        :param dataFile: the file which contains the data

        :return: a Pandas dataframe
    """
    features = ["nlArticles", "enConjunctions",
                "nlConjunctions", "nlPronouns",
                "is"]
    data = []
    with open(dataFile, encoding="utf-8") as file:
        # checking the features in each line
        for line in file:
            featureTracker = []
            featureTracker.append(line[:2])
            featureTracker.append(checkNlArticles(line[3:]))
            featureTracker.append(checkEnConjunctions(line[3:]))
            featureTracker.append(checkNlConjunctions(line[3:]))
            featureTracker.append(checkNlPronouns(line[3:]))
            featureTracker.append(checkIs(line[3:]))
            data.append(featureTracker)

    # converting to a dataframe and renaming the columns to features
    dataframe = pandas.DataFrame(data).rename(columns={0:"label",
                                                       1:features[0],
                                                       2:features[1],
                                                       3:features[2],
                                                       4:features[3],
                                                       5:features[4],})
    return dataframe

def getTrainingDataADA(dataFile):
    """
        Creates a Pandas dataframe for the features of the data the model will be trained with (in .dat file)

        :param dataFile: the file which contains the data

        :return: a Pandas dataframe
    """
    features = ["enArticles", "nlArticles", "enAuxiliary",
                "enConjunctions", "nlPrepositions",
                "enPronouns", "nlPronouns", "enQuestions", "is"]
    data = []
    with open(dataFile, encoding="utf-8") as file:
        # checking the features in each line
        for line in file:
            featureTracker = []
            featureTracker.append(line[:2])
            featureTracker.append(checkEnArticles(line[3:]))
            featureTracker.append(checkNlArticles(line[3:]))
            featureTracker.append(checkEnAuxiliary(line[3:]))
            featureTracker.append(checkEnConjunctions(line[3:]))
            featureTracker.append(checkNlPrepositions(line[3:]))
            featureTracker.append(checkEnPronouns(line[3:]))
            featureTracker.append(checkNlPronouns(line[3:]))
            featureTracker.append(checkEnQuestions(line[3:]))
            featureTracker.append(checkIs(line[3:]))
            data.append(featureTracker)

    # converting to a dataframe and renaming the columns to features
    dataframe = pandas.DataFrame(data).rename(columns={0: "label",
                                                       1: features[0],
                                                       2: features[1],
                                                       3: features[2],
                                                       4: features[3],
                                                       5: features[4],
                                                       6: features[5],
                                                       7: features[6],
                                                       8: features[7],
                                                       9: features[8], })
    return dataframe

def checkEnAuxiliary(line):
    auxiliary = ["be", "am", "are", "have", "has", "do", "does", "did",
                 "get", "got", "was", "were"]
    for word in re.findall(r"[\w']+", line):
        if word.strip().lower() in auxiliary:
            return True
    return False

def checkNlPrepositions(line):
    prepositions = ["met", "van", "naar", "voo", "achter", "naast",
                    "beneden", "boven", "onder", "op", "tussen", "midden",
                    "bij", "binnen", "buiten", "tegen", "rond", "sinds",
                    "zonder", "voor", "na", "om"]
    for word in re.findall(r"[\w']+", line):
        if word.strip().lower() in prepositions:
            return True
    return False

def checkEnQuestions(line):
    questions = ['who', 'what', 'when', 'where', 'why', 'how', 'which', 'whom', 'whose']
    for word in re.findall(r"[\w']+", line):
        if word.strip().lower() in questions:
            return True
    return False

def checkEnArticles(line):
    for word in re.findall(r"[\w']+", line):
        if word.strip().lower() == 'a' or \
            word.strip().lower() == 'an' or \
            word.strip().lower() == 'the':
            return True
    return False

def checkNlArticles(line):
    for word in re.findall(r"[\w']+", line):
        if word.strip().lower() == 'de' or \
            word.strip().lower() == 'het' or \
            word.strip().lower() == 'een':
            return True
    return False

def checkEnConjunctions(line):
    conjunctions = ["and", "but", "for", "yet", "neither", "or",
                    "so", "when", "although", "however", "as",
                    "because", "before"]
    for word in re.findall(r"[\w']+", line):
        if word.strip().lower() in conjunctions:
            return True
    return False

def checkNlConjunctions(line):
    for word in re.findall(r"[\w']+", line):
        if word.strip().lower() == 'en' or \
                word.strip().lower() == 'dus' or \
                word.strip().lower() == 'maar':
            return True
    return False

def checkEnPronouns(line):
    pronouns = ["she", "they", "he", "it", "him", "her",
                "you", "me", "anybody", "somebody", "someone", "anyone"]
    for word in re.findall(r"[\w']+", line):
        if word.strip().lower() in pronouns:
            return True
    return False

def checkNlPronouns(line):
    pronouns = ["hij", "zij", "het", "wij", "ik", "jij", "zij",
                "hem", "haar", "ons", "hen", "hun"]
    for word in re.findall(r"[\w']+", line):
        if word.strip().lower() in pronouns:
            return True
    return False

def checkIs(line):
    for word in re.findall(r"[\w']+", line):
        if word.strip().lower() == 'is':
            return True
    return False


def makeDecisionTree(trainingDataframe, maxDepth, checkedColumns = [], i=0):
    """
        Creates and trains a decision tree model based on training data

        :param trainingDataframe: features of the data
        :param maxDepth: maximum depth allowed
        :param checkedColumns: features already looked at
        :param i: iteration to keep track of the tree depth

        :return: a level in the decision tree
    """
    if trainingDataframe.empty:
        return
    # if classification can be made already
    if checkClassifiability(trainingDataframe, checkedColumns) or i == maxDepth:
        return classify(trainingDataframe)
    else:
        # generating the next level of the tree
        i += 1
        columnToCheck, infoGain, checkedColumns = getColumnToCheck(trainingDataframe, checkedColumns)
        yesDecisionDataframe, noDecisionDataframe = decide(trainingDataframe, columnToCheck)

        check = columnToCheck + "? > InfoGain: " + str(infoGain)
        nextLevel = {check: []}
        # if there is some info-gain on making a decision, go deeper
        if infoGain != 0:
            yesDecisionCheckedColumns = checkedColumns.copy()
            noDecisionCheckedColumns = checkedColumns.copy()
            # generate the tree for both decisions
            yesDecisionTree = makeDecisionTree(yesDecisionDataframe, maxDepth, yesDecisionCheckedColumns, i)
            noDecisionTree = makeDecisionTree(noDecisionDataframe, maxDepth, noDecisionCheckedColumns, i)
            # if both decisions output the same tree
            if yesDecisionTree == noDecisionTree:
                nextLevel = yesDecisionTree
            elif yesDecisionTree == None and noDecisionTree == None:
                return nextLevel
            else:
                nextLevel[check].append(yesDecisionTree)
                nextLevel[check].append(noDecisionTree)
            return nextLevel
        # if there is no info-gain on making a decision, start classification
        else:
            yesDecision = classify(yesDecisionDataframe)
            noDecision = classify(noDecisionDataframe)
            # if both decisions output the same tree-level
            if yesDecision == noDecision:
                nextLevel = yesDecision
            # if both decisions output different tree-levels
            else:
                if yesDecision != None:
                    nextLevel[check].append(yesDecision)
                else:
                    nextLevel[check].append(noDecision)
                if noDecision != None:
                    nextLevel[check].append(noDecision)
                else:
                    nextLevel[check].append(yesDecision)
            return nextLevel


def checkClassifiability(trainingDataframe, checkedColumns):
    """
        Checks if a classification can be made based on the feature

        :param trainingDataframe: features of the data
        :param checkedColumns: features already looked at

        :return: a boolean denoting if a classification can be made
    """
    labels = numpy.unique(trainingDataframe['label'].values)
    if len(labels) == 1 or len(checkedColumns) == trainingDataframe.shape[1]:
        return True
    return False


def classify(trainingDataframe):
    """
        Classifies the training data

        :param trainingDataframe: features of the data

        :return: count of the labels that exist in the data
    """
    if trainingDataframe.empty:
        return None
    labels, count = numpy.unique(trainingDataframe['label'].values,
                                 return_counts=True)
    return labels[count.argmax()]


def getColumnToCheck(trainingDataframe, checkedColumns):
    """
        Chooses which feature to look at

        :param trainingDataframe: features of the data
        :param checkedColumns: features already looked at

        :return: a tuple of - feature to look at, info-gain for the feature, features already checked
    """
    columnsList = list(trainingDataframe.columns)
    infoGain = {}
    for column in columnsList :
        if column == 'label' or column in checkedColumns:
            continue
        else:
            yesDecisionDataframe, noDecisionDataframe = decide(trainingDataframe, column)
            infoGain[column] = calculateInfoGain(trainingDataframe, yesDecisionDataframe, noDecisionDataframe)

    columnToCheck = max(infoGain, key=infoGain.get)
    checkedColumns.append(columnToCheck)
    return columnToCheck, infoGain[columnToCheck], checkedColumns


def decide(trainingDataframe, column):
    """
        Makes a classification decision for a feature in the dataframe

        :param trainingDataframe: features of the data
        :param column: feature being looked at

        :return: dataframes for 'Yes' and 'No' decisions
    """
    yesDecisionDataframe = trainingDataframe.loc[trainingDataframe[column] == True]
    noDecisionDataframe = trainingDataframe.loc[trainingDataframe[column] == False]
    yesDecisionDataframe.name = "case: " + str(column) + " ? Yes"
    noDecisionDataframe.name = "case: " + str(column) + " ? No"
    return yesDecisionDataframe, noDecisionDataframe


def calculateInfoGain(trainingDataframe, yesDecisionDataframe, noDecisionDataframe):
    """
        Calculates the info-gain from a decision

        :param trainingDataframe: features of the data
        :param yesDecisionDataframe: features of the data if the model makes a 'Yes' decision
        :param yesDecisionDataframe: features of the data if the model makes a 'No' decision

        :return: a numerical value equal to the info-gain
    """
    yesDecisionEntropy = calculateEntropy(yesDecisionDataframe)
    noDecisionEntropy = calculateEntropy(noDecisionDataframe)
    remaining = ((yesDecisionDataframe.shape[0] / trainingDataframe.shape[0]) * yesDecisionEntropy) + \
                ((noDecisionDataframe.shape[0] / trainingDataframe.shape[0]) * noDecisionEntropy)
    return calculateEntropy(trainingDataframe) - remaining


def calculateEntropy(dataframe):
    """
        Calculates the entropy for the features dataframe

        :param dataframe: features of the data

        :return: a numerical value equal to the entropy
    """
    count = dataframe['label'].value_counts()
    countSum = count.sum()
    labelProbability = count / countSum
    entropy = sum(labelProbability * numpy.log2(1 / labelProbability))
    return entropy


def makeAdaBoost(trainingDataframe, noOfStumps):
    """
        Creates an AdaBoost object for the features dataframe of the data

        :param trainingDataframe: features of the training data
        :param noOfStumps: number of hypothesis

        :return: an AdaBoost object
    """
    pandas.options.mode.chained_assignment = None
    trainLabelCol = trainingDataframe.loc[:, trainingDataframe.columns == 'label']
    trainOtherCol = trainingDataframe.loc[:, trainingDataframe.columns != 'label']
    trainLabelCol[trainLabelCol == 'en'] = 1
    trainLabelCol[trainLabelCol == 'nl'] = -1

    ADA = AdaBoost(noOfStumps)
    ADA.train(trainOtherCol, trainLabelCol)
    return ADA


def getTestingDataDT(dataFile):
    """
        Creates a Pandas dataframe for the features of the data to be tested (in .dat file)

        :param dataFile: the file which contains the data

        :return: a Pandas dataframe
    """
    features = ["nlArticles", "enConjunctions", "nlConjunctions"]
    data = []
    with open(dataFile) as file:
        # checking the features in each line
        for line in file:
            featureTracker = []
            featureTracker.append(checkNlArticles(line[3:]))
            featureTracker.append(checkEnConjunctions(line[3:]))
            featureTracker.append(checkNlConjunctions(line[3:]))
            data.append(featureTracker)

    # converting to a dataframe and renaming the columns to features
    dataframe = pandas.DataFrame(data).rename(columns={0: features[0],
                                                       1: features[1],
                                                       2: features[2],
                                                       3: "label"})
    return dataframe


def getTestingDataADA(dataFile):
    """
        Creates a Pandas dataframe for the features of the data to be tested (in .dat file)

        :param dataFile: the file which contains the data

        :return: a Pandas dataframe
    """
    features = ["enArticles", "nlArticles", "enAuxiliary",
                "enConjunctions", "nlPrepositions",
                "enPronouns", "nlPronouns", "enQuestions", "is"]
    data = []
    with open(dataFile) as file:
        # checking the features in each line
        for line in file:
            featureTracker = []
            featureTracker.append(checkEnArticles(line[3:]))
            featureTracker.append(checkNlArticles(line[3:]))
            featureTracker.append(checkEnAuxiliary(line[3:]))
            featureTracker.append(checkEnConjunctions(line[3:]))
            featureTracker.append(checkNlPrepositions(line[3:]))
            featureTracker.append(checkEnPronouns(line[3:]))
            featureTracker.append(checkNlPronouns(line[3:]))
            featureTracker.append(checkEnQuestions(line[3:]))
            featureTracker.append(checkIs(line[3:]))
            data.append(featureTracker)

    # converting to a dataframe and renaming the columns to features
    dataframe = pandas.DataFrame(data).rename(columns={0: features[0],
                                                       1: features[1],
                                                       2: features[2],
                                                       3: features[3],
                                                       4: features[4],
                                                       5: features[5],
                                                       6: features[6],
                                                       7: features[7],
                                                       8: features[8], })
    return dataframe



def classifyTestingData(testingDataframe, DT):
    """
        Classifies the testing data based on the unboosted decision tree model

        :param DT: the decision tree model
        :param testingDataframe: a pandas dataframe of testing data

        :return: an array with classification into en or nl language
    """
    keys = list(DT.keys())[0]
    yesDecision = DT[keys][0]
    noDecision = DT[keys][1]
    columnToCheck = keys[:keys.index('>') - 2]

    if testingDataframe[columnToCheck]:
        decision = yesDecision
    else:
        decision = noDecision

    if not isinstance(decision, dict):
        return decision
    else:
        return classifyTestingData(testingDataframe, decision)


def makePredictionADA(ADA, testingDataframe):
    """
        Takes in an ADA object and classifies the testing data

        :param ADA: the ada object
        :param testingDataframe: a pandas dataframe of testing data

        :return: None, prints the language predicted
    """
    predictions = ADA.predict(testingDataframe)
    for pred in predictions:
        if pred == 1:
            print("en")
        else:
            print("nl")


if __name__ == '__main__':
    pandas.options.mode.chained_assignment = None
    # if we want to train and create a model
    if sys.argv[1].lower() == "train":
        dataFile = sys.argv[2]
        hypothesisOut = sys.argv[3]
        algorithmType = sys.argv[4]

        if algorithmType == "dt":
            # If we want to train a decision tree model
            maxDepth = 2
            # processing the training data
            trainingDataframe = getTrainingDataDT(dataFile)
            # creating the model
            DT = makeDecisionTree(trainingDataframe, maxDepth)
            # pprint(DT)
            # saving the model
            hypothesis = open(hypothesisOut, 'wb')
            pickle.dump(DT, hypothesis)
            hypothesis.close()
            print("Hypothesis saved to " + hypothesisOut)
        elif algorithmType == "ada":
            # If we want to train a AdaBoost-ed model
            noOfStumps = 4
            # processing the training data
            trainingDataframe = getTrainingDataADA(dataFile)
            # creating the model
            ADA = makeAdaBoost(trainingDataframe, noOfStumps)
            # saving the model
            hypothesis = open(hypothesisOut, 'wb')
            pickle.dump(ADA, hypothesis)
            hypothesis.close()
            print("Hypothesis saved to " + hypothesisOut)

    # if we want to predict using a saved model
    if sys.argv[1].lower() == "predict":
        hypothesisIn = sys.argv[2]
        dataFile = sys.argv[3]
        hypothesis = open(hypothesisIn, 'rb')
        if hypothesisIn.lower() == "dt.model":
            # making predictions using the decision tree model
            testingDataframe = getTestingDataDT(dataFile)
            DT = pickle.load(hypothesis)
            for i in range(testingDataframe.shape[0]):
                print(classifyTestingData(testingDataframe.iloc[i], DT))

        if hypothesisIn.lower() == "ada.model" or hypothesisIn.lower() == "best.model":
            # making predictions using the AdaBoost-ed model
            testingDataframe = getTestingDataADA(dataFile)
            ADA = pickle.load(hypothesis)
            makePredictionADA(ADA, testingDataframe)

