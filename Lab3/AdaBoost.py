import numpy
from Stump import Stump


class AdaBoost:
    """
        A class that represents an object that performs Adaboost learning
    """
    def __init__(self, noOfHypothesis = 5):
        self.hypotheses = None
        self.hypothesisCount = noOfHypothesis

    def train(self, dataframeOne, dataframeTwo):
        noOfExamples = dataframeOne.shape[0]
        fillVal = 1 / noOfExamples
        weights = numpy.full(noOfExamples, fillVal)
        dataframeTwo = dataframeTwo.to_numpy().flatten().astype(float)

        self.hypotheses = []
        for i in range(self.hypothesisCount):
            currentHypothesis = Stump()
            lowestError = float('inf')
            for column in dataframeOne.columns:
                feature = dataframeOne[column]
                for boolean in [True, False]:
                    predictions = numpy.ones(noOfExamples)
                    predictions[feature != boolean] = -1
                    error = sum(weights[dataframeTwo != predictions])
                    if error > 0.5:
                        continue
                    if error < lowestError:
                        lowestError = error
                        currentHypothesis.feature = column
                        currentHypothesis.check = boolean
            currentHypothesis.weight = 0.5 * numpy.log((1-lowestError) / lowestError)
            predictions = currentHypothesis.predict(dataframeOne)
            weights *= numpy.exp(predictions * dataframeTwo * (-currentHypothesis.weight))
            weights /= numpy.sum(weights)
            self.hypotheses.append(currentHypothesis)

    def predict(self, dataframe):
        predictions = [currentHypothesis.weight * currentHypothesis.predict(dataframe)
                                 for currentHypothesis in self.hypotheses]
        currentPrediction = numpy.sum(predictions, axis=0)
        return numpy.sign(currentPrediction)