import numpy

class Stump:
    """
        A class that represents a stump object
    """
    def __init__(self):
        self.feature = None
        self.weight = None
        self.check = None

    def predict(self, dataframe):
        dataframeFeature = dataframe[self.feature]
        predictions = numpy.ones(dataframe.shape[0])
        predictions[dataframeFeature != self.check] = -1
        return predictions
