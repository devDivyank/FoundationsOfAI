"""
Author: 'zjb'

An implementation of a heap data structure, taken from CSCI-603 course.
"""

# Fixing minor typo in pop method when the heap contains only one item

class Heap(object):
    """
    Heap that orders by a given comparison function, default to less-than.
    """
    __slots__ = ('data', 'size', 'lessfn')

    def __init__(self, lessfn):
        """
        Constructor takes a comparison function.
        :param lessfn: Function that takes in two heap objects and returns true
        if the first arg goes higher in the heap than the second
        """
        self.data = []
        self.size = 0
        self.lessfn = lessfn

    def __parent(self, loc):
        """
        Helper function to compute the parent location of an index
        :param loc: Index in the heap
        :return: Index of parent
        """
        return (loc - 1) // 2

    def __bubbleUp(self, loc):
        """
        Starts from the given location and moves the item at that spot
        as far up the heap as necessary
        :param loc: Place to start bubbling from
        """
        while loc > 0 and \
                self.lessfn(self.data[loc], self.data[self.__parent(loc)]):
            self.data[loc], self.data[self.__parent(loc)] = \
                self.data[self.__parent(loc)], self.data[loc]
            loc = self.__parent(loc)

    def __bubbleDown(self, loc):
        """
        Starts from the given location and moves the item at that spot
        as far down the heap as necessary
        :param loc: Place to start bubbling from
        """
        swapLoc = self.__smallest(loc)
        while swapLoc != loc:
            self.data[loc], self.data[swapLoc] = \
                self.data[swapLoc], self.data[loc]
            loc = swapLoc
            swapLoc = self.__smallest(loc)

    def __smallest(self, loc):
        """
        Finds the "smallest" value of loc and loc's two children.
        Correctly handles end-of-heap issues.
        :param loc: Index
        :return: index of the smallest value
        """
        ch1 = loc * 2 + 1
        ch2 = loc * 2 + 2
        if ch1 >= self.size:
            return loc
        if ch2 >= self.size:
            if self.lessfn(self.data[loc], self.data[ch1]):
                return loc
            else:
                return ch1
        # now consider all 3
        if self.lessfn(self.data[ch1], self.data[ch2]):
            if self.lessfn(self.data[loc], self.data[ch1]):
                return loc
            else:
                return ch1
        else:
            if self.lessfn(self.data[loc], self.data[ch2]):
                return loc
            else:
                return ch2

    def insert(self, item):
        """
        Inserts an item into the heap.
        :param item: Item to be inserted
        """
        if self.size < len(self.data):
            self.data[self.size] = item
        else:
            self.data.append(item)
        self.size += 1
        self.__bubbleUp(self.size - 1)

    def pop(self):
        """
        Removes and returns top of the heap
        :return: Item on top of the heap
        """
        assert self.size > 0, "Popping from an empty Heap"
        retjob = self.data[0]
        self.size -= 1
        # if we are popping the only element, bubbling is unnecessary, so:
        if self.size > 0:
            self.data[0] = self.data.pop(self.size)
            self.__bubbleDown(0)
        else:
            # we are popping the last element from the heap
            self.data.pop(0)
        return retjob

    def __len__(self):
        """
        Defining the "length" of a data structure also allows it to be
        used as a boolean value!
        :return: size of heap
        """
        return self.size

    def __str__(self):
        ret = ""
        for item in range(self.size):
            ret += str(self.data[item]) + " "
        return ret
