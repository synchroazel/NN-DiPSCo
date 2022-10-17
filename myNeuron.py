import operator

import numpy as np

''' a single node implementation '''


class myNode:

    def __init__(self, thres, op='>='):
        self.thres = thres
        ops = {'>': operator.gt,
               '<': operator.lt,
               '>=': operator.ge,
               '<=': operator.le,
               '==': operator.eq}

        # ops['>'] is the same as operator.gt()

        self.op = ops[op]

    def fire(self, x):
        s = np.sum(x)
        if self.op(s, self.thres):
            return 1
        else:
            return 0


# %% TEST

n = myNode(5)  # create a node with a threshold of 5

n.fire([2, 1])  # test if it fires with the given set of linked nodes - NO
n.fire([2, 3])  # test if it fires with the given set of linked nodes - YES
n.fire([6, 8])  # test if it fires with the given set of linked nodes - YES
