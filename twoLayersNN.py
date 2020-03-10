import numpy as np
class TwoLayersNN(object):
    """" TwoLayersNN classifier """

    def __init__(self, inputDim, hiddenDim, outputDim):
        self.params = dict()
        self.params['w1'] = None
        self.params['b1'] = None
        self.params['w2'] = None
        self.params['b2'] = None
        #########################################################################
        # TODO: 20 points                                                       #
        # - Generate a random NN weight matrix to use to compute loss.          #
        # - By using dictionary (self.params) to store value                    #
        #   with standard normal distribution and Standard deviation = 0.0001.  #
        #########################################################################
        deviation = 0.0001
        self.params['w1'] = deviation * np.random.randn(inputDim, hiddenDim)
        self.params['b1'] = np.zeros(hiddenDim)
        self.params['w2'] = deviation * np.random.randn(hiddenDim, outputDim)
        self.params['b2'] = np.zeros(outputDim)

        pass

        #########################################################################
        #                       END OF YOUR CODE                                #
        #########################################################################

    def calLoss(self, x, y, reg):
        """
        TwoLayersNN loss function
        D: Input dimension.
        C: Number of Classes.
        N: Number of example.

        Inputs:
        - x: A numpy array of shape (batchSize, D).
        - y: A numpy array of shape (N,) where value < C.
        - reg: (float) regularization strength.

        Returns a tuple of:
        - loss as single float.
        - gradient with respect to each parameter (w1, b1, w2, b2)
        """
        loss = 0.0
        grads = dict()
        grads['w1'] = None
        grads['b1'] = None
        grads['w2'] = None
        grads['b2'] = None
        #############################################################################
        # TODO: 40 points                                                           #
        # - Compute the NN loss and store to loss variable.                         #
        # - Compute gradient for each parameter and store to grads variable.        #
        # - Use Leaky RELU Activation at hidden and output neurons                  #
        # - Use Softmax loss
        # Note:                                                                     #
        # - Use L2 regularization                                                   #
        # Hint:                                                                     #
        # - Do forward pass and calculate loss value                                #
        # - Do backward pass and calculate derivatives for each weight and bias     #
        #############################################################################
        w1 = self.params['w1']
        b1 = self.params['b1']
        w2 = self.params['w2']
        b2 = self.params['b2']
        result0 = np.dot(x, w1) + b1
        result1 = np.maximum(0.03 * result0, result0)
        result2 = np.dot(result1, w2) + b2
        scores = np.maximum(0.03 * result2, result2)

        n = x.shape[0]
        diff_exp = np.exp(scores)
        sum_log_diff = np.sum(diff_exp, axis=1, keepdims=True)
        prob = diff_exp / sum_log_diff
        # we only need the score of the correct class to compute loss
        prob_y = prob[np.arange(n), y]
        loss_y = -np.log(prob_y)
        # get the average loss
        loss = np.sum(loss_y) / n
        loss = loss + 0.5 * np.sum(w1 ** 2) + 0.5 * np.sum(w2 ** 2)

        ds = prob
        ds[np.arange(n), y] += -1
        ds = ds / n

        grads['w2'] = np.dot(result1.T, ds)
        grads['w2'] += reg * w2
        grads['b2'] = np.sum(ds, axis=0) / n

        ds = np.maximum(0.03 * ds, ds)
        d1 = np.dot(ds, w2.T)
        grads['w1'] = np.dot(x.T, d1)
        grads['w1'] += reg * w1
        grads['b1'] = np.sum(d1, axis=0) / n
        pass

        #############################################################################
        #                          END OF YOUR CODE                                 #
        #############################################################################

        return loss, grads

    def train(self, x, y, lr=5e-3, reg=5e-3, iterations=100, batchSize=200, decay=0.95, verbose=False):
        """
        Train this linear classifier using stochastic gradient descent.
        D: Input dimension.
        C: Number of Classes.
        N: Number of example.

        Inputs:
        - x: training data of shape (N, D)
        - y: output data of shape (N, ) where value < C
        - lr: (float) learning rate for optimization.
        - reg: (float) regularization strength.
        - iter: (integer) total number of iterations.
        - batchSize: (integer) number of example in each batch running.
        - verbose: (boolean) Print log of loss and training accuracy.

        Outputs:
        A list containing the value of the loss function at each training iteration.
        """

        # Run stochastic gradient descent to optimize W.
        lossHistory = []
        for i in range(iterations):
            xBatch = None
            yBatch = None
            #########################################################################
            # TODO: 10 points                                                       #
            # - Sample batchSize from training data and save to xBatch and yBatch   #
            # - After sampling xBatch should have shape (batchSize, D)              #
            #                  yBatch (batchSize, )                                 #
            # - Use that sample for gradient decent optimization.                   #
            # - Update the weights using the gradient and the learning rate.        #
            #                                                                       #
            # Hint:                                                                 #
            # - Use np.random.choice                                                #
            #########################################################################
            samples = np.random.choice(x.shape[0],batchSize)
            xBatch = x[samples]
            yBatch = y[samples]
            loss, dw = self.calLoss(xBatch, yBatch, reg)
            self.params['w1'] -= lr * dw['w1']
            self.params['b1'] -= lr * dw['b1']
            self.params['w2'] -= lr * dw['w2']
            self.params['b2'] -= lr * dw['b2']
            lossHistory.append(loss)
            pass

            #########################################################################
            #                       END OF YOUR CODE                                #
            #########################################################################
            # Decay learning rate
            lr *= decay
            # Print loss for every 100 iterations
            if verbose and i % 100 == 0 and len(lossHistory) is not 0:
                print('Loop {0} loss {1}'.format(i, lossHistory[i]))

        return lossHistory

    def predict(self, x, ):
        """
        Predict the y output.

        Inputs:
        - x: training data of shape (N, D)

        Returns:
        - yPred: output data of shape (N, ) where value < C
        """
        yPred = np.zeros(x.shape[0])
        ###########################################################################
        # TODO: 10 points                                                         #
        # -  Store the predict output in yPred                                    #
        ###########################################################################
        result0 = np.dot(x, self.params['w1']) + self.params['b1']
        result1 = np.maximum(result0 * 0.03, result0)
        result2 = np.dot(result1, self.params['w2']) + self.params['b2']
        scores = np.maximum(result2 * 0.03, result2)
        yPred = np.argmax(scores,axis=1)
        pass

        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return yPred

    def calAccuracy(self, x, y):
        ###########################################################################
        # TODO: 10 points                                                         #
        # -  Calculate accuracy of the predict value and store to acc variable    #
        ###########################################################################
        ypred = self.predict(x)
        accuracy = np.mean(y == ypred) * 100
        pass

        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return accuracy