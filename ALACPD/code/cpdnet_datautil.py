import numpy as np

# Logging
import logging

class DataUtil(object):
    #
    # This class contains data specific information.
    # It does the following:
    #  - Read data from file
    #  - Normalise it
    #  - Split it into train, dev (validation) and test
    #  - Create X and Y for each of the 3 sets (train, dev, test) according to the following:
    #    Every sample (x, y) shall be created as follows:
    #     - x --> window number of values
    #     - y --> one value that is at horizon in the future i.e. that is horizon away past the last value of x
    #    This way X and Y will have the following dimensions:
    #     - X [number of samples, window, number of multivariate time series]
    #     - Y [number of samples, number of multivariate time series]
    
    def __init__(self, filename, train, valid, horizon, window, normalise = 2):
        try:
            fin = open(filename)

            print("Start reading data")
            self.rawdata   = np.loadtxt(fin, delimiter=',')
            print("End reading data")

            self.w         = window
            self.h         = horizon
            self.data      = np.zeros(self.rawdata.shape)
            self.n, self.m = self.data.shape
            self.normalise = normalise
            self.scale     = np.ones(self.m)
        
            self.normalise_data(normalise)
            self.split_data(train, valid)
        except IOError as err:
            # In case file is not found, all of the above attributes will not have been created
            # Hence, in order to check if this call was successful, you can call hasattr on this object 
            # to check if it has attribute 'data' for example
            print("Error opening data file ... %s", err)
        
        
    def normalise_data(self, normalise):
        print("Normalise: %d", normalise)

        if normalise == 0: # do not normalise
            self.data = self.rawdata
        
        if normalise == 1: # same normalisation for all timeseries
            self.data = self.rawdata / np.max(self.rawdata)
        
        if normalise == 2: # normalise each timeseries alone. This is the default mode
            for i in range(self.m):
                self.scale[i] = np.max(np.abs(self.rawdata[:, i]))
                self.data[:, i] = self.rawdata[:, i] / self.scale[i]
    
    def split_data(self, train, valid):
        print("Splitting data into training set (%.2f), validation set (%.2f) and testing set (%.2f)", train, valid, 1 - (train + valid))
        train_set = range(self.w*2 + self.h - 1, int(train * self.n))
        valid_set = range(int(train * self.n), int((train + valid) * self.n))
        test_set  = range(int((train + valid) * self.n), self.n)
        
        self.train = self.get_data(train_set)
        self.valid = self.train 
        self.test  = self.get_data(test_set)
        
    def get_data(self, rng):
        n = len(rng)
        
        X1 = np.zeros((n, 1, self.w, self.m))
        X2 = np.zeros((n, self.w, self.w, self.m))
        Y = np.zeros((n, self.w, self.m))
  
        
        for i in range(n):
            end1   = rng[i] + 1
            start1 = end1 - self.w
            X1[i,0, :,:] = self.data[start1:end1, :]
            for j in range(self.w):
                end2 = rng[i] - self.h - j + 1
                start2 = end2 - self.w
                X2[i,j,:,:] = self.data[start2:end2, :]
            Y[i,:,:]   = self.data[start1:end1, :]  
        import numpy
        X = numpy.concatenate((X1,X2), axis = 1)
        return [X, Y]
