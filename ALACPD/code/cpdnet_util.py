import argparse
from utils import check_path 

class CPDNetInit(object):
    #
    # This class contains all initialisation information that are passed as arguments.
    #
    #    data:            Location of the data file
    #    window:          Number of time values to consider in each input X
    #                        Default: 24*7
    #    horizon:         How far is the predicted value Y. It is horizon values away from the last value of X (into the future)
    #                        Default: 12
    #    CNNFilters:      Number of output filters in the CNN layer
    #                        Default: 100
    #                        If set to 0, the CNN layer will be omitted
    #    CNNKernel:       CNN filter size that will be (CNNKernel, number of multivariate timeseries)
    #                        Default: 6
    #                        If set to 0, the CNN layer will be omitted
    #    GRUUnits:        Number of hidden states in the GRU layer
    #                        Default: 100
    #    SkipGRUUnits:    Number of hidden states in the SkipGRU layer
    #                        Default: 5
    #    skip:            Number of timeseries to skip. 0 => do not add Skip GRU layer
    #                        Default: 24
    #                        If set to 0, the SkipGRU layer will be omitted
    #    dropout:         Dropout frequency
    #                        Default: 0.2
    #    normalise:       Type of normalisation:
    #                     - 0: No normalisation
    #                     - 1: Normalise all timeseries together
    #                     - 2: Normalise each timeseries alone
    #                        Default: 2
    #    batchsize:       Training batch size
    #                        Default: 128
    #    epochs:          Number of training epochs
    #                        Default: 100
    #    initialiser:     The weights initialiser to use.
    #                        Default: glorot_uniform
    #    trainpercent:    How much percentage of the given data to use for training.
    #                        Default: 0.6 (60%)
    #    validpercent:    How much percentage of the given data to use for validation.
    #                        Default: 0.2 (20%)
    #                     The remaining (1 - trainpercent -validpercent) shall be the amount of test data
    #    highway:         Number of timeseries values to consider for the linear layer (AR layer)
    #                        Default: 24
    #                        If set to 0, the AR layer will be omitted
    #    train:           Whether to train the model or not
    #                        Default: True
    #    validate:        Whether to validate the model against the validation data
    #                        Default: True
    #                     If set and train is set, validation will be done while training.
    #    evaltest:        Evaluate the model using testing data
    #                        Default: False
    #    save:            Location and Name of the file to save the model in as follows:
    #                           Model in "save.json"
    #                           Weights in "save.h5"
    #                        Default: None
    #                     This location is also used to save results and history in, as follows:
    #                           Results in "save.txt" if --saveresults is passed
    #                           History in "save_history.csv" if --savehistory is passed
    #    saveresults:     Save results as described in 'save' above.
    #                     This has no effect if --save is not set
    #                        Default: True
    #    savehistory:     Save training / validation history as described in 'save' above.
    #                     This has no effect if --save is not set
    #                        Default: False
    #    load:            Location and Name of the file to load a pretrained model from as follows:
    #                           Model in "load.json"
    #                           Weights in "load.h5"
    #                        Default: None
    #    loss:            The loss function to use for optimisation.
    #                        Default: mean_absolute_error
    #    lr:              Learning rate
    #                        Default: 0.001
    #    optimizer:       The optimiser to use
    #                        Default: Adam
    #    test:            Evaluate the model on the test data
    #                        Default: False
    #    tensorboard:     Set to the folder where to put tensorboard file
    #                        Default: None (no tensorboard callback)
    #    predict:         Predict timeseries using the trained model.
    #                     It takes one of the following values:
    #                     - trainingdata   => predict the training data only
    #                     - validationdata => predict the validation data only
    #                     - testingdata    => predict the testing data only
    #                     - all            => all of the above
    #                     - None           => none of the above
    #                        Default: None
    #    plot:            Generate plots
    #                        Default: False
    #    series_to_plot:  The number of the series that you wish to plot. The value must be less than the number of series available
    #                        Default: 0
    #    autocorrelation: The number of the random series that you wish to plot their autocorrelation. The value must be less or equal to the number of series available
    #                        Default: None
    #    save_plot:       Location and Name of the file to save the plotted images to as follows:
    #                           Autocorrelation in "save_plot_autocorrelation.png"
    #                           Training results in "save_plot_training.png"
    #                           Prediction in "save_plot_prediction.png"
    #                        Default: None
    #    log:             Whether to generate logging
    #                        Default: True
    #    debuglevel:      Logging debuglevel.
    #                     It takes one of the following values:
    #                     - 10 => DEBUG
    #                     - 20 => INFO
    #                     - 30 => WARNING
    #                     - 40 => ERROR
    #                     - 50 => CRITICAL
    #                        Default: 20
    #    logfilename:     Filename where logging will be written.
    #                        Default: log/lstnet
    #
    def __init__(self, args, args_is_dictionary = False):
        if args_is_dictionary is True:
            self.data            =     args["data"]
            self.window          =     args["window"]
            self.horizon         =     args["horizon"]
            self.CNNFilters      =     args["CNNFilters"]
            self.CNNKernel       =     args["CNNKernel"]
            self.GRUUnits        =     args["GRUUnits"]
            self.SkipGRUUnits    =     args["SkipGRUUnits"]
            self.skip            =     args["skip"]
            self.dropout         =     args["dropout"]
            self.normalise       =     args["normalize"]
            self.highway         =     args["highway"]
            self.batchsize       =     args["batchsize"]
            self.epochs          =     args["epochs"]
            self.initialiser     =     args["initializer"]
            self.trainpercent    =     args["trainpercent"]
            self.validpercent    =     args["validpercent"]
            self.highway         =     args["highway"]
            self.train           = not args["no_train"]
            self.validate        = not args["no_validation"]
            self.save            =     args["save"]
            self.saveresults     = not args["no_saveresults"]
            self.savehistory     =     args["savehistory"]
            self.load            =     args["load"]
            self.loss            =     args["loss"]
            self.lr              =     args["lr"]
            self.optimiser       =     args["optimizer"]
            self.evaltest        =     args["test"]
            self.tensorboard     =     args["tensorboard"]
            self.plot            =     args["plot"]
            self.predict         =     args["predict"]
            self.series_to_plot  =     args["series_to_plot"]
            self.autocorrelation =     args["autocorrelation"]
            self.save_plot       =     args["save_plot"]
            self.log             = not args["no_log"]
            self.debuglevel      =     args["debuglevel"]
            self.logfilename     =     args["logfilename"]
        else:
            self.data            =     args.data
            self.window          =     args.window
            self.horizon         =     args.horizon
            self.CNNFilters      =     args.CNNFilters
            self.CNNKernel       =     args.CNNKernel
            self.GRUUnits        =     args.GRUUnits
            self.SkipGRUUnits    =     args.SkipGRUUnits
            self.skip            =     args.skip
            self.dropout         =     args.dropout
            self.normalise       =     args.normalize
            self.highway         =     args.highway
            self.batchsize       =     args.batchsize
            self.epochs          =     args.epochs
            self.initialiser     =     args.initializer
            self.trainpercent    =     args.trainpercent
            self.validpercent    =     args.validpercent
            self.highway         =     args.highway
            self.train           = not args.no_train
            self.validate        = not args.no_validation
            self.save            =     args.save
            self.saveresults     = not args.no_saveresults
            self.savehistory     =     args.savehistory
            self.load            =     args.load
            self.loss            =     args.loss
            self.lr              =     args.lr
            self.optimiser       =     args.optimizer
            self.evaltest        =     args.test
            self.tensorboard     =     args.tensorboard
            self.plot            =     args.plot
            self.predict         =     args.predict
            self.series_to_plot  =     args.series_to_plot
            self.autocorrelation =     args.autocorrelation
            self.save_plot       =     args.save_plot
            self.log             = not args.no_log
            self.debuglevel      =     args.debuglevel
            self.logfilename     =     args.logfilename

    def dump(self):
        print("Data: %s", self.data)
        print("Window: %d", self.window)
        print("Horizon: %d", self.horizon)
        print("CNN Filters: %d", self.CNNFilters)
        print("CNN Kernel: %d", self.CNNKernel)
        print("GRU Units: %d", self.GRUUnits)
        print("Skip GRU Units: %d", self.SkipGRUUnits)
        print("Skip: %d", self.skip)
        print("Dropout: %f", self.dropout)
        print("Normalise: %d", self.normalise)
        print("Highway: %d", self.highway)
        print("Batch size: %d", self.batchsize)
        print("Epochs: %d", self.epochs)
        print("Learning rate: %s", str(self.lr))
        print("Initialiser: %s", self.initialiser)
        print("Optimiser: %s", self.optimiser)
        print("Loss function to use: %s", self.loss)
        print("Fraction of data to be used for training: %.2f", self.trainpercent)
        print("Fraction of data to be used for validation: %.2f", self.validpercent)
        print("Train model: %s", self.train)
        print("Validate model: %s", self.validate)
        print("Test model: %s", self.evaltest)
        print("Save model location: %s", self.save)
        print("Save Results: %s", self.saveresults)
        print("Save History: %s", self.savehistory)
        print("Load Model from: %s", self.load)
        print("TensorBoard: %s", self.tensorboard)
        print("Plot: %s", self.plot)
        print("Predict: %s", self.predict)
        print("Series to plot: %s", self.series_to_plot)
        print("Save plot: %s", self.save_plot)
        print("Create log: %s", self.log)
        print("Debug level: %d", self.debuglevel)
        print("Logfile: %s", self.logfilename)


def SetArguments(data, filename, window=24*7, horizon=12, CNNFilters=100, CNNKernel=6,
                 GRUUnits=100, SkipGRUUnits=5, skip=24, dropout=0.2, normalize=2,
                 highway=24, lr=0.001, batchsize=128, epochs=100, 
                 initializer="glorot_uniform", loss="mean_squared_error", 
                 optimizer="Adam", trainpercent=0.6, validpercent=0.2,
                 save=None, load=None, tensorboard=None, predict=None, 
                 series_to_plot='0', autocorrelation=None, save_plot=None,
                 no_train=False, no_validation=False, test=False,
                 no_saveresults=False, savehistory=False, plot=False, 
                 no_log=False, debuglevel=20, logfilename="log/cpdnet"):
    args = {}
    args["data"] = data
    args["window"] = window
    args["horizon"] = horizon
    args["CNNFilters"] = CNNFilters
    args["CNNKernel"] = CNNKernel
    args["GRUUnits"] = GRUUnits
    args["SkipGRUUnits"] = SkipGRUUnits 
    args["skip"] = skip
    args["dropout"] = dropout
    args["normalize"] = normalize
    args["highway"] = highway
    args["lr"] = lr
    args["batchsize"] = batchsize
    args["epochs"] = epochs
    args["initializer"] = initializer
    args["loss"] = loss
    args["optimizer"] = optimizer
    args["trainpercent"] = trainpercent
    args["validpercent"] = validpercent
    args["save"] = save
    args["load"] = load
    args["tensorboard"] = tensorboard
    args["predict"] = predict
    args["series_to_plot"] = series_to_plot
    args["autocorrelation"] = autocorrelation
    args["save_plot"] = save_plot
    args["no_train"] = no_train
    args["no_validation"] = no_validation
    args["test"] = test
    args["no_saveresults"] = no_saveresults
    args["savehistory"] = savehistory
    args["plot"] = plot
    args["no_log"] = no_log
    args["debuglevel"] = debuglevel
    check_path(filename + logfilename)
    args["logfilename"] = filename + logfilename
    return args

   
        