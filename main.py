"""
main.py creates, trains, and tests machine learning models when provided with test & training data sets.

Authors/Contributors: Dr. Dimitrios Diochnos, Conner Flansburg

Github Repo:
"""

import argparse
import logging as log
import math
import pathlib as pth
import random
import sys
import time
import typing as typ
import traceback
from alive_progress import alive_bar, config_handler
import numpy as np
import pandas as pd
from pyfiglet import Figlet
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, BayesianRidge, Ridge, Lasso
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle

from plotting import smooth_plot

from formatting import banner, printError, success, printWarn

# TODO: write doc for command line flags (with examples)


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Used for Experimenting with Different Models !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
def create_model():
    """
    create_model is used as a wrapper for creation of the model.
    This allows the model's type to be a parameter/constant.
    """
    if MODEL == 'linear':            # use linear regression
        model = LinearRegression()
    elif MODEL == 'ridge':           # use ridge regression
        model = Ridge()
    elif MODEL == 'bayesRidge':      # use bayesian ridge regression
        model = BayesianRidge()
    elif MODEL == 'lasso':           # use lasso regression
        model = Lasso()
    else:                            # if the MODEL param has been set incorrectly, print error & exit
        # * Print Debug Info * #
        printError(f'Invalid Model Selected: {MODEL} does not exist')
        # * End Program * #
        sys.exit(-1)  # recovery impossible, exit

    return model

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #


# ****************************************** Configuration of Progress Bar ****************************************** #
# the global config for the loading bars
config_handler.set_global(spinner='dots_reverse', bar='classic', unknown='stars',
                          title_length=0, length=20, enrich_print=False)
# ******************************************** Parsing Command Line Flags ******************************************** #
# these values will be used if none are passed
Buckets_Num: int = 100       # default value for the number of 'buckets' to used in poisoning splitting
seed: int = 368              # default value for random number generation
trn = 80                     # default value for the size of the training set
mdl = 'ridge'                # default model type to run must be linear, ridge, bayesRidge, or lasso
smth = 20                    # default value for the amount of smoothing to be done

argumentParser = argparse.ArgumentParser()  # create the argument parser

# PCA Reduction Flag
argumentParser.add_argument("-r", "--reduction",            # the command line flag
                            action='store_true',            # if the value is true, store it in dest
                            dest='rdc',                     # the value will be stored in rdc
                            required=False,                 # the flag is not required
                            default=False,                  # if the flag is not provided, than it will be set to false
                            help="informs the program if the input data should be reduced using PCA")
# Number of Buckets Flag
argumentParser.add_argument("-b", "--buckets",              # the command line flag
                            action='store',                 # if the value is true, store it in dest
                            dest='bNum',                    # the value will be stored in BUCKETS_NUM
                            required=False,                 # the flag is not required
                            default=Buckets_Num,            # if the flag is not provided, use Buckets_Num
                            type=int,                       # cast the input so that it's an integer
                            help=f"the number of buckets used in poisoning (defaults to {Buckets_Num})")
# Training Size Flag
argumentParser.add_argument("-t", "--train",                # the command line flag
                            action='store',                 # if the value is true, store it in dest
                            dest='trn',                     # the value will be stored in BUCKETS_NUM
                            required=False,                 # the flag is not required
                            default=trn,                    # if the flag is not provided, use Buckets_Num
                            type=int,                       # cast the input so that it's an integer
                            help=f"percentage of the input to be used as training data as an integer (i.e. 80 -> 80%)")
# Random Seed Flag
argumentParser.add_argument("-s", "--seed",                 # the command line flag
                            action='store',                 # if the value is true, store it in dest
                            dest='sd',                      # the value will be stored in BUCKETS_NUM
                            required=False,                 # the flag is not required
                            default=seed,                   # if the flag is not provided, use Buckets_Num
                            type=int,                       # cast the input so that it's an integer
                            help=f"the number of buckets used in poisoning (defaults to {seed})")
# Learning Model Type Flag
argumentParser.add_argument("-m", "--model",
                            required=False,
                            dest='md',
                            choices=['linear', 'ridge', 'bayesRidge', 'lasso'],
                            default=mdl,
                            type=str,
                            help=f"what learning model should be used/tested (defaults to {mdl})")
# Smoothening Flag
argumentParser.add_argument("-sm", "--smooth",             # the command line flag
                            action='store',                # if the value is true, store it in dest
                            dest='smooth',                 # the value will be stored in smooth
                            required=False,                # the flag is not required
                            default=smth,                  # if the flag is not provided, use smth
                            type=int,                      # cast the input so that it's an integer
                            help=f"the number of iterations used during smoothening (defaults to {smth})")
# ************************************************ Set up the logger ************************************************* #
log_path = pth.Path.cwd() / 'logs' / 'log.txt'
log.basicConfig(level=log.ERROR, filename=str(log_path), format='%(levelname)s-%(message)s')
# ******************************************************************************************************************** #


def main():

    # * Start Up * #
    title: str = Figlet(font='larry3d').renderText('REU 2021')
    SYSOUT.write(f'\033[34;1m{title}\033[00m')  # formatted start up message
    log.debug('Started successfully')

    # * Select Training & Testing Data * #
    data_in = pth.Path.cwd() / 'data' / 'kdfw_processed_data.csv'  # create a Path object

    # * Read in the Data * #
    SYSOUT.write(HDR + ' Reading in file...'); SYSOUT.flush()
    df_in: pd.DataFrame = read_in(str(data_in))
    SYSOUT.write(OVERWRITE + ' File read in successfully! '.ljust(44, '-') + SUCCESS); SYSOUT.flush()
    log.debug('Data read in successfully')

    # print(df_in.info())

    # * Preprocess Data * #
    SYSOUT.write(HDR + ' Scaling data...'); SYSOUT.flush()
    df_in = scale_data(df_in)
    SYSOUT.write(OVERWRITE + ' Data Scaling finished! '.ljust(44, '-') + SUCCESS); SYSOUT.flush()
    log.debug('Data Scaling finished')

    # * Run a Poisoning Attack on the Linear Regression * #
    report = poison_regression(df_in)

    # * Save Report * #
    sorted_cols = list(report['Absolute'].columns)
    sorted_cols.sort()
    report['Absolute'] = report['Absolute'].reindex(columns=sorted_cols)
    report['Absolute'].to_csv(str(pth.Path.cwd() / 'output' / f'absolute.csv'))

    sorted_cols = list(report['Signed'].columns)
    sorted_cols.sort()
    report['Signed'] = report['Signed'].reindex(columns=sorted_cols)
    report['Signed'].to_csv(str(pth.Path.cwd() / 'output' / f'signed.csv'))

    sorted_cols = list(report['Squared'].columns)
    sorted_cols.sort()
    report['Squared'] = report['Squared'].reindex(columns=sorted_cols)
    report['Squared'].to_csv(str(pth.Path.cwd() / 'output' / f'squared.csv'))

    # * Plot the Report * #
    smooth_plot(report['Absolute'], str(pth.Path.cwd() / 'output' / f'absoluteError.png'), 'Absolute Error')
    smooth_plot(report['Signed'], str(pth.Path.cwd() / 'output' / f'signedError.png'), 'Signed Error')
    smooth_plot(report['Squared'], str(pth.Path.cwd() / 'output' / f'squaredError.png'), 'Squared Error')

    log.debug('Program completed successfully')


# *********************** MANIPULATE DATA ********************** #
def read_in(fl: str) -> pd.DataFrame:
    """ read_in reads in the csv & transforms it into a Pandas Dataframe. """
    # * Read in the Data * # (treating '********' as NaNs)
    data: pd.DataFrame = pd.read_csv(fl, dtype=float, na_values='********',
                                     parse_dates=['date'], index_col=0).sort_values(by='date')

    # * Get rid of NaN & infinity * #
    data = data.replace('********', np.nan).replace(np.inf, np.nan)
    data = data.dropna(how='any', axis=1)

    # * Convert the Date to an Int * #
    # ? not having a date doesn't seem to reduce accuracy
    data["date"] = pd.to_datetime(data["date"]).dt.strftime("%Y%m%d")
    data['date'].astype(int)
    # del data['date']  # drop the data column

    # * Drop the OBS Cols (except sknt_max) * #
    # create s dataframe with every col whose name contains 'OBS'
    # drop_df = data.filter(like='OBS')
    # remove the target from the list of cols to be deleted
    # del drop_df['OBS_sknt_max']
    # get the col names from drop_df & drop them from the original frame
    # data.drop(list(drop_df.columns), axis=1, inplace=True)

    if type(data) != pd.DataFrame:
        printError(f'read_in returned a {type(data)}')
        sys.exit(-1)

    return data                   # Return the Dataframe


def scale_data(data: pd.DataFrame) -> pd.DataFrame:
    """ Scale the data using a MinMax Scalar. """

    # * Remove Outliers * #
    data: pd.DataFrame = remove_outlier(data)
    # * Scale/Normalize the data using a MinMax Scalar * #
    model = MinMaxScaler()        # create the scalar
    model.fit(data)               # fit the scalar
    scaled = model.transform(data)  # transform the data
    # cast the scaled data back into a dataframe
    data = pd.DataFrame(scaled, index=data.index, columns=data.columns)

    if type(data) != pd.DataFrame:
        printError(f'scale_data returned a {type(data)}')
        sys.exit(-1)

    return data


def remove_outlier(data: pd.DataFrame) -> pd.DataFrame:
    """
    Replace any value more standard deviations from
    the mean with the mean.

    see: https://colab.research.google.com/drive/1rRQF0bvKSTOR4ZzLEVhRi7Z3GFzzmMQ2#scrollTo=I7qkRpsbHRsf
    """
    sd: pd.DataFrame = data.std()
    mean: pd.DataFrame = data.mean()

    for c in data.columns:
        if c == 'date':
            pass  # don't try to remove outliers from data as that makes no sense
        else:
            limit = 3.0 * sd[c]  # get the limit for this col. any more & we replace it with SD
            # replace any value in this col greater than limit with the standard deviation
            data[c].where(np.abs(data[c] - mean[c]) < limit, mean[c], inplace=True)

    if type(data) != pd.DataFrame:
        printError(f'remove_outlier returned a {type(data)}')
        sys.exit(-1)

    return data


def reduce_data(train: pd.DataFrame, test: pd.DataFrame, bar) -> typ.Dict[str, pd.DataFrame]:

    # * Dimensionality Reduction * #
    bar.text('reducing data...')

    # this dict will store the labels & reduced data for return
    rtn: typ.Dict[str, pd.DataFrame] = {'Train Label': train['OBS_sknt_max'],
                                        'Test Label': test['OBS_sknt_max']}

    model = PCA(n_components=0.65, svd_solver='full')  # create the model
    # fit & reduce the training data
    rtn['Train Data'] = pd.DataFrame(model.fit_transform(train.drop('OBS_sknt_max', axis=1)), index=train.index)
    # reduce the test data
    rtn['Test Data'] = pd.DataFrame(model.transform(test.drop('OBS_sknt_max', axis=1)), index=test.index)

    print(success('reduction complete ---- '+u'\u2713'))
    return rtn
# ************************************************************** #


# *************************** REPORT *************************** #
def signed_error(model, data, label) -> float:
    """ Calculate the Mean Signed Error """
    prediction = model.predict(data)
    actual = label
    num_instances = len(actual)  # the number of examples in the test set

    if len(prediction) != len(actual):  # check that both are equal
        printError('Actual & Prediction are not equal!')
        sys.exit(-1)

    # combine the predict & act so we can loop in parallel
    # compare each prediction with the actual value
    total = 0  # this will hold the sum of each sqr
    # mean sqr = (predicted_value - actual_value)^2 * 1/num_examples
    for predict, actual in zip(prediction, actual):
        total += predict - actual  # add each sqr to total

    # divide the total sqr by the number of examples to get the average
    avr = round(total * (1 / num_instances), 3)
    return avr


def absolute_error(model, data, label) -> float:
    """ Calculate the Mean Absolute Error """
    prediction = model.predict(data)
    actual = label
    num_instances = len(actual)  # the number of examples in the test set

    if len(prediction) != len(actual):  # check that both are equal
        printError('Actual & Prediction are not equal!')
        sys.exit(-1)

    # combine the predict & act so we can loop in parallel
    # compare each prediction with the actual value
    total = 0  # this will hold the sum of each sqr
    # mean sqr = (predicted_value - actual_value)^2 * 1/num_examples
    for predict, actual in zip(prediction, actual):
        total += abs(predict - actual)  # add each sqr to total

    # divide the total sqr by the number of examples to get the average
    avr = round(total * (1 / num_instances), 3)

    return avr


def squared_error(model, data, label) -> float:
    """ Calculate the Mean Squared Error """
    model.predict(data)
    prediction = [round(i, 3) for i in model.predict(data)]
    actual = label
    num_instances = len(actual)  # the number of examples in the test set

    # TODO: look here for error
    # ? maybe the error is in the error rate calculation?
    # ! for debugging, print prediction to file
    # rst = list(zip(prediction, actual))
    # df = pd.DataFrame(rst, columns=['Prediction', 'Actual'])
    # df.to_csv(str(pth.Path.cwd() / 'logs' / 'predict.csv'))
    # print(f'Prediction Mean: {df["Prediction"]}')
    # print(f'Actual Mean: {df["Actual"]}')
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #

    if len(prediction) != len(actual):  # check that both are equal
        printError('Actual & Prediction are not equal!')
        sys.exit(-1)

    # combine the predict & act so we can loop in parallel
    # compare each prediction with the actual value
    total = 0  # this will hold the sum of each sqr
    # mean sqr = (predicted_value - actual_value)^2 * 1/num_examples
    for predict, actual in zip(prediction, actual):
        total += (predict - actual) ** 2  # add each sqr to total

    # divide the total sqr by the number of examples to get the average
    avr = round(total * (1 / num_instances), 3)
    return avr


def beaufort_scale(model, train, test):
    pass
# ************************************************************** #


# ************************* Poisoning ************************* #
def split_poison(data: pd.DataFrame, bar) -> typ.Tuple[typ.List[pd.DataFrame], pd.DataFrame]:
    """
    split_poison will split the provided Pandas dataframe into testing & training data
    using the first 80% of the entries as training, and the rest as testing.
    This should method should only be used during poisoning attacks

    :param data: dataframe to be split
    :param bar: loading bar used for console printing

    :return: dataframes for train & test (in that order)
    """

    log.debug('fixed data split started')
    percent_training = TRAIN_SIZE  # TODO: experiment with different sizes

    num_rows: int = len(data.index)  # get the number of rows
    # get the number of instances to be added to the training data
    size_train: int = math.ceil(percent_training * num_rows)

    # * Divide Data into Testing & Training Sets * #
    bar.text("splitting data...")
    # slice [inclusive : not_inclusive]
    # slice the dataframe [rows, cols] & grab everything up to size_train
    train: pd.DataFrame = data.iloc[:size_train + 1, :]
    # slice the dataframe [rows, cols] & grab everything after size_train
    testing: pd.DataFrame = data.iloc[size_train + 1:, :]

    # * Divide Training Data into N Buckets * #
    bar.text('creating buckets...')
    # break the training data into BUCKETS_NUM number of smaller dataframes
    # with about the same number of instance in each
    spam: typ.List[np.ndarray] = np.array_split(train.to_numpy(), BUCKETS_NUM)
    training: typ.List[pd.DataFrame] = [pd.DataFrame(a, columns=train.columns) for a in spam]

    log.debug('poison data split (random split) finished')
    return training, testing


def poison_reduction(train: pd.DataFrame, test: pd.DataFrame, bar) -> typ.Tuple[pd.DataFrame, pd.DataFrame]:

    # * Dimensionality Reduction * #
    bar.text('reducing data...')
    model = PCA(n_components=0.65, svd_solver='full')  # create the model
    # fit & reduce the training data
    reduced_train = pd.DataFrame(model.fit_transform(train.drop('OBS_sknt_max', axis=1)), index=train.index)
    # reduce the test data
    reduced_test = pd.DataFrame(model.transform(test.drop('OBS_sknt_max', axis=1)), index=test.index)

    log.debug('poison reduction finished')
    return reduced_train, reduced_test


def poison_regression(data_in: pd.DataFrame) -> typ.Dict[str, pd.DataFrame]:
    """
    poison_regression will run a simple linear regression on the passed dataframe
    & poison the training data.
    """
    print(banner(' Poisoning Liner Regression '))
    data_in: pd.DataFrame = pd.DataFrame(data_in)

    def poison_data(test: pd.DataFrame, train_list: typ.List[pd.DataFrame]) -> typ.Dict[str, float]:
        """Perform linear regression with poisoned data"""
        model = create_model()  # create the regression model

        bar.text('joining dataframes...')
        # create a single dataframe from the list of dataframes
        train: pd.DataFrame = pd.concat(train_list)
        # BUCKETS_LABEL is used to save the number of instances used for each iteration,
        # so add to it here.
        global BUCKETS_LABEL
        BUCKETS_LABEL.add(len(train.index))

        # * Reduce (if requested) * #
        train_labels: pd.DataFrame = train['OBS_sknt_max']
        test_labels: pd.DataFrame = test['OBS_sknt_max']
        if REDUCE:
            # if we are reducing call poison_reduction & save the labels
            train_features: pd.DataFrame
            test_features: pd.DataFrame
            train_features, test_features = poison_reduction(train, test, bar)
        else:
            # if we aren't reducing then just create variables with the same names
            train_features: pd.DataFrame
            test_features: pd.DataFrame
            train_features = train.drop('OBS_sknt_max', axis=1)
            test_features = test.drop('OBS_sknt_max', axis=1)

        # +++++++++++++++++++ Model Fit Notes +++++++++++++++++++ #
        # model.fit() takes 2 parameters: X & Y (in that order)
        # X = the labels we suspect are correlated
        # In this case it will be everything but Y
        X: pd.DataFrame = train_features
        # Y = the target label (OBS_sknt_max for wind speed)
        Y: pd.DataFrame = train_labels
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

        bar.text('training model...')
        model.fit(X, Y)  # fit the model using X & Y (see above)

        # * Calculate the Error Scores * #
        bar.text('getting error score...')

        frame = {
            'Absolute': absolute_error(model, test_features, test_labels),
            'Squared': squared_error(model, test_features, test_labels),
            'Signed': signed_error(model, test_features, test_labels)
        }

        return frame

    # this will be a list of lists of the errors scores
    # (each member will be a list of the error score for every bucket)
    report_list = {'Absolute': [], 'Squared': [], 'Signed': []}
    for sm in range(SMOOTH):

        # randomly shuffle the dataframe for each iteration of SMOOTH before splitting it
        data_in = shuffle(data_in)

        with alive_bar(BUCKETS_NUM, title=f'Regression {sm}/{SMOOTH}') as bar:
            bar.text('Starting...')

            # split the data into test & train buckets
            training: typ.List[pd.DataFrame]
            testing: pd.DataFrame
            training, testing = split_poison(data_in, bar)

            # Buckets is used to pass the training data that will actually be used by the model,
            buckets: typ.List[pd.DataFrame] = []

            # These are arrays that will hold the error rates for every bucket
            absolute = []
            squared = []
            signed = []

            # preprocessing is done, train the models
            while training:  # while we can still pop training (we haven't used all buckets)

                # add the last frame from the training set to the set of buckets
                buckets.append(training.pop())

                # !!!!!!!!!!!!!!!!!!!!!! for debugging !!!!!!!!!!!!!!!!!!!!!! #
                # printWarn(f'Training after pop has length of {len(training)}')
                # printWarn(f'Buckets after pop has length of {len(buckets)}')
                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #

                # train the regression using what's left of train_list, & add results to errs list
                current_errors = poison_data(testing, buckets)

                # add the error from this iteration to the lists of error rates
                absolute.append(current_errors['Absolute'])
                squared.append(current_errors['Squared'])
                signed.append(current_errors['Signed'])

                # inform user that this bucket size iteration is done
                # print(success(f'model {len(buckets)}/{BUCKETS_NUM} complete '.ljust(23, '-') + u' \u2713'))
                bar()  # update the loading bar

        # add the error values for each iteration of smooth to the report dictionary
        report_list['Absolute'].append(absolute)
        report_list['Squared'].append(squared)
        report_list['Signed'].append(signed)

    # use the list of lists to create a dataframe that has the training size as the cols & smooth iter as rows
    # this will create a dataframe of the error score for every iteration & training size.
    # get the average values later
    report: typ.Dict[str, pd.DataFrame] = {
        'Absolute': pd.DataFrame(report_list['Absolute'], columns=BUCKETS_LABEL, dtype=float).round(3),
        'Squared': pd.DataFrame(report_list['Squared'], columns=BUCKETS_LABEL, dtype=float).round(3),
        'Signed': pd.DataFrame(report_list['Signed'], columns=BUCKETS_LABEL, dtype=float).round(3)
    }

    print(banner(' Poisoning Finished '))
    return report


if __name__ == '__main__':
    log.debug('Starting...')

    # *** Set the Program's Parameters & Constants *** #
    # (we are not inside a function so the global keyword isn't needed) #
    usr = argumentParser.parse_args()  # grab the input flags

    BUCKETS_NUM: int = usr.bNum          # BUCKETS_NUM is the number of 'buckets' to used in poisoning splitting
    SEED: int = usr.sd                   # SEED is used as the seed value for random number generation
    REDUCE: bool = usr.rdc               # REDUCE is a bool that says if the data should be reduced using PCA
    TRAIN_SIZE: float = usr.trn * 0.01   # TRAIN is used to determine the percentage of the input used for the training set
    MODEL: str = usr.md                  # MODEL is the type of prediction/learning model to build
    SMOOTH: int = usr.smooth             # SMOOTH is the amount of iterations to be used during smoothening
    BUCKETS_LABEL: typ.Set[int] = set()  # BUCKETS_LABEL is used to label the x-axis of the error score plot

    # *** Seed the Random Libraries Using the Provided Seed Value *** #
    random.seed(SEED)
    np.random.seed(SEED)

    # *** Used for Printing *** #
    HDR = '*' * 6
    SUCCESS = u' \u2713\n' + '\033[0m'     # print the checkmark & reset text color
    OVERWRITE = '\r' + '\033[32;1m' + HDR  # overwrite previous text & set the text color to green
    NO_OVERWRITE = '\033[32;1m' + HDR      # NO_OVERWRITE colors lines green that don't use overwrite
    SYSOUT = sys.stdout                    # SYSOUT set the standard out for the program to the console

    start = time.perf_counter()  # start the timer
    main()                       # execute the program
    stop = time.perf_counter()   # end the timer
    print(f'Execution time: {stop - start:0.4f} Seconds')
    log.debug('closing...\n')


'''
Generic Exception Handler (because I kept needing to look it up):
    except Exception as err:
        # * Print Error Info * #
        printWarn(traceback.format_exc())
        
        # * Get Line Number * #
        lineNm = sys.exc_info()[-1].tb_lineno  # print line number error occurred on
        printError(f'Error encountered on line {lineNm}\nMessage: {repr(err)}')
        log.error(f'Error encountered on line {lineNm} \nMessage: {repr(err)}')
        
        # * Print Debug Info * #
        # printWarn(f'Some value: {that value}')

        # * End Program * #
        sys.exit(-1)  # recovery impossible, exit
'''
