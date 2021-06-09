"""
main.py creates, trains, and tests machine learning models when provided with test & training data sets.

Authors/Contributors: Dr. Dimitrios Diochnos, Conner Flansburg

Github Repo:
"""

import sys
import math
import random
import traceback
import logging as log
import numpy as np
import pandas as pd
import pathlib as pth
import typing as typ
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from pprint import pprint
from formatting import printBanner, banner, printError, printWarn, printSuccess
from alive_progress import alive_bar, config_handler
from pyfiglet import Figlet
from sklearn.linear_model import LinearRegression

SEED: int = 498
HDR = '*' * 6
SUCCESS = u' \u2713\n'+'\033[0m'       # print the checkmark & reset text color
OVERWRITE = '\r' + '\033[32;1m' + HDR  # overwrite previous text & set the text color to green
NO_OVERWRITE = '\033[32;1m' + HDR      # NO_OVERWRITE colors lines green that don't use overwrite
SYSOUT = sys.stdout                    # SYSOUT set the standard out for the program to the console

# Set up the logger
log_path = pth.Path.cwd() / 'logs' / 'log.txt'
log.basicConfig(level=log.DEBUG, filename=str(log_path), format='%(levelname)s-%(message)s')

# set up the global config for the loading bars
config_handler.set_global(spinner='dots_reverse', bar='smooth', unknown='stars', title_length=0, length=20)

# set the seed for the random library
random.seed(SEED)


def main():

    # * Start Up * #
    title: str = Figlet(font='larry3d').renderText('Weather Data')
    SYSOUT.write(f'\033[34;1m{title}\033[00m')  # formatted start up message
    log.debug('Started successfully')

    # * Select Training & Testing Data * #
    data_in = pth.Path.cwd() / 'data' / 'kdfw_processed_data.csv'  # create a Path object

    # * Read in the Data * #
    SYSOUT.write(HDR + ' Reading in file...')
    SYSOUT.flush()
    df_in: pd.DataFrame = read_in(str(data_in))
    SYSOUT.write(OVERWRITE + ' File read in successfully! '.ljust(50, '-') + SUCCESS)
    SYSOUT.flush()
    log.debug('Data read in successfully')

    # * Preprocess Data * #
    SYSOUT.write(HDR + ' Scaling data...')
    SYSOUT.flush()
    df_in = scale_data(df_in)
    SYSOUT.write(OVERWRITE + ' Data Scaling finished! '.ljust(50, '-') + SUCCESS)
    SYSOUT.flush()
    log.debug('Data Scaling finished')

    # * Preform the Linear Regression * #
    regression_report: pd.DataFrame = run_regression(df_in)
    log.debug('Regression performed successfully')

    # * Run the Neural Network * #
    # run_network()
    print('')
    print(banner(' Regression Report '))
    # format the data frame
    regression_report = regression_report.round(decimals=3)
    # regression_report = regression_report.transpose()

    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None, ):
        print(regression_report)

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

    # * Convert the datetime/drop date col * #
    data['date'] = pd.to_datetime(data['date'])
    # ? converting the date doesn't seem to make it compatible
    del data['date']  # drop the data column

    # * Drop the OBS Cols (except sknt_max) * #
    # create s dataframe with every col whose name contains 'OBS'
    drop_df = data.filter(like='OBS')
    # remove the target from the list of cols to be deleted
    del drop_df['OBS_sknt_max']
    # get the col names from drop_df & drop them from the original frame
    data.drop(list(drop_df.columns), axis=1, inplace=True)

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
        limit = 3.0 * sd[c]  # get the limit for this col. any more & we replace it with SD
        # replace any value in this col greater than limit with the standard deviation
        data[c].where(np.abs(data[c] - mean[c]) < limit, mean[c], inplace=True)

    if type(data) != pd.DataFrame:
        printError(f'remove_outlier returned a {type(data)}')
        sys.exit(-1)

    return data


def reduce_data(train: pd.DataFrame, test: pd.DataFrame, bar) -> typ.Dict[str, pd.DataFrame]:

    # * Dimensionality Reduction * #
    bar.text('reducing data')

    # this dict will store the labels & reduced data for return
    rtn: typ.Dict[str, pd.DataFrame] = {'Train Label': train['OBS_sknt_max'],
                                        'Test Label': test['OBS_sknt_max']}

    model = PCA()  # create the model
    # fit & reduce the training data
    rtn['Train Data'] = pd.DataFrame(model.fit_transform(train.drop('OBS_sknt_max', axis=1)), index=train.index)
    # reduce the training data
    rtn['Test Data'] = pd.DataFrame(model.transform(test.drop('OBS_sknt_max', axis=1)), index=test.index)

    return rtn


def split_random(data_in: pd.DataFrame, bar) -> typ.Tuple[pd.DataFrame, pd.DataFrame]:
    # TODO: comment
    log.debug('Random data split started')

    num_rows: int = len(data_in.index)  # get the number of rows
    # get the number of instances to be added to the training data
    size_train: int = math.ceil(0.80 * num_rows)
    # get the number of instances to be added to the testing data
    size_test: int = math.ceil(0.20 * num_rows)

    # create the empty train dataframe
    train: pd.DataFrame = pd.DataFrame(columns=data_in.columns)
    # create the empty test dataframe
    test: pd.DataFrame = pd.DataFrame(columns=data_in.columns)

    # *** Divide the Data *** #
    bar.text('splitting data')
    # iterrows returns the row index (grab it), then the row as a dict (ignore)
    for index, __ in data_in.iterrows():                # loop over every row
        r = data_in.loc[index]                          # get the row using the index
        if len(train.axes[0]) >= size_train:            # if train is full
            test = test.append(r, ignore_index=True)    # add to test
        elif len(test.axes[0]) >= size_test:            # if test is full
            train = train.append(r, ignore_index=True)  # add to train
        else:                                           # if neither are full,
            choice = random.choice(['train', 'test'])
            if choice == 'train':                           # if train was chosen
                train = train.append(r, ignore_index=True)  # add to train
            else:                                           # otherwise add to test
                test = test.append(r, ignore_index=True)
        bar()

    if type(train) != pd.DataFrame:
        printError(f'split_random returned train as a {type(train)}')
        sys.exit(-1)

    if type(test) != pd.DataFrame:
        printError(f'split_random returned test as a {type(test)}')
        sys.exit(-1)

    log.debug('Random data split finished')
    return train, test


def split_fixed(data_in: pd.DataFrame, bar) -> typ.Tuple[pd.DataFrame, pd.DataFrame]:
    # TODO: comment
    log.debug('Fixed data split started')

    num_rows: int = len(data_in.index)  # get the number of rows
    # get the number of instances to be added to the training data
    size_train: int = math.ceil(0.80 * num_rows)

    # create the empty train dataframe
    train: pd.DataFrame = pd.DataFrame(columns=data_in.columns)
    # create the empty test dataframe
    test: pd.DataFrame = pd.DataFrame(columns=data_in.columns)

    # *** Divide the Data *** #
    bar.text('splitting data')
    # iterrows returns the row index (grab it), then the row as a dict (ignore)
    count: int = 0  # this lets us know when to switch from train to test
    for index, __ in data_in.iterrows():  # loop over every row
        r = data_in.loc[index]            # get the row by using the index
        if count < size_train:            # if we haven't filled the train set
            train = train.append(r)       # add to the training set
            count += 1                    # increment count
        else:
            test = test.append(r)         # otherwise add to test
        bar()

    log.debug('Fixed data split finished')
    return train, test
# ************************************************************** #


# *************************** REPORT *************************** #
def create_report(rand: typ.Dict[str, float], fixed: typ.Dict[str, float]) -> pd.DataFrame:

    # TODO: expand this to work with Neural Network after it's added

    SYSOUT.write(HDR + ' Generating Report...')
    SYSOUT.flush()

    cols: typ.List[str] = ['Regression (random)', 'Regression (fixed)']
    rws: typ.List[str] = ['Mean Absolute Error', 'Mean Squared Error', 'Mean Signed Error']

    # get the error values from the passed dictionaries
    results: np.array = np.array([[rand['absolute'], fixed['absolute']],
                                  [rand['squared'], fixed['squared']],
                                  [rand['signed'], fixed['signed']]], dtype=float)

    # create a dataframe of the result metrics
    df: pd.DataFrame = pd.DataFrame(results, columns=cols, index=rws, dtype=float)
    df.round(3)

    SYSOUT.write(OVERWRITE + ' Report Generated! '.ljust(50, '-') + SUCCESS)
    SYSOUT.flush()
    return df


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
        total += (predict - actual) ** 2  # add each sqr to total

    # divide the total sqr by the number of examples to get the average
    avr = round(total * (1 / num_instances), 3)
    return avr


def beaufort_scale(model, train, test):
    pass
# ************************************************************** #


# *************************** MODELS *************************** #
def run_regression(data_in: pd.DataFrame):
    """ run_regression will run a simple linear regression on the passed dataframe. """

    print(banner(' Starting Liner Regression '))
    data_in: pd.DataFrame = pd.DataFrame(data_in)
    b_total: int = len(data_in.index) + 2  # this is used by the progress bar

    def random_data() -> typ.Dict[str, float]:
        """Perform linear regression with randomly divided data"""
        model = LinearRegression()  # create the regression model
        bar_rand.text('model built')
        bar_rand()

        train: pd.DataFrame
        test: pd.DataFrame
        train, test = split_random(data_in, bar_rand)  # randomly split dataset
        reduced: typ.Dict[str, pd.DataFrame] = reduce_data(train, test, bar_rand)

        # +++++++++++++++++++ Model Fit Notes +++++++++++++++++++ #
        # model.fit() takes 2 parameters: X & Y (in that order)
        # X = the labels we suspect are correlated
        # In this case it will be everything but Y
        # ! At this point we can't drop the col because they have no names because of reduce_data
        # perform preprocessing
        X: pd.DataFrame = reduced['Train Data']
        # Y = the target label (OBS_sknt_max for windspeed)
        Y: pd.DataFrame = reduced['Train Label']
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

        bar_rand.text('training model')
        model.fit(X, Y)  # fit the model using X & Y (see above)
        bar_rand.text('model trained')
        bar_rand()

        # calculate the error scores
        results: typ.Dict = {
            'absolute': absolute_error(model, reduced['Test Data'], reduced['Test Label']),
            'squared': squared_error(model, reduced['Test Data'], reduced['Test Label']),
            'signed': signed_error(model, reduced['Test Data'], reduced['Test Label']),
        }

        return results

    def fixed_data() -> typ.Dict[str, float]:
        """Perform linear regression with a fixed data division"""
        model = LinearRegression()  # create the regression model
        bar_fxd.text('model built')
        bar_fxd()

        train, test = split_fixed(data_in, bar_fxd)  # split dataset
        reduced: typ.Dict[str, pd.DataFrame] = reduce_data(train, test, bar_fxd)

        # +++++++++++++++++++ Model Fit Notes +++++++++++++++++++ #
        # model.fit() takes 2 parameters: X & Y (in that order)
        # X = the labels we suspect are correlated
        # In this case it will be everything but Y
        # perform preprocessing
        X: pd.DataFrame = reduced['Train Data']
        # Y = the target label (OBS_sknt_max for windspeed)
        Y: pd.DataFrame = reduced['Train Label']
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

        bar_fxd.text('training model')
        model.fit(X, Y)  # fit the model using X & Y (see above)
        bar_fxd.text('model trained')
        bar_fxd()

        # calculate the error scores
        results = {
            'absolute': absolute_error(model, reduced['Test Data'], reduced['Test Label']),
            'squared': squared_error(model, reduced['Test Data'], reduced['Test Label']),
            'signed': signed_error(model, reduced['Test Data'], reduced['Test Label']),
        }

        return results

    with alive_bar(b_total, title='Regression (random split)') as bar_rand:
        log.debug('Regression - random_data() called')
        rand = random_data()   # run the regression with a random data split
        log.debug('Regression - random_data() completed')

    with alive_bar(b_total, title='Regression (fixed split)') as bar_fxd:
        log.debug('Regression - fixed_data() called')
        fixed = fixed_data()   # run the regression with a fixed data split
        log.debug('Regression - fixed_data() completed')

    print(banner(' Liner Regression Finished '))

    # pass the dicts with the error values to the report generator
    report: pd.DataFrame = create_report(rand, fixed)

    return report


def run_network():

    # TODO: create neural network model

    def random_data():
        # TODO: used random data split
        pass

    def fixed_data():
        # TODO: use fixed data split
        pass

    # TODO: Split the Data Randomly
    # TODO: Run the Network
    # create_report()

    # TODO: Split the Data by Location in File
    # TODO: Run the Network
    # create_report()

    pass
# ************************************************************** #


if __name__ == '__main__':
    log.debug('\nStarting...')
    main()
    log.debug('closing...\n')
