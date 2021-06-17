"""
main.py creates, trains, and tests machine learning models when provided with test & training data sets.

Authors/Contributors: Dr. Dimitrios Diochnos, Conner Flansburg

Github Repo:
"""

import sys
import math
import random
import argparse
# import traceback
import logging as log
import traceback

import numpy as np
import pandas as pd
import pathlib as pth
import typing as typ
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
# from pprint import pprint
from formatting import banner, printError, success, printWarn
from pyfiglet import Figlet
from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle
from yaspin import yaspin


# ******************************************** Parsing Command Line Flags ******************************************** #
# these values will be used if none are passed
Buckets_Num: int = 10        # BUCKETS_NUM is the number of 'buckets' to used in poisoning splitting
seed: int = 368              # SEED is used as the seed value for random number generation

argumentParser = argparse.ArgumentParser()  # create the argument parser

# Run Standard Regression Flag
argumentParser.add_argument("-stdr", "--standard-regress",  # the command line flag
                            action='store_true',            # if the value is true, store it in dest
                            dest='regress',                 # the value will be stored in RUN_STND_RGRS
                            required=False,                 # the flag is not required
                            default=False,                  # if the flag is not provided, than it will be set to false
                            help="run the standard regression without poisoning"
                            )

# Number of Buckets Flag
argumentParser.add_argument("-b", "--buckets",              # the command line flag
                            action='store',                 # if the value is true, store it in dest
                            dest='bNum',                    # the value will be stored in BUCKETS_NUM
                            required=False,                 # the flag is not required
                            default=Buckets_Num,            # if the flag is not provided, use Buckets_Num
                            type=int,                       # cast the input so that it's an integer
                            help=f"the number of buckets used in poisoning (defaults to {Buckets_Num})"
                            )

# Random Seed Flag
argumentParser.add_argument("-s", "--seed",                 # the command line flag
                            action='store',                 # if the value is true, store it in dest
                            dest='sd',                      # the value will be stored in BUCKETS_NUM
                            required=False,                 # the flag is not required
                            default=seed,                   # if the flag is not provided, use Buckets_Num
                            type=int,                       # cast the input so that it's an integer
                            help=f"the number of buckets used in poisoning (defaults to {Buckets_Num})"
                            )
# ************************************************ Set up the logger ************************************************* #
log_path = pth.Path.cwd() / 'logs' / 'log.txt'
log.basicConfig(level=log.ERROR, filename=str(log_path), format='%(levelname)s-%(message)s')
# ******************************************************************************************************************** #


def main():

    # * Start Up * #
    title: str = Figlet(font='larry3d').renderText('Weather Data')
    SYSOUT.write(f'\033[34;1m{title}\033[00m')  # formatted start up message
    log.debug('Started successfully')

    # * Select Training & Testing Data * #
    data_in = pth.Path.cwd() / 'data' / 'kdfw_processed_data.csv'  # create a Path object

    # * Read in the Data * #
    SYSOUT.write(HDR + ' Reading in file...'); SYSOUT.flush()
    df_in: pd.DataFrame = read_in(str(data_in))
    SYSOUT.write(OVERWRITE + ' File read in successfully! '.ljust(44, '-') + SUCCESS); SYSOUT.flush()
    log.debug('Data read in successfully')

    # * Preprocess Data * #
    SYSOUT.write(HDR + ' Scaling data...'); SYSOUT.flush()
    df_in = scale_data(df_in)
    SYSOUT.write(OVERWRITE + ' Data Scaling finished! '.ljust(44, '-') + SUCCESS); SYSOUT.flush()
    log.debug('Data Scaling finished')

    # * Preform the Standard Linear Regression * #
    if RUN_STND_RGRS:  # if param RUN_STND_RGRS (run standard regression) is true, execute
        report: pd.DataFrame = run_regression(df_in)
        log.debug('Regression performed successfully')

        # * Display & Save Standard Linear Regression Report * #
        print(f"\n    {banner(' Regression Report ')}")
        report = report.round(decimals=3)                           # format the data frame
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.precision', 3):
            print(report)                                           # display results
        rOut = pth.Path.cwd() / 'output' / 'regression_report.csv'  # create file path
        report.to_csv(str(rOut))                                    # save the results to a file
        print('')                                                   # print newline after the report

    # * Run a Poisoning Attack on the Linear Regression * #
    report = poison_regression(df_in)

    # * Display & Save Report * #
    print(f"\n    {banner(' Poisoning Report ')}")
    report = report.round(decimals=3)                          # format the data frame
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.precision', 3):
        print(report)                                          # display results
    rOut = pth.Path.cwd() / 'output' / 'poisoning_report.csv'  # create file path
    report.to_csv(str(rOut))                                   # save the results to a file
    print('')                                                  # print newline after the report

    # create scatter plot of the mean squared error rate vs number of buckets
    # scatter_plot(report, x_axis='Bucket(s)', y_axis='Mean Squared Error', title='Poisoned Regression', file=str(pth.Path.cwd() / 'output' / 'poison_scatter.png'))

    error_plot(report, str(pth.Path.cwd() / 'output' / 'poison_line.png'))

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


def reduce_data(train: pd.DataFrame, test: pd.DataFrame, spnr) -> typ.Dict[str, pd.DataFrame]:

    # * Dimensionality Reduction * #
    spnr.text = 'reducing data...'

    # this dict will store the labels & reduced data for return
    rtn: typ.Dict[str, pd.DataFrame] = {'Train Label': train['OBS_sknt_max'],
                                        'Test Label': test['OBS_sknt_max']}

    model = PCA(n_components=0.65, svd_solver='full')  # create the model
    # fit & reduce the training data
    rtn['Train Data'] = pd.DataFrame(model.fit_transform(train.drop('OBS_sknt_max', axis=1)), index=train.index)
    # reduce the test data
    rtn['Test Data'] = pd.DataFrame(model.transform(test.drop('OBS_sknt_max', axis=1)), index=test.index)

    spnr.write(success('reduction complete ---- '+u'\u2713'))
    return rtn


def split_random(data_in: pd.DataFrame, spnr) -> typ.Tuple[pd.DataFrame, pd.DataFrame]:
    """
    split_random will randomly split the provided Pandas dataframe into testing & training data
    (using SEED as the seed value).

    :param data_in: dataframe to be split
    :param spnr: spinner used for console printing

    :return: dataframes for train & test (in that order)
    """

    log.debug('random data split started')

    spnr.text = 'splitting data...'
    train = data_in.sample(frac=0.8, random_state=SEED)  # create a df with 80% of instances
    test = data_in.drop(train.index)                     # create a df with the remaining 20%

    spnr.write(success('data split complete --- '+u'\u2713'))
    return train, test


def split_fixed(data_in: pd.DataFrame, spnr) -> typ.Tuple[pd.DataFrame, pd.DataFrame]:
    """
    split_fixed will split the provided Pandas dataframe into testing & training data
    using the first 80% of the entries as training, and the rest as testing.

    :param data_in: dataframe to be split
    :param spnr: spinner used for console printing

    :return: dataframes for train & test (in that order)
    """

    log.debug('fixed data split started')

    num_rows: int = len(data_in.index)  # get the number of rows
    # get the number of instances to be added to the training data
    size_train: int = math.ceil(0.80 * num_rows)

    spnr.text = "splitting data..."
    # slice the dataframe [rows, cols] & grab everything up to size_train
    train: pd.DataFrame = data_in.iloc[:size_train+1, :]
    # slice the dataframe [rows, cols] & grab everything after size_train
    test: pd.DataFrame = data_in.iloc[size_train+1:, :]

    log.debug('fixed data split finished')
    spnr.write(success('data split complete --- '+u'\u2713'))
    return train, test
# ************************************************************** #


# *************************** REPORT *************************** #
def create_report(rand: typ.Dict[str, float], fixed: typ.Dict[str, float]) -> pd.DataFrame:
    """
    create_report will used the provided values to create a dataframe containing
    information on the error scores of the fixed & random linear regression models.

    :param rand: dictionary with the error scores for the rand model
    :param fixed: dictionary with the error scores for the fixed model


    :return: dataframes for train & test (in that order)
    """

    SYSOUT.write(HDR + ' Generating Report...'); SYSOUT.flush()

    cols: typ.List[str] = ['Regression (random)', 'Regression (fixed)']
    rws: typ.List[str] = ['Mean Absolute Error', 'Mean Squared Error', 'Mean Signed Error']

    # get the error values from the passed dictionaries
    results: np.array = np.array([[rand['absolute'], fixed['absolute']],
                                  [rand['squared'], fixed['squared']],
                                  [rand['signed'], fixed['signed']]], dtype=float)

    # create a dataframe of the result metrics
    df: pd.DataFrame = pd.DataFrame(results, columns=cols, index=rws, dtype=float)
    df.round(3)

    SYSOUT.write(OVERWRITE + ' Report Generated! '.ljust(44, '-') + SUCCESS); SYSOUT.flush()
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
    model.predict(data)
    prediction = [round(i, 5) for i in model.predict(data)]
    actual = label
    num_instances = len(actual)  # the number of examples in the test set

    # ! for debugging, print prediction to file
    # rst = list(zip(prediction, actual))
    # df = pd.DataFrame(rst, columns=['Prediction', 'Actual'])
    # df.to_csv(str(pth.Path.cwd() / 'logs' / 'predict.csv'))
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


# *************************** MODELS *************************** #
def run_regression(data_in: pd.DataFrame):
    """ run_regression will run a simple linear regression on the passed dataframe. """

    print(banner(' Starting Liner Regression '))
    data_in: pd.DataFrame = pd.DataFrame(data_in)

    def random_data() -> typ.Dict[str, float]:
        """Perform linear regression with randomly divided data"""
        model = LinearRegression()  # create the regression model

        train: pd.DataFrame
        test: pd.DataFrame
        train, test = split_random(data_in, spnr)  # randomly split dataset
        reduced: typ.Dict[str, pd.DataFrame] = reduce_data(train, test, spnr)

        # +++++++++++++++++++ Model Fit Notes +++++++++++++++++++ #
        # model.fit() takes 2 parameters: X & Y (in that order)
        # X = the labels we suspect are correlated
        # In this case it will be everything but Y
        # ! At this point we can't drop the col because they have no names because of reduce_data
        # perform preprocessing
        X: pd.DataFrame = reduced['Train Data']
        # Y = the target label (OBS_sknt_max for wind speed)
        Y: pd.DataFrame = reduced['Train Label']
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

        spnr.text = 'training model...'
        model.fit(X, Y)  # fit the model using X & Y (see above)
        spnr.write(success('training complete ----- '+u'\u2713'))

        spnr.text = 'getting error score...'
        # calculate the error scores
        results: typ.Dict = {
            'absolute': absolute_error(model, reduced['Test Data'], reduced['Test Label']),
            'squared': squared_error(model, reduced['Test Data'], reduced['Test Label']),
            'signed': signed_error(model, reduced['Test Data'], reduced['Test Label']),
        }

        spnr.write(success('error score computed -- ' + u'\u2713'))
        return results

    def fixed_data() -> typ.Dict[str, float]:
        """Perform linear regression with a fixed data division"""
        model = LinearRegression()  # create the regression model

        train: pd.DataFrame
        test: pd.DataFrame
        train, test = split_fixed(data_in, spnr)  # split dataset
        reduced: typ.Dict[str, pd.DataFrame] = reduce_data(train, test, spnr)

        # +++++++++++++++++++ Model Fit Notes +++++++++++++++++++ #
        # model.fit() takes 2 parameters: X & Y (in that order)
        # X = the labels we suspect are correlated
        # In this case it will be everything but Y
        # perform preprocessing
        X: pd.DataFrame = reduced['Train Data']
        # Y = the target label (OBS_sknt_max for wind speed)
        Y: pd.DataFrame = reduced['Train Label']
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

        spnr.text = 'training model...'
        model.fit(X, Y)  # fit the model using X & Y (see above)
        spnr.write(success('training complete ----- '+u'\u2713'))

        spnr.text = 'getting error score...'
        # calculate the error scores
        results = {
            'absolute': absolute_error(model, reduced['Test Data'], reduced['Test Label']),
            'squared': squared_error(model, reduced['Test Data'], reduced['Test Label']),
            'signed': signed_error(model, reduced['Test Data'], reduced['Test Label']),
        }

        spnr.write(success('error score computed -- ' + u'\u2713'))
        return results

    with yaspin(text='Starting Regression (random split)...') as spnr:
        spnr.write('Starting Regression (random split)...')
        rand = random_data()   # run the regression with a random data split

    with yaspin(text='Starting Regression (fixed split)...') as spnr:
        spnr.write('Starting Regression (fixed split)...')
        fixed = fixed_data()   # run the regression with a fixed data split

    print(banner(' Liner Regression Finished '))

    # pass the dicts with the error values to the report generator
    report: pd.DataFrame = create_report(rand, fixed)

    return report
# ************************************************************** #


# ************************* Poisoning ************************* #
def split_poison(data: pd.DataFrame, spnr) -> typ.Tuple[typ.List[pd.DataFrame], pd.DataFrame]:
    """
    split_poison will split the provided Pandas dataframe into testing & training data
    using the first 80% of the entries as training, and the rest as testing.
    This should method should only be used during poisoning attacks

    :param data: dataframe to be split
    :param spnr: spinner used for console printing

    :return: dataframes for train & test (in that order)
    """

    log.debug('fixed data split started')

    data = shuffle(data)  # randomly reorganize the dataframe befor splitting it

    num_rows: int = len(data.index)  # get the number of rows
    # get the number of instances to be added to the training data
    size_train: int = math.ceil(0.80 * num_rows)

    # * Divide Data into Testing & Training Sets * #
    spnr.text = "splitting data..."
    # slice [inclusive : not_inclusive]
    # slice the dataframe [rows, cols] & grab everything up to size_train
    train: pd.DataFrame = data.iloc[:size_train + 1, :]
    # slice the dataframe [rows, cols] & grab everything after size_train
    testing: pd.DataFrame = data.iloc[size_train + 1:, :]
    spnr.write(success('data split complete --- ' + u'\u2713'))

    # * Divide Training Data into 10 Buckets * #
    spnr.text = 'creating buckets...'
    # break the training data into BUCKETS_NUM number of smaller dataframes
    # with about the same number of instance in each
    spam: typ.List[np.ndarray] = np.array_split(train.to_numpy(), BUCKETS_NUM)
    training: typ.List[pd.DataFrame] = [pd.DataFrame(a, columns=train.columns) for a in spam]

    log.debug('fixed data split finished')
    spnr.write(success('bucket split complete '.ljust(23, '-') + u' \u2713'))
    return training, testing


def poison_regression(data_in: pd.DataFrame) -> pd.DataFrame:
    """
    poison_regression will run a simple linear regression on the passed dataframe
    & poison the training data.
    """
    print(banner(' Poisoning Liner Regression '))
    data_in: pd.DataFrame = pd.DataFrame(data_in)

    def poison_data(test: pd.DataFrame, train_list: typ.List[pd.DataFrame]) -> pd.DataFrame:
        """Perform linear regression with poisoned data"""
        model = LinearRegression()  # create the regression model

        spnr.text = 'joining dataframes...'
        # create a single dataframe from the list of dataframes
        train: pd.DataFrame = pd.concat(train_list)
        # BUCKETS_LABEL is used to save the number of instances used for each iteration,
        # so add to it here.
        global BUCKETS_LABEL
        BUCKETS_LABEL.append(len(train.index))

        # +++++++++++++++++++ Model Fit Notes +++++++++++++++++++ #
        # model.fit() takes 2 parameters: X & Y (in that order)
        # X = the labels we suspect are correlated
        # In this case it will be everything but Y
        X: pd.DataFrame = train.drop('OBS_sknt_max', axis=1)
        # Y = the target label (OBS_sknt_max for wind speed)
        Y: pd.DataFrame = train['OBS_sknt_max']
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

        spnr.text = 'training model...'
        model.fit(X, Y)  # fit the model using X & Y (see above)

        # * Calculate the Error Scores * #
        spnr.text = 'getting error score...'

        # name the columns for this instance
        cols: typ.List[str] = ['Bucket(s)',  'Mean Absolute Error', 'Mean Squared Error', 'Mean Signed Error']

        # get the data for the dataframe (should have the 3 error scores for this instance)
        results: np.array = np.array([[len(train_list),
                                       absolute_error(model, test.drop('OBS_sknt_max', axis=1), test['OBS_sknt_max']),
                                       squared_error(model, test.drop('OBS_sknt_max', axis=1), test['OBS_sknt_max']),
                                       signed_error(model, test.drop('OBS_sknt_max', axis=1), test['OBS_sknt_max'])]],
                                     dtype=float)

        # create a dataframe of the result metrics
        df: pd.DataFrame = pd.DataFrame(results, columns=cols, dtype=float)
        df.round(3)

        return df

    with yaspin(text='Starting Regression (poisoned)...') as spnr:
        spnr.write('Starting Regression (poisoned)...')
        training: typ.List[pd.DataFrame]
        testing: pd.DataFrame
        training, testing = split_poison(data_in, spnr)            # split the data into test & train buckets
        # preprocessing is done, train the models
        # create an empty dataframe to hold the error scores
        errs: typ.List[pd.DataFrame] = []
        while training:  # while train_list still has training data,
            # train the regression using what's left of train_list, & add results to errs list
            errs.append(poison_data(testing, training))
            training.pop()  # remove the last item from the training data
            spnr.write(success(f'model {len(errs)}/10 complete '.ljust(23, '-') + u' \u2713'))

    report = pd.concat(errs)  # transform the list of frames into a single frame

    print(banner(' Poisoning Finished '))
    return report


def grid_plot(df: pd.DataFrame, file: str):
    """
    scatter_plot creates a scatter plot of the dataframe using
    Pandas libraries.
    """

    # TODO: make it display more values along the y axis
    # create the figure that will hold all 4 plots
    fig, axes = plt.subplots(nrows=3, ncols=1)

    # create the plot & place it in the upper left corner
    df.plot(ax=axes[0],
            kind='line',
            x='Bucket(s)',
            y='Mean Absolute Error',
            color='blue',
            style='--',  # the line style
            legend=True)
    # axes[0].set_title('Mean Absolute Error')

    # create the plot & place it in the upper right corner
    df.plot(ax=axes[1],
            kind='line',
            x='Bucket(s)',
            y='Mean Squared Error',
            color='red',
            style='-.',  # the line style
            legend=True)
    # axes[1].set_title('Mean Squared Error')

    # create the plot & place it in the lower left corner
    df.plot(ax=axes[2],
            kind='line',
            x='Bucket(s)',
            y='Mean Signed Error',
            style='-',  # the line style
            color='green',
            legend=True)
    # axes[2].set_title('Mean Signed Error')

    # save the plot to the provided file path
    plt.savefig(file)
    # show the plot
    plt.show()

    return


def error_plot(df: pd.DataFrame, file: str):
    """
    error_plot creates a line plot of the report dataframe using
    Pandas libraries, & plots all 3 error scores on the same figure.
    """

    # get the axes so the plots can be made on the same figure
    ax = plt.gca()
    # set the values for the 'Bucket(s)' axis
    ax.set_xticks(BUCKETS_LABEL)
    ax.set_ylabel('Error Score')  # label the y-axis

    # plot the mean absolute error
    df.plot(ax=ax,
            kind='line',
            x='Bucket(s)',
            y='Mean Absolute Error',
            color='blue',
            style='--',  # the line style
            x_compat=True,
            use_index=True,
            grid=True,
            legend=True)

    # plot the mean squared error
    df.plot(ax=ax,
            kind='line',
            x='Bucket(s)',
            y='Mean Squared Error',
            color='red',
            style='-.',  # the line style
            x_compat=True,
            use_index=True,
            grid=True,
            legend=True)

    # plot the mean signed error
    df.plot(ax=ax,
            kind='line',
            x='Bucket(s)',
            y='Mean Signed Error',
            style=':',  # the line style
            x_compat=True,
            color='green',
            use_index=True,
            grid=True,
            legend=True)

    # save the plot to the provided file path
    plt.savefig(file)
    # show the plot
    plt.show()

    return


if __name__ == '__main__':
    log.debug('Starting...')

    # *** Set the Program's Parameters & Constants *** #
    # (we are not inside a function so the global keyword isn't needed) #
    usr_in = argumentParser.parse_args()  # grab the input flags

    BUCKETS_NUM = usr_in.bNum       # BUCKETS_NUM is the number of 'buckets' to used in poisoning splitting
    RUN_STND_RGRS = usr_in.regress  # RUN_STND_RGRS tells the program if the standard regression (no poison) should run
    SEED = usr_in.sd                # SEED is used as the seed value for random number generation
    BUCKETS_LABEL = []              # BUCKETS_LABEL is used to label the x-axis of the error score plot

    # *** Seed the Random Libraries Using the Provided Seed Value *** #
    random.seed(SEED)
    np.random.seed(SEED)

    # *** Used for Printing *** #
    HDR = '*' * 6
    SUCCESS = u' \u2713\n' + '\033[0m'     # print the checkmark & reset text color
    OVERWRITE = '\r' + '\033[32;1m' + HDR  # overwrite previous text & set the text color to green
    NO_OVERWRITE = '\033[32;1m' + HDR      # NO_OVERWRITE colors lines green that don't use overwrite
    SYSOUT = sys.stdout                    # SYSOUT set the standard out for the program to the console

    main()
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