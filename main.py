"""
main.py creates, trains, and tests machine learning models when provided with test & training data sets.

Authors/Contributors: Dr. Dimitrios Diochnos, Conner Flansburg

Github Repo:
"""

import sys
import math
import random
# import traceback
import logging as log
import numpy as np
import pandas as pd
import pathlib as pth
import typing as typ
import matplotlib.pyplot as plt
from collections import namedtuple
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
# from pprint import pprint
from formatting import banner, printError, success
from pyfiglet import Figlet
from sklearn.linear_model import LinearRegression
from yaspin import yaspin


SEED: int = 368
HDR = '*' * 6
SUCCESS = u' \u2713\n'+'\033[0m'       # print the checkmark & reset text color
OVERWRITE = '\r' + '\033[32;1m' + HDR  # overwrite previous text & set the text color to green
NO_OVERWRITE = '\033[32;1m' + HDR      # NO_OVERWRITE colors lines green that don't use overwrite
SYSOUT = sys.stdout                    # SYSOUT set the standard out for the program to the console

# Set up the logger
log_path = pth.Path.cwd() / 'logs' / 'log.txt'
log.basicConfig(level=log.ERROR, filename=str(log_path), format='%(levelname)s-%(message)s')

# set the seed for the random library
random.seed(SEED)

# TODO: figure out poison reduce


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
    SYSOUT.write(OVERWRITE + ' File read in successfully! '.ljust(44, '-') + SUCCESS)
    SYSOUT.flush()
    log.debug('Data read in successfully')

    # * Preprocess Data * #
    SYSOUT.write(HDR + ' Scaling data...')
    SYSOUT.flush()
    df_in = scale_data(df_in)
    SYSOUT.write(OVERWRITE + ' Data Scaling finished! '.ljust(44, '-') + SUCCESS)
    SYSOUT.flush()
    log.debug('Data Scaling finished')

    # * Preform the Linear Regression * #
    report: pd.DataFrame = run_regression(df_in)
    log.debug('Regression performed successfully')

    # * Display & Save Report * #
    print(f"\n    {banner(' Regression Report ')}")
    report = report.round(decimals=3)     # format the data frame
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.precision', 3):
        print(report)                                # display results
    rOut = pth.Path.cwd() / 'output' / 'regression_report.csv'  # create file path
    report.to_csv(str(rOut))                         # save the results to a file
    print('')  # print newline after the report

    # * Run a Poisoning Attack on the Linear Regression * #
    report = poison_regression(df_in)

    # * Display & Save Report * #
    print(f"\n    {banner(' Poisoning Report ')}")
    report = report.round(decimals=3)  # format the data frame
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.precision', 3):
        print(report)  # display results
    rOut = pth.Path.cwd() / 'output' / 'poisoning_report.csv'  # create file path
    report.to_csv(str(rOut))  # save the results to a file
    print('')  # print newline after the report
    # create scatter plot of the mean squared error rate vs number of buckets
    scatter_plot(report, x_axis='bucket(s)', y_axis='Mean Squared Error')

    # * Run the Neural Network * #
    # run_network()

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
    # TODO: comment
    log.debug('random data split started')

    spnr.text = 'splitting data...'
    train = data_in.sample(frac=0.8, random_state=SEED)  # create a df with 80% of instances
    test = data_in.drop(train.index)                     # create a df with the remaining 20%

    spnr.write(success('data split complete --- '+u'\u2713'))
    return train, test


def split_fixed(data_in: pd.DataFrame, spnr) -> typ.Tuple[pd.DataFrame, pd.DataFrame]:
    # TODO: comment
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

    SYSOUT.write(OVERWRITE + ' Report Generated! '.ljust(44, '-') + SUCCESS)
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
    model.predict(data)
    prediction = [round(i, 5) for i in model.predict(data)]
    actual = label
    num_instances = len(actual)  # the number of examples in the test set

    # ! for debugging, print prediction to file
    rst = list(zip(prediction, actual))
    df = pd.DataFrame(rst, columns=['Prediction', 'Actual'])
    df.to_csv(str(pth.Path.cwd() / 'logs' / 'predict.csv'))
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


# ************************* Poisoning ************************* #
# TODO: change the caller of split_poison to use the new return type
Datum = namedtuple('Datum', ['Ftrs', 'Label'])


def split_poison(data_in: pd.DataFrame, spnr) -> typ.Tuple[typ.List[Datum], Datum]:
    # TODO: comment
    log.debug('fixed data split started')

    num_rows: int = len(data_in.index)  # get the number of rows
    # get the number of instances to be added to the training data
    size_train: int = math.ceil(0.80 * num_rows)

    # * Divide Data into Testing & Training Sets * #
    spnr.text = "splitting data..."
    # slice [inclusive : not_inclusive]
    # slice the dataframe [rows, cols] & grab everything up to size_train
    train: pd.DataFrame = data_in.iloc[:size_train + 1, :]
    # slice the dataframe [rows, cols] & grab everything after size_train
    test: pd.DataFrame = data_in.iloc[size_train + 1:, :]
    spnr.write(success('data split complete --- ' + u'\u2713'))

    # * Reduce the Data * #
    # ? should this happen before or after the bucket split?
    reduced = reduce_data(data_in.iloc[:size_train + 1, :], data_in.iloc[size_train + 1:, :], spnr)

    # * Divide Training Data into 10 Buckets * #
    spnr.text = 'creating buckets...'
    # now that the data has been split into train & test, divide train into 10 buckets
    # each bucket should get 251 examples except for the last one which will get the rest.
    # Each bucket should contain the data from the reduced dataframe.
    train_list: typ.List[Datum]

    # ? change so that the first 'Datum' has the extra instances
    training = [  # bucket 1  instances 0 - 251
                Datum(Ftrs=reduced['Train Data'].iloc[:251, :], Label=train['OBS_sknt_max'].iloc[:251]),
                  # bucket 2  instances 252 - 502
                Datum(Ftrs=reduced['Train Data'].iloc[252:503, :], Label=train['OBS_sknt_max'].iloc[252:503]),
                  # bucket 3  instances 502 - 753
                Datum(Ftrs=reduced['Train Data'].iloc[503:754, :], Label=train['OBS_sknt_max'].iloc[503:754]),
                  # bucket 4  instances 753 - 1004
                Datum(Ftrs=reduced['Train Data'].iloc[754:1005, :], Label=train['OBS_sknt_max'].iloc[754:1005]),
                  # bucket 5  instances 1004 - 1255
                Datum(Ftrs=reduced['Train Data'].iloc[1005:1256, :], Label=train['OBS_sknt_max'].iloc[1005:1256]),
                  # bucket 6  instances 1255 - 1506
                Datum(Ftrs=reduced['Train Data'].iloc[1256:1507, :], Label=train['OBS_sknt_max'].iloc[1256:1507]),
                  # bucket 7  instances 1506 - 1757
                Datum(Ftrs=reduced['Train Data'].iloc[1507:1758, :], Label=train['OBS_sknt_max'].iloc[1507:1758]),
                  # bucket 8  instances 1757 - 2008
                Datum(Ftrs=reduced['Train Data'].iloc[1758:2009, :], Label=train['OBS_sknt_max'].iloc[1758:2009]),
                  # bucket 9  instances 2008 - 2259
                Datum(Ftrs=reduced['Train Data'].iloc[2009:2260, :], Label=train['OBS_sknt_max'].iloc[2009:2260]),
                  # bucket 10 instances 2259 - end
                Datum(Ftrs=reduced['Train Data'].iloc[2260:, :], Label=train['OBS_sknt_max'].iloc[2260:])
    ]

    # create the testing tuple using the reduced datset & the original label
    testing = Datum(Ftrs=reduced['Test Data'], Label=test['OBS_sknt_max'])

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

    def poison_data(test: Datum, train_list: typ.List[Datum]) -> pd.DataFrame:
        """Perform linear regression with poisoned data"""
        model = LinearRegression()  # create the regression model

        spnr.text = 'joining dataframes...'
        # create a single dataframe from the list of Datum tuples
        train: pd.DataFrame = pd.concat([df.Ftrs for df in train_list])
        # the training data is now joined into a single data frame
        # now do the same for the training labels
        labels: pd.DataFrame = pd.concat([df.Label for df in train_list])

        # +++++++++++++++++++ Model Fit Notes +++++++++++++++++++ #
        # model.fit() takes 2 parameters: X & Y (in that order)
        # X = the labels we suspect are correlated
        # In this case it will be everything but Y
        X: pd.DataFrame = train
        # Y = the target label (OBS_sknt_max for wind speed)
        Y: pd.DataFrame = labels
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

        spnr.text = 'training model...'
        model.fit(X, Y)  # fit the model using X & Y (see above)

        spnr.text = 'getting error score...'
        # calculate the error scores

        # name the columns for this instance
        cols: typ.List[str] = ['bucket(s)',  'Mean Absolute Error',
                               'Mean Squared Error', 'Mean Signed Error']

        # get the data for the dataframe (should have the 3 error scores for this instance)
        results: np.array = np.array([[len(train_list),
                                       absolute_error(model, test.Ftrs, test.Label),
                                       squared_error(model, test.Ftrs, test.Label),
                                       signed_error(model, test.Ftrs, test.Label)]],
                                     dtype=float)

        # create a dataframe of the result metrics
        df: pd.DataFrame = pd.DataFrame(results, columns=cols, dtype=float)
        df.round(3)

        return df

    with yaspin(text='Starting Regression (poisoned)...') as spnr:
        spnr.write('Starting Regression (poisoned)...')
        training: typ.List[Datum]
        testing: Datum
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


def scatter_plot(df: pd.DataFrame, x_axis: str, y_axis: str):
    # TODO: create graph

    # scatter plot
    df.plot(kind='scatter',
            x=x_axis,
            y=y_axis,
            color='red')

    # set the title
    plt.title('ScatterPlot')

    # show the plot
    plt.show()

    pass


if __name__ == '__main__':
    log.debug('Starting...')
    main()
    log.debug('closing...\n')
