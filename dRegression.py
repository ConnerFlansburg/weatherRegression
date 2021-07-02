#
# Simple attempt to run linear regression towards predicting wind
#

import sys
import math
import random
import logging as log

import numpy as np
import pandas as pd
import pathlib as pth

from pyfiglet import Figlet
from formatting import banner, printError, success
from alive_progress import alive_bar, config_handler

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from numpy import median
# from sklearn.preprocessing import MinMaxScaler --- we will see if we need this
from create_plot import plot_histogram

# ! suppress sci-kit warnings
import warnings
from scipy.linalg import LinAlgWarning
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=LinAlgWarning)

# ===================================================================================

#
# Global variables and preliminary stuff
#

SEED: int = 368
HDR = '*' * 6
SUCCESS = u' \u2713\n'+'\033[0m'       # print the checkmark & reset text color
OVERWRITE = '\r' + '\033[32;1m' + HDR  # overwrite previous text & set the text color to green
NO_OVERWRITE = '\033[32;1m' + HDR      # NO_OVERWRITE colors lines green that don't use overwrite
SYSOUT = sys.stdout                    # SYSOUT set the standard out for the program to the console

# Set up the logger
log_path = pth.Path.cwd() / 'logs' / 'log.txt'
log.basicConfig(level=log.ERROR, filename=str(log_path), format='%(levelname)s-%(message)s')

#  Configuration of Progress Bar
# the global config for the loading bars
config_handler.set_global(spinner='dots_reverse', bar='classic', unknown='stars',
                          title_length=0, length=20, enrich_print=False)

# set the seed for the random library
random.seed(SEED)

# training set ratio
TRAINING_SET_RATIO: float = 0.8
NUM_BUCKETS: int = 5
SMOOTH_ITERATIONS: int = 5

# ===================================================================================


def read_in(fl: str) -> pd.DataFrame:
    """ read_in reads in the csv & transforms it into a Pandas Dataframe. """
    # * Read in the Data * # (treating '********' as NaNs)
    data: pd.DataFrame = pd.read_csv(fl, dtype=float, na_values='********', parse_dates=['date'], index_col=0).sort_values(by='date')

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


def split_into_training_and_test_data(our_data_frame, num_rows, num_cols):
    training = []
    testing  = []
    num_training = int(round(TRAINING_SET_RATIO * num_rows))
    num_testing  = num_rows - num_training
    for i in range(num_rows):
        df_current_row = our_data_frame.iloc[i]
        assert len(df_current_row) == num_cols
        if i < num_training:
            training.append(df_current_row)
        else:
            testing.append(df_current_row)
    return training, testing


def filter_dataset_and_split_instances_and_labels (some_dataset):
    labels = []
    filtered_instances = []
    for i in range(len(some_dataset)):
        labels.append(some_dataset[i][3])
        current_row = []
        for j in range(len(some_dataset[i])):
            if j <= 4:
                pass
            else:
                current_row.append(some_dataset[i][j])
        filtered_instances.append(current_row)
    return filtered_instances, labels


def permute_instances_and_labels_using_permutation (instances, labels, perm):
    assert len(instances) == len(perm)
    assert len(labels) == len(perm)
    inverse = []
    for i in range(len(perm)):
        inverse.append(-1)
    for i in range(len(perm)):
        inverse[perm[i]] = i
    # now create the permuted instances and labels
    perm_instances = []
    perm_labels = []
    for i in range(len(perm)):
        perm_instances.append(instances[inverse[i]])
        perm_labels.append(labels[inverse[i]])
    return perm_instances, perm_labels


def calculate_averages(all_mse):
    size = len(all_mse[0])  # get the number of values from 1 smoothing loop
    averages = []
    # loop over the outer list & for each list of values from a smoothing iter calc the median
    for j in range(size):
        averages.append(median(all_mse[j]))
    return averages


def write_error_to_files(kind_string, all_error, error_avgs, breakpoints):
    fd_all = open(kind_string + "_all.txt", "w")
    fd_min = open(kind_string + "_min.csv", "w")
    fd_max = open(kind_string + "_max.csv", "w")
    fd_mdn = open(kind_string + "_mdn.csv", "w")
    the_iters = len(all_error)
    size = len(all_error[0])
    assert len(breakpoints) == size
    for j in range(size):
        min = all_error[0][j]
        max = all_error[0][j]
        fd_all.write("%d: " %(breakpoints[j]))
        for i in range(the_iters):
            fd_all.write("%.6lf " %(all_error[i][j]))
            if all_error[i][j] < min:
                min = all_error[i][j]
            if all_error[i][j] > max:
                max = all_error[i][j]
        fd_all.write("\n")  # TODO: change to csv?
        # write 1 line to the min file
        fd_min.write(f"{breakpoints[j]},{min}\n")
        # write 1 line to the max file
        fd_max.write(f"{breakpoints[j]},{max}\n")
        # write 1 line to the averages file
        fd_mdn.write(f"{breakpoints[j]},{error_avgs[j]}\n")
    fd_all.close()
    fd_min.close()
    fd_max.close()
    fd_mdn.close()


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


    # df_in contains the original dataset
    num_rows = df_in.shape[0]
    num_cols = df_in.shape[1]
    #print('num rows: %d' %(num_rows))
    #print('num cols: %d' %(num_cols))

    SYSOUT.write(HDR + ' Splitting into training and test set ...')
    SYSOUT.flush()
    prelim_train_data, prelim_test_data = split_into_training_and_test_data(df_in, num_rows, num_cols)
    SYSOUT.write(OVERWRITE + ' Splitted into training and test set successfully! '.ljust(44, '-') + SUCCESS)
    SYSOUT.flush()
    log.debug('Splitted into training and test set successfully')

    SYSOUT.write(HDR + ' Splitting datasets into instances and labels ...')
    SYSOUT.flush()
    our_training_instances, our_training_labels = filter_dataset_and_split_instances_and_labels(prelim_train_data)
    our_test_instances, our_test_labels = filter_dataset_and_split_instances_and_labels(prelim_test_data)
    SYSOUT.write(OVERWRITE + ' Splitted datasets into instances and labels successfully! '.ljust(44, '-') + SUCCESS)
    SYSOUT.flush()
    log.debug('Splitted datasets into instances and labels successfully')
    # print('size of training instances: %d' %(len(our_training_instances)))
    # print('size of training labels: %d' %(len(our_training_labels)))

    # * Loop over buckets * #
    squared_errors = []  # a record of the observed mean squared errors
    root_errors = []  # a record of the observed root mean errors
    breakpoints = []
    all_training_sets = []  # collect every training label used
    for s in range(SMOOTH_ITERATIONS):

        the_permutation = np.random.permutation(len(our_training_instances))
        our_permuted_training_instances, our_permuted_training_labels = permute_instances_and_labels_using_permutation(
            our_training_instances, our_training_labels, the_permutation)

        current_iter_mse = []
        current_iter_rmse = []

        # ! attempt to plot the distribution ! #
        all_training_sets.extend(our_training_labels)
        # plot_histogram(our_training_labels, 'Training')
        # plot_histogram(our_test_labels, 'Testing')

        # Work with subsets of the whole training set -- useful for poisoning as well
        bucket_size = int(math.floor(len(our_training_instances) / NUM_BUCKETS))
        with alive_bar(NUM_BUCKETS, title=f'Smoothing {s+1}/{SMOOTH_ITERATIONS}') as bar:
            for b in range(NUM_BUCKETS):

                # * Get the size of the training dataset& set the number of buckets * #
                if b < NUM_BUCKETS - 1:
                    current_size = bucket_size * (b + 1)
                else:
                    current_size = len(our_permuted_training_instances)
                current_instances = []
                current_labels = []
                if s == 0:
                    breakpoints.append(current_size)

                # * Create a subset of the training data * #
                for i in range(current_size):
                    current_instances.append(our_permuted_training_instances[i])
                    current_labels.append(our_permuted_training_labels[i])

                # * Perform linear regression * #
                assert len(current_instances) == current_size  # check that dimensions are correct
                assert len(current_labels) == current_size     # check that dimensions are correct
                # run a linear regression
                # clf = LinearRegression(fit_intercept=True).fit(current_instances, current_labels)
                # clf = Lasso(0.01)  # run a lasso regression
                clf = Ridge(0.01)  # run a ridge regression
                clf.fit(current_instances, current_labels)  # fit the model with the current training set

                # * Predict values * #
                predicted = clf.predict(our_test_instances)

                # * Calculate (root) mean squared error * #
                # calculate the errors
                our_mse = round(mean_squared_error(our_test_labels, predicted), 4)
                our_rmse = round(math.sqrt(our_mse), 4)

                # add the errors to the list
                current_iter_mse.append(our_mse)
                current_iter_rmse.append(our_rmse)

                # * Print the mean squared error * #
                # print('The mean squared error is: %.15lf' %(our_mse))
                # print('The root mean squared error is: %.15lf' %(our_rmse))
                bar()  # update the loading bar

        print(f'median sq. error = {median(current_iter_mse)}')
        print(f'max sq. error = {max(current_iter_mse)}')
        print(f'min sq. error = {min(current_iter_mse)}')
        # add the current error values to the global list
        squared_errors.append(current_iter_mse)
        root_errors.append(current_iter_rmse)

    # ! plot histogram of all training sets
    plot_histogram(all_training_sets, 'All Training')

    # get the average error
    mse_median = calculate_averages(squared_errors)
    rmse_median = calculate_averages(root_errors)
    # display it for the user
    # print(mse_avgs)
    # print(rmse_avgs)
    # write it to a file
    write_error_to_files("mse", squared_errors, mse_median, breakpoints)
    write_error_to_files("rmse", root_errors, rmse_median, breakpoints)


if __name__ == "__main__":
    main()
