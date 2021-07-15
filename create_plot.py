
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pathlib as pth
import numpy as np
import sys
import typing as typ
from collections import Counter
from formatting import printWarn

# TODO: change it so the median & average for the mean squared error are on the same plot
# TODO: change it so the median & average for the root mean squared error are on the same plot


def plot_values(title, fl, inpt):

    # * read in the file as a dataframe * #
    averages: pd.DataFrame = pd.read_csv(inpt, header=None, names=['Size', 'Error'])

    # * create a line plot of the training size vs error * #
    plt.title(title)
    ax = plt.gca()                  # get the current ax
    ax.set_ylabel('Error Score')    # label the y-axis
    ax.set_xlabel('Training Size')  # label the x-axis

    fig = plt.gcf()                 # get the current figure
    fig.set_size_inches(14, 9)      # set the size of the image (width, height)

    # how many degrees the x-axis labels should be rotated
    rotate = 0
    # rotate = 45
    # rotate = 90

    averages.plot(ax=ax,           # set the axis to the current axis
                  kind='line',
                  x='Size',
                  y='Error',
                  color='blue',
                  style='--',  # the line style
                  x_compat=True,
                  rot=rotate,  # how many degrees to rotate the x-axis labels
                  # use_index=True,
                  grid=True,
                  legend=True,
                  # marker='o',    # what type of data markers to use?
                  # mfc='black'    # what color should they be?
                  )

    fig.tight_layout()  # tighten the layout
    plt.savefig(fl)     # save the plot
    # ! BUG: not showing the plot causes a bunch of lines to show up
    plt.show()          # show the plot
    return


def beaufort_scale(original: float) -> int:

    # speed should be in m/s, round it
    original = round(original, 1)

    # Determine the Beaufort Scale value & return it
    if original < 0.5:
        return 0

    elif 0.5 <= original <= 1.5:
        return 1

    elif 1.6 <= original <= 3.3:
        return 2

    elif 3.4 <= original <= 5.5:
        return 3

    elif 5.5 <= original <= 7.9:
        return 4

    elif 8 <= original <= 10.7:
        return 5

    elif 10.8 <= original <= 13.8:
        return 6

    elif 13.9 <= original <= 17.1:
        return 7

    elif 17.2 <= original <= 20.7:
        return 8

    elif 20.8 <= original <= 24.4:
        return 9

    elif 24.5 <= original <= 28.4:
        return 10

    elif 28.5 <= original <= 32.6:
        return 11

    elif 32.7 <= original:
        return 12

    # if this has been reached then an error has occured
    printWarn(f'ERROR: beaufort_scale expected flot got {original}, {type(original)}')
    sys.exit(-1)  # cannot recover from this error, so exit


# TODO: Check that this runs correctly
# ? Should I be taking the average of this range (start:stop) over every permutation?
# TODO: Code Version 1 - use only this current permutation, making a new one for each perm
# TODO: Code Version 2 - use all permutations, making one for each index

def plot_spikes(city: str, start_index: int, stop_index: int, perm: int):
    """
    This will create a histogram for the index range pass using
    only the passed permutation.
    """

    # * Read in the Permutated Wind Speeds * #
    path: str = str(pth.Path.cwd() / 'output' / f'{city}' / 'permutation' / f'wind_speed_{perm}.csv')
    permutation: pd.DataFrame = pd.read_csv(path, names=['Wind'])

    # * Get the Wind Speeds of Interest in the Beafort Scale * #
    wind_speeds = [beaufort_scale(i) for i in permutation[start_index:stop_index+1]]

    # * Plot the Beaufort Scale Values using a Histogram * #
    sns.set(
        font_scale=2,                    # resizes the font (of everything)
        rc={"figure.figsize": (14, 9)},  # set the size of the plot
        style="darkgrid"                 # sets the plots style
    )
    plt.title(f'{city} {start_index}-{stop_index} Spike Distribution')  # set the plots title

    bins = 50  # the number of bins to use
    sns.histplot(            # make the histogram
        data=wind_speeds,    # the dataframe
        x='Wind',            # the column from the dataframe
        bins=bins,
        color='blue',
        alpha=0.5,           # how transparent the bar should be (1=solid, 0=invisible)
        # kde=True,          # draw a density line
        stat='probability',  # make the histogram show percentages
        discrete=True,
    )

    # * Display the Plot & Save it * #
    save = str(pth.Path.cwd() / 'output' / f'{city}' / f'{city}_spike_distribution.png')
    ax = plt.gca()                         # get the current ax
    ax.set_xlabel('Beaufort Scale Value')  # label the x-axis
    plt.legend()                           # show the legend
    plt.savefig(save)                      # save the plot
    plt.show()                             # display the plot


def plot_spikes_avr(city: str, start_index: int, stop_index: int):
    """
    This will create a histogram for the index range pass using every permutation.
    """
    # ? should I take the average or just use all the values?
    avr_speed_lists = []  # this will be used if we want to average the values of our indexes
    speed_list = []  # this will be used if we want to use every wind speed (no average)

    for perm in range(10):
        # Read in the Current Permutation
        path: str = str(pth.Path.cwd() / 'output' / f'{city}' / 'permutation' / f'wind_speed_{perm}.csv')
        permutation: pd.DataFrame = pd.read_csv(path, names=['Wind'])
        # Convert the Current Permutation to a List of Beaufort Scale Values
        speeds: typ.List[int] = [beaufort_scale(i) for i in permutation[start_index:stop_index + 1]]

        # ! Pick One: either average values or all values
        avr_speed_lists.append(speeds)  # add a new row to the matrix
        speed_list.extend(speeds)       # extend the list of speeds to include the new values

    # ! If averaging, average columns
    wind_speeds = np.average(avr_speed_lists, axis=0)
    # ! If not averaging, use every value
    wind_speeds = speed_list

    # * Plot the Beaufort Scale Values using a Histogram * #
    sns.set(
        font_scale=2,                    # resizes the font (of everything)
        rc={"figure.figsize": (14, 9)},  # set the size of the plot
        style="darkgrid"                 # sets the plots style
    )
    plt.title(f'{city} {start_index}-{stop_index} Spike Distribution Over Smooth Iter')  # set the plots title

    bins = 50  # the number of bins to use
    sns.histplot(            # make the histogram
        data=wind_speeds,    # the dataframe
        x='Wind',            # the column from the dataframe
        bins=bins,
        color='blue',
        alpha=0.5,           # how transparent the bar should be (1=solid, 0=invisible)
        # kde=True,          # draw a density line
        stat='probability',  # make the histogram show percentages
        discrete=True,
    )

    # * Display the Plot & Save it * #
    save = str(pth.Path.cwd() / 'output' / f'{city}' / f'{city}_spike_distribution_smooth.png')
    ax = plt.gca()                         # get the current ax
    ax.set_xlabel('Beaufort Scale Value')  # label the x-axis
    plt.legend()                           # show the legend
    plt.savefig(save)                      # save the plot
    plt.show()                             # display the plot


def plot_histograms():

    # create a list of every csv file
    csv_files = [
        'kast', 'kboi', 'kbro', 'kchs', 'kcmh', 'kcys',
        'kdbq', 'kdlh', 'keug', 'kgeg', 'kjax', 'klch',
        'klit', 'koma', 'kroa', 'kdfw'
    ]

    # loop over each file
    metadata = []  # this will be used to store the rows created inside the loop
    scores = []  # records the score frequencies
    for csv in csv_files:
        # * Read in the File * #
        fl = str(pth.Path.cwd() / 'data' / f'{csv}_processed_data.csv')
        df: pd.DataFrame = pd.read_csv(
            fl, dtype=float, na_values='********',
            parse_dates=['date'], index_col=0
        )
        df.sort_values(by='date')  # sort the dataframe by date

        # !!!!!!!!!! Get the NaN Report !!!!!!!!!! #
        # zero_report(df, csv)
        # !! Create a Row in the Metadata Table !! #
        # row = [len(df.index), len(df.columns)]  # (stem, instances, features)
        # metadata.append(row)
        # !!! Create a Row in the Score Report !!! #
        # row = scale_values_report(df, csv)
        # scores.append(row)
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #

        # * Get rid of NaN & infinity * #
        df = df.replace('********', np.nan).replace(np.inf, np.nan)
        df = df.dropna(how='any', axis=1)

        # * Get the Wind Speeds Values from the Data * #
        data = list(df['OBS_sknt_max'])

        # * Slice the Data into Training & Testing * #
        train_data = data[: int(len(data) * .80)+1]  # up to & including 80%
        test_data = data[int(len(data) * .80)+1:]    # everything after 80%

        # * Convert the Data to Beaufort Scale * #
        beaufort_values = pd.DataFrame(  # convert the values to the beaufort scale & put in a dataframe
            list(zip([beaufort_scale(i) for i in train_data], [beaufort_scale(i) for i in test_data])),
            columns=['Training Data', 'Testing Data']
        )

        # * Plot the Data * #
        sns.set(font_scale=2)   # resizes the font (of everything)
        sns.set_style("darkgrid")  # set the style
        # set the figures properties
        plt.figure(figsize=(14, 9), constrained_layout=True)  # set the size of the figure (width, height)
        plt.title(f'{csv} distribution')                      # set the plots title

        bins = 50  # the number of bins to use
        # Testing Data
        sns.histplot(
            data=beaufort_values,  # the dataframe
            x='Testing Data',      # the column from the dataframe
            label='Testing',
            bins=bins,
            color='blue',
            alpha=0.5,             # how transparent the bar should be (1=solid, 0=invisible)
            # kde=True,            # draw a density line
            stat='probability',    # make the histogram show percentages
            discrete=True,
        )

        # Training Data
        sns.histplot(
            data=beaufort_values,  # the dataframe
            x='Training Data',     # the column from the dataframe
            label='Training',
            bins=bins,
            color='yellow',
            alpha=0.5,             # how transparent the bar should be (1=solid, 0=invisible)
            # kde=True,            # draw a density line
            stat='probability',    # make the histogram show percentages
            discrete=True,
        )

        # * Display the Plot & Save it * #
        ax = plt.gca()  # get the current ax
        ax.set_xlabel('Beaufort Scale Value')  # label the x-axis
        plt.legend()        # show the legend
        save = str(pth.Path.cwd() / 'output' / 'histograms' / f'bScale_histogram_{csv}.png')
        plt.savefig(save)   # save the plot
        # plt.show()        # display the plot

    # * Create Metadata * #
    # turn metadata into a dataframe
    # meta_report = pd.DataFrame(np.array(metadata), columns=['Instances', 'Features'], index=csv_files)
    # meta_report.to_csv(str(pth.Path.cwd() / 'logs' / f'features_and_instances.csv'))  # export as csv

    # score_report = pd.DataFrame(np.array(scores), index=csv_files)  # turn score report into a dataframe
    # with open(str(pth.Path.cwd() / 'logs' / f'beaufort_scores_table.tex'), 'w') as f:
    #     f.write(score_report.to_latex())           # export the report as a latex table

    return


def zero_report(full_df: pd.DataFrame, stem: str):

    # this will hold the reports counters
    report = {
        'NaNs': 0,
        'NaN Rows': 0,
        'Zeros': 0,
        '0.5s': 0
    }

    for row, speed in enumerate(list(full_df['OBS_sknt_max'])):
        # check each 'row' for NaNs, 0s, & low values
        if np.isnan(speed):          # if the speed is a NaN value
            report['NaNs'] += 1      # increment counter
            report['NaN Rows'] += 1  # get row
        elif speed == 0:             # if the speed is less than 0
            report['Zeros'] += 1     # increment counter
        elif 0 < speed < 0.5:        # if the speed is less than 0.5
            report['0.5s'] += 1      # increment counter

    # print results
    # print(f'The Number of NaNs: {report["NaNs"]}')
    # print(f'The Number of True Zeros: {report["Zeros"]}')
    # print(f'The Number of Low Values: {report["0.5s"]}\n')

    # export & append the report for each city to the total report
    file_path = str(pth.Path.cwd() / 'logs' / 'NaN_Report.txt')
    with open(file_path, "a") as f:
        f.write(f'{stem} Report:\n')
        f.write(f'The Number of NaNs: {report["NaNs"]}\n')
        if report['NaN Rows'] != 0:
            f.write(f'NaNs Occur on Rows: {report["NaN Rows"]}\n')
        f.write(f'The Number of True Zeros: {report["Zeros"]}\n')
        f.write(f'The Number of Low Values: {report["0.5s"]}\n\n')

    return


def scale_values_report(data: pd.DataFrame):
    """ return a single row of the report, collate in main """

    df = [beaufort_scale(i) for i in data['OBS_sknt_max']]  # convert to beaufort scale

    # get a dict with the beaufort value as the key & the frequency as the value/entry
    counts = Counter(df)
    row = [          # create a row for the final report
        counts[0],   # number of times a score of 0 was recorded
        counts[1],   # number of times a score of 1 was recorded
        counts[2],   # number of times a score of 2 was recorded
        counts[3],   # number of times a score of 3 was recorded
        counts[4],   # number of times a score of 4 was recorded
        counts[5],   # number of times a score of 5 was recorded
        counts[6],   # number of times a score of 6 was recorded
        counts[7],   # number of times a score of 7 was recorded
        counts[8],   # number of times a score of 8 was recorded
        counts[9],   # number of times a score of 9 was recorded
        counts[10],  # number of times a score of 10 was recorded
        counts[11],  # number of times a score of 11 was recorded
        counts[12]   # number of times a score of 12 was recorded
    ]

    return row


def plot_errors(city_list: typ.List[str]):
    """
    For each city in city_list, plot the median & average of both
    the root mean squared error & mean squared error.
    """
    for city in city_list:  # for each city,

        # *** Plot the Median Mean Squared Error *** #
        in_mdn: str = str(pth.Path.cwd() / 'output' / f'{city}' / 'error_values' / 'mse_mdn.csv')
        out_mdn: str = str(pth.Path.cwd() / 'output' / f'{city}' / 'error_plots' / f'squared_mdn_err_plot.png')
        plot_values(f'{city} Mean Squared Median', out_mdn, in_mdn)

        # *** Plot the Average Mean Squared Error *** #
        in_avr: str = str(pth.Path.cwd() / 'output' / f'{city}' / 'error_values' / 'mse_avr.csv')
        out_avr: str = str(pth.Path.cwd() / 'output' / f'{city}' / 'error_plots' / f'squared_avr_err_plot.png')
        plot_values(f'{city} Mean Squared Average', out_avr, in_avr)

        # *** Plot the Median Root  Mean Squared Error *** #
        in_mdn: str = str(pth.Path.cwd() / 'output' / f'{city}' / 'error_values' / 'rmse_mdn.csv')
        out_mdn: str = str(pth.Path.cwd() / 'output' / f'{city}' / 'error_plots' / f'root_squared_mdn_err_plot.png')
        plot_values(f'{city} Root Mean Squared Median', out_mdn, in_mdn)

        # *** Plot the Average Root  Mean Squared Error *** #
        in_avr: str = str(pth.Path.cwd() / 'output' / f'{city}' / 'error_values' / 'rmse_avr.csv')
        out_avr: str = str(pth.Path.cwd() / 'output' / f'{city}' / 'error_plots' / f'root_squared_avr_err_plot.png')
        plot_values(f'{city} Root Mean Squared Average', out_avr, in_avr)

    return


if __name__ == "__main__":

    plot_errors(['kdfw', 'kcys', 'kroa'])

    # TODO: call plot_spikes() & plot_spikes_avr()

    pass
