
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pathlib as pth
import numpy as np
import pickle
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

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
    plt.rc('font', size=15)         # ! testing font size changes
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #

    ax = plt.gca()                  # get the current ax
    ax.set_ylabel('Error Score')    # label the y-axis
    ax.set_xlabel('Training Size')  # label the x-axis

    ax.set_xlim([100, 2513])        # set the range of the x ticks

    fig = plt.gcf()                 # get the current figure
    fig.set_size_inches(14, 9)      # set the size of the image (width, height)

    # how many degrees the x-axis labels should be rotated
    # rotate = 0
    rotate = 45
    # rotate = 90

    g = averages.plot(ax=ax,           # set the axis to the current axis
                      kind='line',
                      x='Size',
                      y='Error',
                      color='blue',
                      style='--',  # the line style
                      x_compat=True,
                      rot=rotate,  # how many degrees to rotate the x-axis labels
                      use_index=True,
                      grid=True,
                      legend=True,
                      # marker='o',    # what type of data markers to use?
                      # mfc='black'    # what color should they be?
                      )

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
    # this sets the x-axis ticks
    g.set_xticks(averages['Size'])

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #

    fig.tight_layout()  # tighten the layout
    plt.savefig(fl)     # save the plot
    # plt.show()          # show the plot
    plt.clf()
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


# def plot_spikes(city: str, start_index: int, stop_index: int, perm: int):
#     """
#     This will create a histogram for the index range pass using
#     only the passed permutation. Index values are both inclusive.
#     """
#
#     # * Read in the Permutated Wind Speeds * #
#     path: str = str(pth.Path.cwd() / 'output' / f'{city}' / 'permutation' / f'wind_speed_{perm}.csv')
#     permutation: pd.DataFrame = pd.read_csv(path, index_col=0)
#
#     # * Get the Wind Speeds of Interest in the Beaufort Scale * #
#     wind_speeds = [beaufort_scale(i) for i in permutation['Wind'][start_index:stop_index+1]]
#
#     # * Plot the Beaufort Scale Values using a Histogram * #
#     sns.set(
#         font_scale=2,                    # resizes the font (of everything)
#         rc={"figure.figsize": (14, 9)},  # set the size of the plot
#         style="darkgrid"                 # sets the plots style
#     )
#     # Set the title (add 1 to make the titles more human readable)
#     plt.title(f'{city} {start_index+1}-{stop_index+1} Spike Distribution')
#
#     bins = 50  # the number of bins to use
#     sns.histplot(            # make the histogram
#         data=wind_speeds,    # the dataframe
#         # x='Wind',            # the column from the dataframe
#         bins=bins,
#         color='blue',
#         alpha=0.5,           # how transparent the bar should be (1=solid, 0=invisible)
#         # kde=True,          # draw a density line
#         stat='probability',  # make the histogram show percentages
#         discrete=True,
#     )
#
#     # * Display the Plot & Save it * #
#     save = str(pth.Path.cwd() / 'output' / f'{city}' / f'{city}_spike_distribution.png')
#     ax = plt.gca()                         # get the current ax
#     ax.set_xlabel('Beaufort Scale Value')  # label the x-axis
#     plt.savefig(save)                      # save the plot
#     # plt.show()                             # display the plot
#     return


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #

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
        plt.clf()

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
        plot_values(f'{city} Mean Squared Error Median', out_mdn, in_mdn)

        # *** Plot the Average Mean Squared Error *** #
        in_avr: str = str(pth.Path.cwd() / 'output' / f'{city}' / 'error_values' / 'mse_avr.csv')
        out_avr: str = str(pth.Path.cwd() / 'output' / f'{city}' / 'error_plots' / f'squared_avr_err_plot.png')
        plot_values(f'{city} Mean Squared Error Average', out_avr, in_avr)

        # *** Plot the Median Root Mean Squared Error *** #
        in_mdn: str = str(pth.Path.cwd() / 'output' / f'{city}' / 'error_values' / 'rmse_mdn.csv')
        out_mdn: str = str(pth.Path.cwd() / 'output' / f'{city}' / 'error_plots' / f'root_squared_mdn_err_plot.png')
        plot_values(f'{city} Root Mean Squared Error Median', out_mdn, in_mdn)

        # *** Plot the Average Root Mean Squared Error *** #
        in_avr: str = str(pth.Path.cwd() / 'output' / f'{city}' / 'error_values' / 'rmse_avr.csv')
        out_avr: str = str(pth.Path.cwd() / 'output' / f'{city}' / 'error_plots' / f'root_squared_avr_err_plot.png')
        plot_values(f'{city} Root Mean Squared Error Average', out_avr, in_avr)

    return


def find_worst_permutation(city: str, size: int):

    # * Read in the Pickled Record * #
    jar: str = str(pth.Path.cwd() / 'output' / f'{city}' / 'spike_record.p')
    spikes: typ.Dict[int, typ.Dict[int, float]] = pickle.load(open(jar, "rb"))

    # * For the Passed Size, Find the Worst Permutation * #
    # spikes[size] will contain a dictionary that holds the error score values
    # for each smooth iteration, and the inner dict is keyed by the smooth_iter
    # (which is equal to permutation value). So find the key with the worst
    # error score.
    permutation = max(spikes[size], key=spikes[size].get)
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
    # print(f'The Worst Permutation Found is: {permutation}')
    # pprint.pprint(spikes)
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
    return permutation


def plot_spikes(city: str, size: int, perm: int):
    """
    This will create a histogram for the index range pass using
    only the passed permutation. Index values are both inclusive.
    """

    # read in the pickled dictionary
    jar: str = str(pth.Path.cwd() / 'output' / f'{city}' / 'train_set_record.p')
    training_sets: typ.Dict[int, typ.Dict] = pickle.load(open(jar, "rb"))

    # * Get the Wind Speeds of Interest in the Beaufort Scale * #
    # pprint.pprint(training_sets[size][perm])
    wind_speeds = [beaufort_scale(i) for i in training_sets[size][perm]]

    # * Plot the Beaufort Scale Values using a Histogram * #
    sns.set(
        font_scale=2,  # resizes the font (of everything)
        rc={"figure.figsize": (14, 9)},  # set the size of the plot
        style="darkgrid"  # sets the plots style
    )

    bins = 50  # the number of bins to use
    hst = sns.histplot(      # make the histogram
        data=wind_speeds,    # the dataframe
        # x='Wind',          # the column from the dataframe
        bins=bins,
        color='blue',
        alpha=0.5,           # how transparent the bar should be (1=solid, 0=invisible)
        # kde=True,          # draw a density line
        # stat='probability',  # make the histogram show percentages
        discrete=True,
    )
    hst.set(
        title=f'{city} {size} Spike Distribution',  # set the title
        xlabel='Beaufort Scale Value'

    )

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
    hst.set_xticks(range(13))  # set the x axis labels
    hst.set_xticklabels([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #

    # * Display the Plot & Save it * #
    save = str(pth.Path.cwd() / 'output' / f'{city}' / f'{city}_{size}_spike_distribution.png')
    plt.savefig(save)  # save the plot
    # plt.show()         # display the plot
    plt.clf()
    return


def plot_spike_over_training(city: str, size: int, perm: int, annotate: bool):

    # * Read in the File * #
    fl = str(pth.Path.cwd() / 'data' / f'{city}_processed_data.csv')
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

    # * Get the Spike Data * #
    # read in the pickled dictionary
    jar: str = str(pth.Path.cwd() / 'output' / f'{city}' / 'train_set_record.p')
    training_sets: typ.Dict[int, typ.Dict] = pickle.load(open(jar, "rb"))

    # * Get the Wind Speeds of Interest in the Beaufort Scale * #
    wind_speeds = [beaufort_scale(i) for i in training_sets[size][perm]]

    # * Plot the Data * #
    sns.set(font_scale=2)   # resizes the font (of everything)
    sns.set_style("darkgrid")  # set the style
    # set the figures properties
    plt.figure(figsize=(14, 9), constrained_layout=True)  # set the size of the figure (width, height)
    plt.title(f'{city} {size} spike distribution vs training dataset')  # set the plots title

    bins = 50  # the number of bins to use
    # Spike Dataset
    spk = sns.histplot(
        data=wind_speeds,  # the dataframe
        # x='Testing Data',      # the column from the dataframe
        label='Spike',
        bins=bins,
        color='blue',
        alpha=0.5,             # how transparent the bar should be (1=solid, 0=invisible)
        # kde=True,            # draw a density line
        stat='probability',    # make the histogram show percentages
        discrete=True,
    )

    # Complete Training Dataset
    trn = sns.histplot(
        data=beaufort_values,  # the dataframe
        x='Training Data',     # the column from the dataframe
        label='Training Data',
        bins=bins,
        color='yellow',
        alpha=0.5,             # how transparent the bar should be (1=solid, 0=invisible)
        # kde=True,            # draw a density line
        stat='probability',    # make the histogram show percentages
        discrete=True,
    )

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
    spk.set_xticks(range(13))  # set the x axis labels
    spk.set_xticklabels([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
    if annotate:
        s = 0
        for spam in spk.patches:
            s += spam.get_height()
        for spam in spk.patches:
            spk.annotate(f'{round(spam.get_height(), 3)}',
                         (spam.get_x() + spam.get_width() / 2., spam.get_height()),
                         ha='center', va='center',
                         size=10,
                         xytext=(0, -12),
                         textcoords='offset points')
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #

    # * Display the Plot & Save it * #
    ax = plt.gca()  # get the current ax
    ax.set_xlabel('Beaufort Scale Value')  # label the x-axis
    plt.legend()        # show the legend
    save = str(pth.Path.cwd() / 'output' / f'{city}' / f'{city}_{size}_spike_vs_training.png')
    plt.savefig(save)   # save the plot
    # plt.show()        # display the plot
    plt.clf()
    return


def plot_spike_over_testing(city: str, size: int, perm: int, annotate: bool):
    # * Read in the File * #
    fl = str(pth.Path.cwd() / 'data' / f'{city}_processed_data.csv')
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

    # ! get the size of the test set (len starts at 1 NOT 0)
    print(f'Length of the Testing Set for {city} is {len(test_data)}')

    # * Convert the Data to Beaufort Scale * #
    beaufort_values = pd.DataFrame(  # convert the values to the beaufort scale & put in a dataframe
        list(zip([beaufort_scale(i) for i in train_data], [beaufort_scale(i) for i in test_data])),
        columns=['Training Data', 'Testing Data']
    )

    # * Get the Spike Data * #
    # read in the pickled dictionary
    jar: str = str(pth.Path.cwd() / 'output' / f'{city}' / 'train_set_record.p')
    training_sets: typ.Dict[int, typ.Dict] = pickle.load(open(jar, "rb"))

    # * Get the Wind Speeds of Interest in the Beaufort Scale * #
    wind_speeds = [beaufort_scale(i) for i in training_sets[size][perm]]

    # * Plot the Data * #
    sns.set(font_scale=2)   # resizes the font (of everything)
    sns.set_style("darkgrid")  # set the style
    # set the figures properties
    plt.figure(figsize=(14, 9), constrained_layout=True)  # set the size of the figure (width, height)
    plt.title(f'{city} {size} spike distribution vs testing dataset')  # set the plots title

    bins = 50  # the number of bins to use
    # Spike Dataset
    spk = sns.histplot(
        data=wind_speeds,  # the dataframe
        # x='Testing Data',      # the column from the dataframe
        label='Spike',
        bins=bins,
        color='blue',
        alpha=0.5,             # how transparent the bar should be (1=solid, 0=invisible)
        # kde=True,            # draw a density line
        stat='probability',    # make the histogram show percentages
        discrete=True,
    )

    # Complete Testing Dataset
    tst = sns.histplot(
        data=beaufort_values,  # the data is a list of speeds convert to b scale
        x='Testing Data',     # the column from the dataframe
        label='Testing Data',
        bins=bins,
        color='yellow',
        alpha=0.5,             # how transparent the bar should be (1=solid, 0=invisible)
        # kde=True,            # draw a density line
        stat='probability',    # make the histogram show percentages
        discrete=True,
    )

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
    spk.set_xticks(range(13))  # set the x axis labels
    spk.set_xticklabels([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
    if annotate:
        s = 0
        for spam in spk.patches:
            s += spam.get_height()
        for spam in spk.patches:
            spk.annotate(f'{round(spam.get_height(), 3)}',
                         (spam.get_x() + spam.get_width() / 2., spam.get_height()),
                         ha='center', va='center',
                         size=10,
                         xytext=(0, -12),
                         textcoords='offset points')
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #

    # * Display the Plot & Save it * #
    ax = plt.gca()  # get the current ax
    ax.set_xlabel('Beaufort Scale Value')  # label the x-axis
    plt.legend()        # show the legend
    save = str(pth.Path.cwd() / 'output' / f'{city}' / f'{city}_{size}_spike_vs_testing.png')
    plt.savefig(save)   # save the plot
    # plt.show()        # display the plot
    plt.clf()
    return


def plot_spike_over_complete(city: str, size: int, perm: int, annotate: bool):
    # * Read in the File * #
    fl = str(pth.Path.cwd() / 'data' / f'{city}_processed_data.csv')
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

    # * Get the Spike Data * #
    # read in the pickled dictionary
    jar: str = str(pth.Path.cwd() / 'output' / f'{city}' / 'train_set_record.p')
    training_sets: typ.Dict[int, typ.Dict] = pickle.load(open(jar, "rb"))

    # * Get the Wind Speeds of Interest in the Beaufort Scale * #
    wind_speeds = [beaufort_scale(i) for i in training_sets[size][perm]]

    # * Plot the Data * #
    sns.set(font_scale=2)   # resizes the font (of everything)
    sns.set_style("darkgrid")  # set the style
    # set the figures properties
    plt.figure(figsize=(14, 9), constrained_layout=True)  # set the size of the figure (width, height)
    plt.title(f'{city} {size} spike distribution vs complete dataset')  # set the plots title

    bins = 50  # the number of bins to use
    # Spike Dataset
    spk = sns.histplot(
        data=wind_speeds,  # the dataframe
        # x='Testing Data',      # the column from the dataframe
        label='Spike',
        bins=bins,
        color='blue',
        alpha=0.5,             # how transparent the bar should be (1=solid, 0=invisible)
        # kde=True,            # draw a density line
        stat='probability',    # make the histogram show percentages
        discrete=True,
    )

    # Complete Dataset
    cmp = sns.histplot(
        data=[beaufort_scale(i) for i in data],  # the data is a list of speeds convert to b scale
        # x='Training Data',     # the column from the dataframe
        label='Complete Dataset',
        bins=bins,
        color='yellow',
        alpha=0.5,             # how transparent the bar should be (1=solid, 0=invisible)
        # kde=True,            # draw a density line
        stat='probability',    # make the histogram show percentages
        discrete=True,
    )

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
    spk.set_xticks(range(13))  # set the x axis labels
    spk.set_xticklabels([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
    if annotate:
        s = 0
        for spam in spk.patches:
            s += spam.get_height()
        for spam in spk.patches:
            spk.annotate(f'{round(spam.get_height(), 3)}',
                         (spam.get_x() + spam.get_width() / 2., spam.get_height()),
                         ha='center', va='center',
                         size=10,
                         xytext=(0, -12),
                         textcoords='offset points')
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #

    # * Display the Plot & Save it * #
    ax = plt.gca()  # get the current ax
    ax.set_xlabel('Beaufort Scale Value')  # label the x-axis
    plt.legend()        # show the legend
    save = str(pth.Path.cwd() / 'output' / f'{city}' / f'{city}_{size}_spike_vs_complete.png')
    plt.savefig(save)   # save the plot
    # plt.show()        # display the plot
    plt.clf()
    return
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #


if __name__ == "__main__":

    # * Plot the Confusion Matrix * #
    plot_errors(['kdfw', 'kcys', 'kroa'])

    # * Plot the Spike for the Worst Permutation * #
    p = find_worst_permutation('kdfw', 500)
    plot_spikes('kdfw', 500, p)
    plot_spike_over_testing(city='kdfw', size=500, perm=p, annotate=False)
    plot_spike_over_training(city='kdfw', size=500, perm=p, annotate=False)
    plot_spike_over_complete(city='kdfw', size=500, perm=p, annotate=False)

    p = find_worst_permutation('kcys', 500)
    plot_spikes(city='kcys', size=500, perm=p)
    plot_spike_over_testing(city='kcys', size=500, perm=p, annotate=False)
    plot_spike_over_training(city='kcys', size=500, perm=p, annotate=False)
    plot_spike_over_complete(city='kcys', size=500, perm=p, annotate=False)

    p = find_worst_permutation('kroa', 500)
    plot_spikes(city='kroa', size=500, perm=p)
    plot_spike_over_testing(city='kroa', size=500, perm=p, annotate=False)
    plot_spike_over_training(city='kroa', size=500, perm=p, annotate=False)
    plot_spike_over_complete(city='kroa', size=500, perm=p, annotate=False)

    p = find_worst_permutation('kroa', 1200)
    plot_spikes(city='kroa', size=1200, perm=p)
    plot_spike_over_testing(city='kroa', size=1200, perm=p, annotate=False)
    plot_spike_over_training(city='kroa', size=1200, perm=p, annotate=False)
    plot_spike_over_complete(city='kroa', size=1200, perm=p, annotate=False)


