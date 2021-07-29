
import sys
from formatting import printWarn
import pandas as pd
import pathlib as pth
import pickle
import typing as typ
import numpy as np
from create_plot import find_worst_permutation

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


def test_and_train(test, train, city, size):

    test_values_count: typ.Dict[int, int] = {
        0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0,
        7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0
    }

    train_values_count: typ.Dict[int, int] = {
        0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0,
        7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0
    }

    # * Convert the Data to Beaufort Scale & Count the Occurrences * #
    for i in train:
        train_values_count[beaufort_scale(i)] += 1

    for i in test:
        test_values_count[beaufort_scale(i)] += 1

    p = str(pth.Path.cwd() / 'output' / f'{city}' / f'{city}_hist_count.txt')
    with open(p, "a") as file:
        file.write('')
        file.write(f'Testing Dataset for {city} (size {size}):\n')
        file.write(f'Force 0:  {test_values_count[0]}\n')
        file.write(f'Force 1:  {test_values_count[1]}\n')
        file.write(f'Force 2:  {test_values_count[2]}\n')
        file.write(f'Force 3:  {test_values_count[3]}\n')
        file.write(f'Force 4:  {test_values_count[4]}\n')
        file.write(f'Force 5:  {test_values_count[5]}\n')
        file.write(f'Force 6:  {test_values_count[6]}\n')
        file.write(f'Force 7:  {test_values_count[7]}\n')
        file.write(f'Force 8:  {test_values_count[8]}\n')
        file.write(f'Force 9:  {test_values_count[9]}\n')
        file.write(f'Force 10: {test_values_count[10]}\n')
        file.write(f'Force 11: {test_values_count[11]}\n')
        file.write(f'Force 12: {test_values_count[12]}\n\n')

    p = str(pth.Path.cwd() / 'output' / f'{city}' / f'{city}_hist_count.txt')
    with open(p, "a") as file:
        file.write('')
        file.write(f'Training Dataset for {city} (size {size}):\n')
        file.write(f'Force 0:  {train_values_count[0]}\n')
        file.write(f'Force 1:  {train_values_count[1]}\n')
        file.write(f'Force 2:  {train_values_count[2]}\n')
        file.write(f'Force 3:  {train_values_count[3]}\n')
        file.write(f'Force 4:  {train_values_count[4]}\n')
        file.write(f'Force 5:  {train_values_count[5]}\n')
        file.write(f'Force 6:  {train_values_count[6]}\n')
        file.write(f'Force 7:  {train_values_count[7]}\n')
        file.write(f'Force 8:  {train_values_count[8]}\n')
        file.write(f'Force 9:  {train_values_count[9]}\n')
        file.write(f'Force 10: {train_values_count[10]}\n')
        file.write(f'Force 11: {train_values_count[11]}\n')
        file.write(f'Force 12: {train_values_count[12]}\n\n')

    return


def spike(city, size, perm):

    # * Get the Spike Data * #
    # read in the pickled dictionary
    jar: str = str(pth.Path.cwd() / 'output' / f'{city}' / 'train_set_record.p')
    training_sets: typ.Dict[int, typ.Dict] = pickle.load(open(jar, "rb"))

    spike_values_count: typ.Dict[int, int] = {
        0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0,
        7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0
    }

    for i in training_sets[size][perm]:
        spike_values_count[beaufort_scale(i)] += 1

    p = str(pth.Path.cwd() / 'output' / f'{city}' / f'{city}_hist_count.txt')
    with open(p, "a") as file:
        file.write('')
        file.write(f'Spike Data for {city} (size {size}):\n')
        file.write(f'Force 0:  {spike_values_count[0]}\n')
        file.write(f'Force 1:  {spike_values_count[1]}\n')
        file.write(f'Force 2:  {spike_values_count[2]}\n')
        file.write(f'Force 3:  {spike_values_count[3]}\n')
        file.write(f'Force 4:  {spike_values_count[4]}\n')
        file.write(f'Force 5:  {spike_values_count[5]}\n')
        file.write(f'Force 6:  {spike_values_count[6]}\n')
        file.write(f'Force 7:  {spike_values_count[7]}\n')
        file.write(f'Force 8:  {spike_values_count[8]}\n')
        file.write(f'Force 9:  {spike_values_count[9]}\n')
        file.write(f'Force 10: {spike_values_count[10]}\n')
        file.write(f'Force 11: {spike_values_count[11]}\n')
        file.write(f'Force 12: {spike_values_count[12]}\n\n')

    return


def complete_dataset(data, city, size):

    complete_values_count: typ.Dict[int, int] = {
        0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0,
        7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0
    }

    for i in data:
        complete_values_count[beaufort_scale(i)] += 1

    p = str(pth.Path.cwd() / 'output' / f'{city}' / f'{city}_hist_count.txt')
    with open(p, "a") as file:
        file.write('')
        file.write(f'Complete Dataset for {city} (size {size}):\n')
        file.write(f'Force 0:  {complete_values_count[0]}\n')
        file.write(f'Force 1:  {complete_values_count[1]}\n')
        file.write(f'Force 2:  {complete_values_count[2]}\n')
        file.write(f'Force 3:  {complete_values_count[3]}\n')
        file.write(f'Force 4:  {complete_values_count[4]}\n')
        file.write(f'Force 5:  {complete_values_count[5]}\n')
        file.write(f'Force 6:  {complete_values_count[6]}\n')
        file.write(f'Force 7:  {complete_values_count[7]}\n')
        file.write(f'Force 8:  {complete_values_count[8]}\n')
        file.write(f'Force 9:  {complete_values_count[9]}\n')
        file.write(f'Force 10: {complete_values_count[10]}\n')
        file.write(f'Force 11: {complete_values_count[11]}\n')
        file.write(f'Force 12: {complete_values_count[12]}\n\n')

    return


def get_hist_count(city, size, perm):

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
    train_data = data[: int(len(data) * .80) + 1]  # up to & including 80%
    test_data = data[int(len(data) * .80) + 1:]  # everything after 80%

    # * Call the Counters * #
    test_and_train(test_data, train_data, city, size)
    complete_dataset(data, city, size)
    spike(city, size, perm)


if __name__ == "__main__":

    p = find_worst_permutation('kdfw', 500)
    get_hist_count(city='kdfw', size=500, perm=p)

    p = find_worst_permutation('kcys', 500)
    get_hist_count(city='kcys', size=500, perm=p)

    p = find_worst_permutation('kroa', 500)
    get_hist_count(city='kroa', size=500, perm=p)

    p = find_worst_permutation('kroa', 1200)
    get_hist_count(city='kroa', size=1200, perm=p)
