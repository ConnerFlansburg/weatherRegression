#!/bin/python3.9 python

import tkinter
from tkinter import filedialog
from tkfilebrowser import askopenfilename
import pandas as pd
from pprint import pprint
import sys
import pathlib


def main(filename) -> None:

    data_in = pd.read_csv(filename, low_memory=False)     # read in the csv
    
    names = list(data_in.columns)       # get the names of the columns
    
    num_cols = len(names)               # get the number of cols
    
    num_rows = data_in.shape[0]  # get the number of rows
    
    print(filename)
    print(f'\nNumber of columns {num_cols}')
    print(f'Number of rows {num_rows}\n')
    print(f'Attribute Names:')
    pprint(names)


if __name__ == '__main__':

    parent = tkinter.Tk()  # prevent root window caused by Tkinter
    parent.overrideredirect(1)  # Avoid it appearing and then disappearing quickly
    parent.withdraw()  # Hide the window

    f_name = filedialog.askopenfilename(parent=parent)

    # set standard output to a file
    out_name = f_name.replace('_processed', '')
    out_name = out_name.replace('.csv', '')
    out = pathlib.Path.cwd() / f'{out_name}_info.txt'  # create a Path object
    out = str(out)
    
    sys.stdout = open(out, 'w')
    
    main(f_name)
    
    sys.stdout.close()  # close the file stdout is writing to