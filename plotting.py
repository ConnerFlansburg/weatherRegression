import pandas as pd
import matplotlib.pyplot as plt


def grid_plot(df: pd.DataFrame, file: str):
    """
    scatter_plot creates a scatter plot of the dataframe using
    Pandas libraries.
    """

    # create the figure that will hold all 4 plots
    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True)
    axes[0].ticklabel_format(style='sci', useMathText=True)
    axes[1].ticklabel_format(style='sci', useMathText=True)
    axes[2].ticklabel_format(style='sci', useMathText=True)

    # set the values for the 'Training Size' axis
    axes[0].set_ylabel('Error Score')  # label the y-axis
    axes[1].set_ylabel('Error Score')  # label the y-axis

    # axes[2].set_xticks(BUCKETS_LABEL)  # make a tick for every bucket
    axes[2].set_ylabel('Error Score')  # label the y-axis

    # rotate = 45  # how many degrees the x-axis labels should be rotated
    rotate = 90  # how many degrees the x-axis labels should be rotated
    # create the plot & place it in the upper left corner
    df.plot(ax=axes[0],
            kind='line',
            x='Training Size',
            y='Mean Absolute Error',
            color='blue',
            style='--',      # the line style
            x_compat=True,
            rot=rotate,      # how many degrees to rotate the x-axis labels
            # use_index=True,
            grid=True,
            legend=True,
            # marker='o',    # what type of data markers to use?
            # mfc='black'    # what color should they be?
            )
    # axes[0].set_title('Mean Absolute Error')

    # create the plot & place it in the upper right corner
    df.plot(ax=axes[1],
            kind='line',
            x='Training Size',
            y='Mean Squared Error',
            color='red',
            style='-.',    # the line style
            rot=rotate,    # how many degrees to rotate the x-axis labels
            x_compat=True,
            # use_index=True,
            grid=True,
            legend=True,
            # marker='o',  # what type of data markers to use?
            # mfc='black'  # what color should they be?
            )
    # axes[1].set_title('Mean Squared Error')

    # create the plot & place it in the lower left corner
    df.plot(ax=axes[2],
            kind='line',
            x='Training Size',
            y='Mean Signed Error',
            color='green',
            style='-',     # the line style
            rot=rotate,    # how many degrees to rotate the x-axis labels
            x_compat=True,
            # use_index=True,
            grid=True,
            legend=True,
            # marker='o',  # what type of data markers to use?
            # mfc='black'  # what color should they be?
            )
    # axes[2].set_title('Mean Signed Error')

    fig.tight_layout()

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
    # set the values for the 'Training Size' axis
    # ax.set_xticks(BUCKETS_LABEL)  # make a tick for every bucket
    ax.set_ylabel('Error Score')  # label the y-axis
    ax.invert_xaxis()
    ax.ticklabel_format(style='sci', useMathText=True)  # format the sci notation
    # set the format of sci notation
    ax.ticklabel_format(style='sci', useMathText=True)

    # plot the mean absolute error
    df.plot(ax=ax,
            kind='line',
            x='Training Size',
            y='Mean Absolute Error',
            color='blue',
            style='--',  # the line style
            x_compat=True,
            use_index=True,
            grid=True,
            legend=True,
            marker='o',  # what type of data markers to use?
            mfc='black'  # what color should they be?
            )

    # plot the mean squared error
    df.plot(ax=ax,
            kind='line',
            x='Training Size',
            y='Mean Squared Error',
            color='red',
            style='-.',  # the line style
            x_compat=True,
            use_index=True,
            grid=True,
            legend=True,
            marker='o',  # what type of data markers to use?
            mfc='black'    # what color should they be?
            )

    # plot the mean signed error
    df.plot(ax=ax,
            kind='line',
            x='Training Size',
            y='Mean Signed Error',
            style=':',  # the line style
            x_compat=True,
            color='green',
            use_index=True,
            grid=True,
            legend=True,
            marker='o',  # what type of data markers to use?
            mfc='black'    # what color should they be?
            )

    # save the plot to the provided file path
    plt.savefig(file)
    # show the plot
    plt.show()

    return


def smooth_plot(df: pd.DataFrame, file: str, title: str):
    """
    error_plot creates a line plot of the report dataframe using
    Pandas libraries, & plots all 3 error scores on the same figure.
    """

    # * Create a New Dataframe with the Desired Stats * #
    # sort the data frame by training size so the new dataframe is in order

    sorted_cols = list(df.columns)
    # in the original dataframe each col is a all the values for a set number of buckets/training size,
    # & each row is a single example/instance/smooth iteration.
    # Now we want to transform the dataframe so that each col contains the min/max/median over every
    # smooth iter, & each row is a set number of buckets/training size
    metrics_df: pd.DataFrame = pd.DataFrame({
        'Size': sorted_cols,
        'Min': df.min(),       # get the min values for each col & make it a col in the new df
        'Max': df.max(),       # get the max values for each col & make it a col in the new df
        'Median': df.median()  # get the median values for each col & make it a col in the new df
    })

    # * Plot the New Dataframe * #
    # create the figure that will hold all 4 plots
    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True)

    # set the plots title
    fig.suptitle(title)

    # set the size of the figure (width, height)
    fig.set_size_inches(14, 9)

    # set the format of any sci notation
    axes[0].ticklabel_format(style='sci', useMathText=True)
    axes[1].ticklabel_format(style='sci', useMathText=True)
    axes[2].ticklabel_format(style='sci', useMathText=True)

    # label the y-axis
    axes[0].set_ylabel('Min Error')
    axes[1].set_ylabel('Median Error')
    axes[2].set_ylabel('Max Error')

    # label the x-axis
    axes[2].set_xlabel('Training Size')

    # add an x-axis tick for every training size
    axes[2].set_xticks(sorted_cols)

    # rotate = 45  # how many degrees the x-axis labels should be rotated
    rotate = 90  # how many degrees the x-axis labels should be rotated

    # create the plot & place it in the upper left corner (min vs size)
    metrics_df.plot(ax=axes[0],
                    kind='line',
                    x='Size',
                    y='Min',
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
    # axes[0].set_title('Mean Absolute Error')

    # create the plot & place it in the upper right corner (median vs size)
    metrics_df.plot(ax=axes[1],
                    kind='line',
                    x='Size',
                    y='Median',
                    color='red',
                    style='-.',  # the line style
                    rot=rotate,  # how many degrees to rotate the x-axis labels
                    x_compat=True,
                    # use_index=True,
                    grid=True,
                    legend=True,
                    # marker='o',  # what type of data markers to use?
                    # mfc='black'  # what color should they be?
                    )
    # axes[1].set_title('Mean Squared Error')

    # create the plot & place it in the lower left corner (max vs size)
    metrics_df.plot(ax=axes[2],
                    kind='line',
                    x='Size',
                    y='Median',
                    color='green',
                    style='-',  # the line style
                    rot=rotate,  # how many degrees to rotate the x-axis labels
                    x_compat=True,
                    # use_index=True,
                    grid=True,
                    legend=True,
                    # marker='o',  # what type of data markers to use?
                    # mfc='black'  # what color should they be?
                    )
    # axes[2].set_title('Mean Signed Error')

    fig.tight_layout()

    # save the plot to the provided file path
    plt.savefig(file)
    # show the plot
    plt.show()
    return
