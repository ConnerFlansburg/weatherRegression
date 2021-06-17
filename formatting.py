

import sys

HDR = '*' * 6
SUCCESS = u' \u2713\n'+'\033[0m'       # print the checkmark & reset text color
OVERWRITE = '\r' + '\033[32;1m' + HDR  # overwrite previous text & set the text color to green
NO_OVERWRITE = '\033[32;1m' + HDR      # NO_OVERWRITE colors lines green that don't use overwrite
SYSOUT = sys.stdout                    # SYSOUT set the standard out for the program to the console


def printWarn(message: str):
    """
    printWarn is used for coloring warnings yellow.

    :param message: The message to be colored.
    :type message: str

    :rtype: str
    """
    print(f"\033[33m{message}\033[00m")


def printSuccess(message: str):
    """ Colors a string green & returns it."""
    print(f"\033[32;1m{message}\033[00m")


def success(message: str) -> str:
    """ Colors a string green & returns it."""
    return f"\033[32;1m{message}\033[00m"


def printError(message: str):
    """
    printError is used for coloring error messages red.

    :param message: The message to be printed.
    :type message: str

    :return: printError does not return, but rather prints to the console.
    :rtype: None
    """
    print(f"\033[91;1m{message}\033[00m")


def printPercentage(decimalScore: float) -> str:
    """ Prints the passed decimal as a percentage & colors it based on it's value. """

    if decimalScore > 0.75:  # > 75 print in green
        return f'\033[32;1m{round(decimalScore*100,2)}%\033[00m'

    elif 0.45 < decimalScore < 0.75:  # > 45 and < 75 print yellow
        return f'\033[33;1m{round(decimalScore*100,2)}%\033[00m'

    elif decimalScore < 0.45:  # < 45 print in red
        return f'\033[91;1m{round(decimalScore*100,2)}%\033[00m'

    else:  # don't add color, but print accuracy
        return f'{decimalScore}'


def colorDecimal(decimalScore: float) -> str:
    """ Colors a decimal based on it's value & then returns it"""

    if decimalScore > 0.75:  # > 75 print in green
        return f'\033[32;1m{decimalScore}\033[00m'

    elif 0.45 < decimalScore < 0.75:  # > 45 and < 75 print yellow
        return f'\033[33;1m{decimalScore}\033[00m'

    elif decimalScore < 0.45:  # < 45 print in red
        return f'\033[91;1m{decimalScore}\033[00m'

    else:  # don't add color, but print accuracy
        return f'{decimalScore}'


def banner(msg: str) -> str:
    """ Centers a string & returns it """
    return msg.center(50, '*')


def printBanner(msg: str):
    """ Like banner, but prints rather than return """
    SYSOUT.write(msg.center(50, '*'))


