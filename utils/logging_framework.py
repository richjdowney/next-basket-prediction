"""
Framework for generating log messages across the codebase
Logger can be used to generate different messages depending on the severity: `debug`, `info`, `warning`.  The
log messages contain a timestamp, the severity level and the path to the module where the log was generated
```
#    >>> from utils.log import log
#    >>> log.info("Information")
#    >>> log.debug("Debug code")
#    >>> log.warning("Warning message")
#    >>> log.error("Error message")
"""

import logging

# Change the logging level depending on whether the code is being run or debugged
# LOGGING_LEVEL = logging.DEBUG  # only showing debug messages
LOGGING_LEVEL = 20  # (logging.INFO)  # only showing info messages
# LOGGING_LEVEL = logging.WARNING  # only showing warning messages

FMT = "%(asctime)s: [%(levelname)s]: %(pathname)s:%(lineno)s: %(message)s"

DT_FMT = "%y-%m-%d %H:%M:%S"

logging.basicConfig(level=LOGGING_LEVEL, format=FMT, datefmt=DT_FMT)

log = logging


def format_title(msg: str) -> str:
    """Function to format a message title for logging
        Parameters
        ----------
        msg : str
            string to print to log
        Returns
        ----------
        result : str
            formatted log message
    """

    max_msg_len = 120
    assert len(msg) <= 120, "max message length for a title is {}".format(max_msg_len)
    buffer_len = 10
    ln_len = len(msg) + (buffer_len * 2) + 2

    ln = ln_len * "#"
    buffer = buffer_len * " "
    result = """
    {ln}
    #{buffer}{msg}{buffer}#
    {ln}
    """.format(
        ln=ln, buffer=buffer, msg=msg
    )

    return result
