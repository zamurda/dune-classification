"""
Custom errors are defined here to help with scripting using dctools.
"""

import functools


class FeatureExtractionError(Exception):
    """
    Intended as an aggregator error which can be raised after catching Value,Type and Index errors.
    Recommended to catch this error and set the observation to a negative, non-zero number for easier data filtering.
    """
    pass


def exception_aggregator(func):
    """
    Wrapper function for aggregating errors and raising FeatureExtractionError.
    Quite hackish as it raises an exception whilst handling other exceptions, but works.
    Use wisely!
    """
    @functools.wraps(func)
    def aggregator(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (RuntimeError, IndexError, ValueError, TypeError):
            raise FeatureExtractionError
    return aggregator
