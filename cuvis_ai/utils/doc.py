import warnings
import functools

# taken from https://stackoverflow.com/questions/2536307/decorators-in-the-python-standard-lib-deprecated-specifically

def deprecated(func):
    """This decorator indicates that the used function is deprecated and should no longer be used

    Parameters
    ----------
    func : Callable
        The function that is deprecated

    Returns
    -------
    Callable
        The annotated function
    """
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        warnings.warn("Call to deprecated function {}.".format(func.__name__),
                      category=DeprecationWarning,
                      stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)  # reset filter
        return func(*args, **kwargs)
    return new_func