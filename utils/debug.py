import functools
import time

__all__ = ['log_time']

def log_time(func):
    """
    A decorator to log the amount of time spent running code between specific lines.

    The `log_time` decorator wraps a function to provide a mechanism for logging the cumulative time
    spent executing specific lines of code. It can be used for performance profiling by recording how
    much time is spent between checkpoints within the function. This is done by passing a custom 
    `log_line_time` function into the wrapped function's keyword arguments, which can be called to 
    log time at specific lines.

    Parameters
    ----------
    func : callable
        The function to be wrapped and timed.

    Returns
    -------
    callable
        The wrapped function with timing capabilities.

    Examples
    --------
    To use the `log_time` decorator, simply apply it to a function. Within the function, use the 
    `log_line_time` keyword argument to log time at specific lines.

    >>> @log_time
    >>> def example_function(log_line_time=None):
    >>>     # Some code
    >>>     log_line_time(10)  # Log time at line 10
    >>>     # More code
    >>>     log_line_time(20)  # Log time at line 20
    >>> 
    >>> example_function()

    This will print the cumulative time spent between the logged lines when the function is called.

    Notes
    -----
    - The `log_line_time` function is injected into the wrapped function's keyword arguments.
    - It uses a nonlocal variable to track time across function calls.
    - The cumulative time spent at each line is printed out after the wrapped function completes.
    """
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Dictionary to store cumulative time spent on each line
        time_spent = {}
        last_time = time.time()
        
        # Custom function to log cumulative time
        def log_line_time(line_number):
            nonlocal last_time
            current_time = time.time()
            time_spent[line_number] = time_spent.get(line_number, 0) + (current_time - last_time)
            last_time = current_time

        # Add the log function to the arguments
        kwargs['log_line_time'] = log_line_time

        # Call the original function
        result = func(*args, **kwargs)

        # Print the cumulative time spent
        for line, duration in time_spent.items():
            print(f"Cumulative time spent at line {line}: {duration:.6f} seconds")

        return result

    return wrapper