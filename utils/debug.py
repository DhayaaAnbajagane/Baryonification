import functools
import time

def log_time(func):
    '''
    Simple function that allows logging amount of time spent 
    running code between line A and line B
    '''
    
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