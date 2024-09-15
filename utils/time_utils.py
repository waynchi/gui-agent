import time
from functools import wraps


def time_function(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Capture the start time
        print(f"Starting '{func.__name__}'...")

        result = func(*args, **kwargs)  # Execute the function

        end_time = time.time()  # Capture the end time
        duration = end_time - start_time
        print(f"Finished '{func.__name__}'. Duration: {duration:.3f} seconds.")

        return result

    return wrapper
