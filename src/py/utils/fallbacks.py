def unable2playsound(filename):
    print(f"Unable to play sound: {filename}")
    return

class MyPlaysoundException(Exception):
    pass

def deprecated(reason=None):
    """
    Decorator to mark a function as deprecated.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            if reason:
                print(f"Warning: {func.__name__} is deprecated. Reason: {reason}")
            else:
                print(f"Warning: {func.__name__} is deprecated.")
            return func(*args, **kwargs)
        return wrapper
    return decorator
