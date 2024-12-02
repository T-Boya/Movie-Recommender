import numpy as np

def isNoneType(value) -> bool:
    """Check if the value is of NoneType or NaN."""
    return value is None or (isinstance(value, float) and np.isnan(value))

def parse_actors(actors: str) -> list:
    """Parse the actors string into a list of actors."""
    return [x.strip() for x in actors.split(',')]

def convert_to_type(value, target_type):
    try:
        if isNoneType(value):
            return None
        return target_type(value)
    except (ValueError, TypeError):
        print(f"Error converting {value} to {target_type}")
        return None