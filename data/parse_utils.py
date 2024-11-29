import numpy as np


def parse_actors(actors: str) -> list:
    """Parse the actors string into a list of actors."""
    return [x.strip() for x in actors.split(',')]

def convert_to_type(value, target_type):
    try:
        if value is None or np.isnan(value):
            return value
        return target_type(value)
    except (ValueError, TypeError):
        print(f"Error converting {value} to {target_type}")
        return None