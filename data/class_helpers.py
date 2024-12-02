import numpy as np

def calculate_average_movie(movie_vectors):
    # Initialize the output vector
    output_vector = []

    # Get all keys (attributes)
    keys = movie_vectors[0].keys()

    # Compute the average for each column
    for key in keys:
        column = [vector[key] if vector[key] is not None else np.nan for vector in movie_vectors]
        if isinstance(column[0], (int, float, np.float64)):
            average = np.nanmean(column)
            output_vector.append(average)
        elif isinstance(column[0], np.ndarray):
            average = np.nanmean(np.stack(column), axis=0)
            output_vector.append(average)
        else:
            output_vector.append(np.nan)  # Placeholder for non-numeric columns

    # TODO: validate output_vector

    return dict(zip(keys,output_vector))


# TODO: use new validate_attibute_vectors function more widely, also in main.py
# TODO: use the average movie of actors in main.py distance calculations