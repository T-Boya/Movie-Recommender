from data.read_from_csv import load_movie_data
import numpy as np
from sentence_transformers import SentenceTransformer, SimilarityFunction, util

model = SentenceTransformer('all-mpnet-base-v2')
model.similarity_fn_name = SimilarityFunction.EUCLIDEAN

movies, actors, metrics = load_movie_data()

# Print the first 5 movies in the database
# for movie in list(movies.values())[:1]:
#     print(f"Movie: {movie.title}")
#     for attribute, vector in movie.attribute_vectors.items():
#         if vector is not None:
#             print(f"Attribute: {attribute}, Vector: {vector}, Value: {movie.get_normalized_vector(attribute)}")
#     print("\n")

def calculate_vector_magnitude(vector):
    """
    Calculate the magnitude of a vector.
    
    Parameters:
    vector (np.array): The vector for which to calculate the magnitude.
    
    Returns:
    float: The magnitude of the vector.
    """
    return np.linalg.norm(vector)

def normalize_matrix_columns(matrix):
    """
    Normalize the columns of a matrix such that the minimum value in each column is 0 and the maximum value is 1.
    
    Parameters:
    matrix (np.ndarray): The input matrix to normalize.
    
    Returns:
    np.ndarray: The normalized matrix.
    """
    if matrix.size > 0:
        min_values = np.min(matrix, axis=0)
        max_values = np.max(matrix, axis=0)
        range_values = max_values - min_values
        range_values[range_values == 0] = 1  # Avoid division by zero
        normalized_matrix = (matrix - min_values) / range_values
        return normalized_matrix
    return matrix

def calculate_distance(attribute,x,y):
    match attribute:
        case 'title':
            dist = -model.similarity(x, y)[0][0].item()
            dist = np.nan_to_num(dist, nan=0.0)  # Replace NaN with 0
            return dist * 0.1 # cosine similarity
        case 'genre':
            all_keys = x.union(y)
            all_dict = {key: 0 if key in x and key in y else 1 for key in all_keys}
            dist = calculate_vector_magnitude(list(all_dict.values())) # jaccard distance
            dist = np.nan_to_num(dist, nan=0.0)  # Replace NaN with 0
            return dist
        case 'plot':
            dist = -model.similarity(x, y)[0][0].item()
            dist = np.nan_to_num(dist, nan=0.0)  # Replace NaN with 0
            return dist # cosine similarity
        case 'actors':
            dist = []
            for key in x:
                if key in y:
                    np.append(dist, calculate_distance(key, x[key], y[key]))
                else:
                    np.append(dist, 0.0)
            # TODO: do we normalize these two before summing?
            dist = calculate_vector_magnitude(dist) # euclidean distance
            dist = np.nan_to_num(dist, nan=0.0)  # Replace NaN with 0
            return dist
        case 'director':
            dist = []
            for key in x:
                if key in y:
                    np.append(dist, calculate_distance(key, x[key], y[key]))
                else:
                    np.append(dist, 0.0)
            # TODO: do we normalize these two before summing?
            dist = calculate_vector_magnitude(dist) # euclidean distance
            dist = np.nan_to_num(dist, nan=0.0)  # Replace NaN with 0
            return dist
        case _:
            dist = calculate_vector_magnitude(x - y) # euclidean distance
            dist = np.nan_to_num(dist, nan=0.0)  # Replace NaN with 0
            return dist

def calculate_distances(movie_id, movie_database):
    """
    Calculate the Euclidean distance from the given movie to all other movies in the database.
    
    Parameters:
    movie_id (str): The ID of the movie to compare.
    movie_database (MovieDatabase): The database containing all movies.
    
    Returns:
    dict: A dictionary where keys are the Euclidean distances and values are lists of movie IDs.
    """
    distances = []
    distance_map = {}
    target_movie = movie_database[movie_id]
    
    if not target_movie:
        raise ValueError(f"Movie with ID {movie_id} not found in the database.")
    
    # Fetch all keys
    valid_keys = [key for key, value in target_movie.attribute_vectors.items() if \
                  (value is not None and key in target_movie.PRECALCULATED_VECTORS) or \
                    (value is None and key in target_movie.ON_THE_FLY_VECTORS)]

    # Get normalized vectors for each valid key
    target_movie.calculate_on_the_fly_vectors(movie_database, actors)
    target_vectors = {key: target_movie.get_normalized_vector(key) for key in valid_keys}
    
    for other_movie in movie_database.values():
        if other_movie.id == movie_id:
            continue
        
        other_movie.calculate_on_the_fly_vectors(movie_database, actors)
        other_vectors = {key: other_movie.get_normalized_vector(key) for key in valid_keys if other_movie.attribute_vectors[key] is not None}
        distance = []
        
        for key in target_vectors:
            if key in other_vectors:
                # not all distances are created equal, have a vector of distances and weight distance by importance before summing
                # importance is defined by attribute
                distance = np.append(distance, calculate_distance(key, target_vectors[key], other_vectors[key]))
            else:
                # if the other movie doesn't have the same attribute, add the minimum distance
                distance = np.append(distance, 0.0)

        distances.append(distance)
        distance_map[len(distances) - 1] = other_movie.id
        
    distances = np.array(distances)  # Convert distances to a NumPy array

    distances = normalize_matrix_columns(distances)  # Normalize distances matrix by column such that min and max are zero and one

    # use distance_map to map distances back to movie ids
    return {distance_map[idx]: calculate_vector_magnitude(distance) for idx, distance in enumerate(distances)}

# Calculate the Euclidean distances from the first movie to all other movies
movie_id = list(movies.keys())[0]
distances = calculate_distances(movie_id, movies)

movie_display_count = 10
print(f"Found {movie_display_count} movies similar to movie: {movies[movie_id].title}")
# Get the 5 movies with the smallest distance
closest_movies = sorted(distances.items(), key=lambda item: item[1])[:movie_display_count]
for movie_id, distance in closest_movies:
    print(f"Movie: {movies[movie_id].title}, Distance: {distance}")