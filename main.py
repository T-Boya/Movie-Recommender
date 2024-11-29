from data.read_from_csv import load_movie_data
import numpy as np
from collections import defaultdict
from sentence_transformers import SentenceTransformer, SimilarityFunction, util

model = SentenceTransformer('all-mpnet-base-v2')
model.similarity_fn_name = SimilarityFunction.EUCLIDEAN

movies, actors, metrics = load_movie_data()

# Print the first 5 movies in the database
for movie in list(movies.values())[:1]:
    print(f"Movie: {movie.title}")
    for attribute, vector in movie.attribute_vectors.items():
        if vector is not None:
            print(f"Attribute: {attribute}, Vector: {vector}, Value: {movie.get_normalized_vector(attribute)}")
    print("\n")

def calculate_distance(attribute,x,y):
    match attribute:
        case 'title':
            dist = -model.similarity(x, y)[0][0].item()
            return dist * 0.1 # cosine similarity
        case 'genre':
            dist = -model.similarity(x, y)[0][0].item()
            return dist # cosine similarity
        case 'plot':
            dist = -model.similarity(x, y)[0][0].item()
            return dist # cosine similarity
        case _:
            return np.linalg.norm(x - y) # euclidean distance

def calculate_distances(movie_id, movie_database):
    """
    Calculate the Euclidean distance from the given movie to all other movies in the database.
    
    Parameters:
    movie_id (str): The ID of the movie to compare.
    movie_database (MovieDatabase): The database containing all movies.
    
    Returns:
    dict: A dictionary where keys are the Euclidean distances and values are lists of movie IDs.
    """
    distances = defaultdict(list)
    target_movie = movie_database[movie_id]
    
    if not target_movie:
        raise ValueError(f"Movie with ID {movie_id} not found in the database.")
    
    # Fetch all keys that don't have None as a value
    valid_keys = [key for key, value in target_movie.attribute_vectors.items() if value is not None]

    # Get normalized vectors for each valid key
    target_vectors = {key: target_movie.get_normalized_vector(key) for key in valid_keys}
    
    for other_movie in movie_database.values():
        if other_movie.id == movie_id:
            continue
        
        other_vectors = {key: other_movie.get_normalized_vector(key) for key in valid_keys if other_movie.attribute_vectors[key] is not None}
        distance = 0
        
        for key in target_vectors:
            if key in other_vectors:
                # not all distances are created equal, have a vector of distances and weight distance by importance before summing
                # importance is defined by attribute
                distance += calculate_distance(key, target_vectors[key], other_vectors[key])
        
        distances[distance].append(other_movie.id)
    
    return dict(distances)

# Calculate the Euclidean distances from the first movie to all other movies
movie_id = list(movies.keys())[0]
print("Finding movies similar to movie:", movies[movie_id].title)
distances = calculate_distances(movie_id, movies)

# Get the 5 movies with the smallest distance
closest_movies = sorted(distances.items())[:5]

print("The 5 movies with the smallest distance are:")
for distance, movie_ids in closest_movies:
    for movie_id in movie_ids:
        print(f"Movie: {movies[movie_id].title}, Distance: {distance}")
