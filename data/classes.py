import hashlib
import random

import numpy as np
import copy

from parse_utils import to_numpy_array

class MovieMetrics:
    """
    A class to represent aggregated metrics across movies in the database.

    Attributes
    ----------
    attribute_vectors : dict
        A dictionary to store attribute vectors with attribute name as the key.
        Example: {'rating': {'min': 1.0, 'max': 5.0, 'movies': {1, 2}}}
    """

    def __init__(self):
        """Initialize the MovieDatabase with empty dictionary for attribute vectors."""
        self.attribute_vectors = {}

    def update_attribute_vectors(self, movie):
        """
        Update the attribute vectors with the new movie's vectors.

        Parameters
        ----------
        movie : Movie
            The movie object whose vectors are to be used in the updated.
        """
        vectors = movie.get_attribute_vectors()
        for key, vector in vectors.items():
            if isinstance(vector, np.ndarray) and vector.size == 1:
                value = vector[0]
                if key not in self.attribute_vectors:
                    # Initialize the attribute vector if it does not exist
                    self.attribute_vectors[key] = {'min': value, 'max': value, 'movies': {movie.id}}
                else:
                    original_min_max = self.attribute_vectors[key]
                    if value < original_min_max['min']:
                        # Update the minimum value for the attribute
                        self.attribute_vectors[key]['min'] = value
                    if value > original_min_max['max']:
                        # Update the maximum value for the attribute
                        self.attribute_vectors[key]['max'] = value
    
class Movie:
    def __init__(self, title, year, genre, rating, director, actors, plot, budget, 
                 box_office, duration, country, language, awards, 
                 imdb_rating, imdb_votes, imdb_id, aggregated_metrics):
        self.title = title
        self.year = year
        self.genre = genre
        self.rating = rating # this should be R, PG-13, etc.
        self.director = director
        self.actors = actors
        self.plot = plot
        self.budget = budget
        self.box_office = box_office
        self.duration = duration
        self.country = country
        self.language = language
        self.awards = awards
        self.imdb_rating = imdb_rating
        self.imdb_votes = imdb_votes
        self.imdb_id = imdb_id
        self.id = self.generate_movie_id()
        self.attribute_vectors = self.generate_attribute_vectors()
        self.aggregated_metrics = aggregated_metrics

    def generate_movie_id(self) -> str:
        """Generate a unique ID for a movie based on its attributes."""
        unique_string = f"{self.title}{self.year}{self.director}"
        movie_id = hashlib.md5(unique_string.encode()).hexdigest()
        return movie_id
    
    def get_attribute_vectors(self):
        """Return the vector representations for each attribute."""
        return copy.deepcopy(self.attribute_vectors)
    
    def generate_attribute_vectors(self):
        """Generate vector representations for each attribute."""
        #  TODO: all vectors need to be scaled to the same length or distance calculation will be skewed
        vectors = {
            # 'title': self.vectorize(self.title),
            'year': to_numpy_array(self.year),
            # 'genre': self.vectorize(self.genre),
            'rating': None, # this should be R, PG-13, etc.
            'director': None,
            'actors': None,
            # 'plot': self.vectorize(self.plot),
            'budget': to_numpy_array(self.budget),
            'box_office': to_numpy_array(self.box_office),
            'duration': to_numpy_array(self.duration),
            'country': None,
            'language': None,
            'awards': None,
            'imdb_rating': to_numpy_array(self.imdb_rating),
            'imdb_votes': None,
            # 'imdb_id' is not a useful vector for recommendation
        }
        # vectors = self.normalize_vectors(vectors) # TODO: move this to the MovieDatabase class
        return vectors
    
    def get_normalized_vector(self, key):
        """Return the normalized vector for a specific attribute."""
        if key in self.attribute_vectors:
            min_val = self.aggregated_metrics.attribute_vectors[key]['min']
            max_val = self.aggregated_metrics.attribute_vectors[key]['max']
            if max_val != min_val:
                return (self.attribute_vectors[key] - min_val) / (max_val - min_val)
            else:
                return to_numpy_array(0.5)
        return None


    def vectorize(self, attribute):
        """Convert an attribute to a vector representation."""
        # This is a placeholder implementation. You should replace it with actual vectorization logic.
        if isinstance(attribute, str):
            return np.array([ord(char) for char in attribute])
        elif isinstance(attribute, (int, float)):
            return np.array([attribute])
        elif isinstance(attribute, list):
            return np.array([self.vectorize(item) for item in attribute])
        else:
            return np.array([])
        
    def normalize_vectors(self, vectors):
        """Normalize the vectors to unit length."""
        for key, vector in vectors.items():
            if isinstance(vector, np.ndarray):
                # Scale the vector to unit length
                norm = np.linalg.norm(vector)
                if norm != 0:
                    vectors[key] = vector / norm
        return vectors
        
    def update_vector(self, key, value):
        """Update the vector for a specific attribute."""
        self.attribute_vectors[key] = value

    def get_id(self) -> str:
        """Return the unique ID of the movie."""
        return self.id

    def __str__(self):
        return f"{self.title} ({self.year})"

    def __repr__(self):
        return (f"Movie({self.title}, {self.year}, {self.genre}, {self.rating}, "
                f"{self.director}, {self.actors}, {self.plot}, {self.budget}, "
                f"{self.box_office}, {self.duration}, {self.country}, "
                f"{self.language}, {self.awards}, {self.imdb_rating}, "
                f"{self.imdb_votes}, {self.imdb_id})")
    
class Actor:
    def __init__(self, name):
        self.name = name
        self.movies = set()
        self.movie_appearances = 0

    def add_movie(self, movie):
        if movie not in self.movies:
            self.movies.add(movie)
            self.movie_appearances += 1

    def __str__(self):
        return f"{self.name} ({self.movie_appearances} movies)"

    def __repr__(self):
        return f"Actor({self.name}, {self.movies}, {self.movie_appearances})"