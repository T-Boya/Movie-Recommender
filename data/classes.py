import hashlib
import random

import numpy as np
import copy

from parse_utils import to_numpy_array

class MovieDatabase:
    """
    A class to represent a movie database.

    Attributes
    ----------
    movies : dict
        A dictionary to store movies with movie id as the key.
        Example: {1: <Movie object>, 2: <Movie object>}
    attribute_vectors : dict
        A dictionary to store attribute vectors with attribute name as the key.
        Example: {'rating': {'min': 1.0, 'max': 5.0, 'movies': {1, 2}}}
    """

    def __init__(self):
        """Initialize the MovieDatabase with empty dictionaries for movies and attribute vectors."""
        self.movies = {}
        self.actors = {}
        self.attribute_vectors = {}

    def add_movie(self, movie):
        """
        Add a movie to the database and update attribute vectors.

        Parameters
        ----------
        movie : Movie
            The movie object to be added to the database.
        """
        # Add the movie to the movies dictionary
        self.movies[movie.id] = movie
        # Update attribute vectors with the new movie's vectors
        updated_fields, movies_to_normalize = self.update_attribute_vectors(movie)
        # Normalize vectors across all movies if there are updated fields
        if updated_fields:
            self.normalize_vectors_across_movies(updated_fields, movies_to_normalize, movie.id)
        # Add actors to the actors dictionary
        for actor_name in movie.actors:
            if actor_name not in self.actors:
                self.actors[actor_name] = Actor(actor_name)
            self.actors[actor_name].add_movie(movie.id)

    def update_attribute_vectors(self, movie):
        """
        Update the attribute vectors with the new movie's vectors.

        Parameters
        ----------
        movie : Movie
            The movie object whose vectors are to be updated.

        Returns
        -------
        updated_fields : dict
            A dictionary of updated fields with their original min and max values.
        """
        vectors = movie.get_attribute_vectors()
        updated_fields = {}
        movies_to_normalize = {movie.id}
        for key, vector in vectors.items():
            if isinstance(vector, np.ndarray) and vector.size == 1:
                value = vector[0]
                if key not in self.attribute_vectors:
                    # Initialize the attribute vector if it does not exist
                    self.attribute_vectors[key] = {'min': value, 'max': value, 'movies': {movie.id}}
                    updated_fields[key] = {'min': value, 'max': value}
                else:
                    original_min_max = self.attribute_vectors[key]
                    if value < original_min_max['min']:
                        # Update the minimum value for the attribute
                        self.attribute_vectors[key]['min'] = value
                        movies_to_normalize.update(self.attribute_vectors[key]['movies'])
                    if value > original_min_max['max']:
                        # Update the maximum value for the attribute
                        self.attribute_vectors[key]['max'] = value
                        movies_to_normalize.update(self.attribute_vectors[key]['movies'])
                    updated_fields[key] = {'min': original_min_max['min'], 'max': original_min_max['max']}
                    # Add the movie id to the set of movies for this attribute
                    self.attribute_vectors[key]['movies'].add(movie.id)
        return updated_fields, movies_to_normalize

    def normalize_vectors_across_movies(self, updated_fields, movies_to_normalize, new_movie_id):
        """
        Normalize the attribute vectors across all movies.

        Parameters
        ----------
        updated_fields : dict
            A dictionary of updated fields with their original min and max values.
        new_movie_id : int
            The id of the new movie being added.
        """
        for key, original_min_max in updated_fields.items():
            min_max = self.attribute_vectors[key]
            min_value = min_max['min']
            max_value = min_max['max']
            original_min_value = original_min_max['min']
            original_max_value = original_min_max['max']
            range_value = max_value - min_value
            for movie_id in movies_to_normalize:
                movie = self.movies[movie_id]
                vectors = movie.get_attribute_vectors()
                if key in vectors and isinstance(vectors[key], np.ndarray) and vectors[key].size == 1:
                    init_value = vectors[key][0]
                    if range_value > 0:
                        if movie_id == new_movie_id:
                            # Normalize the new movie
                            vectors[key] = (vectors[key] - min_value) / range_value
                        else:
                            # Scale already-normalized score up/down based on change to min/max
                            current_value = vectors[key] * (original_max_value - original_min_value) + original_min_value
                            vectors[key] = (current_value - min_value) / range_value
                    else:
                        # If range_value is 0, set the normalized value to 0.5 to differentiate
                        vectors[key] = np.array([0.5])
                    # Update the movie's vector with the normalized value
                    movie.update_vector(key, vectors[key])
                    if key == 'year' and min_value < 2012 and movie.title == 'Prometheus':
                        _start = init_value
                        _end = vectors[key][0]
                        pass

    def get_movies(self):
        """
        Get a list of all movies in the database.

        Returns
        -------
        list
            A list of all movie objects in the database.
        """
        return list(self.movies.values())
    
class Movie:
    def __init__(self, title, year, genre, rating, director, actors, plot, budget, 
                 box_office, duration, country, language, awards, 
                 imdb_rating, imdb_votes, imdb_id):
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