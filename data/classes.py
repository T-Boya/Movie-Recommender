import hashlib
from typing import Any, Dict, List, Union, get_args, get_origin

from sentence_transformers import SentenceTransformer, util

import numpy as np
import copy

from torch import Tensor

from data.class_helpers import calculate_average_movie
from data.parse_utils import isNoneType

model = SentenceTransformer('all-mpnet-base-v2')

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
        for key, value in vectors.items():
            if isinstance(value, float):
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
    ATTRIBUTE_VECTOR_TYPES = {
        'title': (Union[List[Tensor], np.ndarray, Tensor], type(None)),
        'year': (float, type(None)),
        # 'genre': (str, type(None)),
        'rating': (type(None)),
        'director': (type(None)),
        'actors': (Any, type(None)), # TODO: strengthen this type
        'plot': (Union[List[Tensor], np.ndarray, Tensor], type(None)),
        'budget': (float, type(None)),
        'box_office': (float, type(None)),
        'duration': (float, type(None)),
        'country': (type(None)),
        'language': (type(None)),
        'awards': (type(None)),
        'imdb_rating': (float, type(None)),
        'imdb_votes': (type(None)),
        # 'imdb_id': (type(None)),
    }

    PRECALCULATED_VECTORS = [
        'title',
        'year',
        'rating',
        'director',
        'plot',
        'budget',
        'box_office',
        'duration',
        'country',
        'language',
        'awards',
        'imdb_rating'
        'imdb_votes',
    ]
    ON_THE_FLY_VECTORS = ['actors']
        
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
        self.attribute_vectors = {}
        self.aggregated_metrics = aggregated_metrics

        self.generate_attribute_vectors()


    def validate_on_the_fly_vectors(self):
        """Validate that on-the-fly vectors are calculated for all attributes that require them."""
        if not all(self.attribute_vectors[key] is not None for key in self.ON_THE_FLY_VECTORS):
            raise ValueError("On-the-fly vectors not calculated for all attributes.")

    def calculate_on_the_fly_vectors(self, movies, actors):
        """Calculate vectors on the fly for attributes that are not precalculated."""
        self.get_actor_vectors(movies, actors)
        self.validate_on_the_fly_vectors()

    def validate_precalculated_and_on_the_fly_vectors_lists(self):
        """Validate that precalculated and on-the-fly vectors lists do not overlap and their union is equivalent to attribute_vectors."""
        if set(self.PRECALCULATED_VECTORS) & set(self.ON_THE_FLY_VECTORS):
            raise ValueError("Precalculated and on-the-fly vectors lists cannot overlap.")
        if not set(self.PRECALCULATED_VECTORS + self.ON_THE_FLY_VECTORS) == set(self.attribute_vectors.keys()):
            raise ValueError("Precalculated and on-the-fly vectors lists do not match attribute vectors.")
        if not len(self.PRECALCULATED_VECTORS + self.ON_THE_FLY_VECTORS) == len(self.attribute_vectors.keys()):
            raise ValueError("Precalculated and on-the-fly vectors lists contain duplicates.")

    def validate_attribute_vectors(self):
        """Validate that attribute vectors align with ATTRIBUTE_VECTOR_TYPES keys and types."""
        instance_attributes = set(self.attribute_vectors.keys())
        attribute_vector_types_keys = set(self.ATTRIBUTE_VECTOR_TYPES.keys())
        
        if attribute_vector_types_keys != instance_attributes:
            missing_keys = attribute_vector_types_keys - instance_attributes
            extra_keys = instance_attributes - attribute_vector_types_keys
            raise ValueError(f"ATTRIBUTE_VECTOR_TYPES keys and attribute vectors do not match. "
                             f"Missing keys: {missing_keys}, Extra keys: {extra_keys}")
        
        for key, value in self.attribute_vectors.items():
            expected_types = self.ATTRIBUTE_VECTOR_TYPES[key]
            if not isinstance(expected_types, (list, tuple)):
                expected_types = [expected_types]
            if not any(self._is_instance_of_type(value, t) for t in expected_types):
                raise TypeError(f"Attribute '{key}' has type {type(value)}, but expected types are {expected_types}")

    def _is_instance_of_type(self, value, expected_type):
        if expected_type is Any:
            return True
        origin = get_origin(expected_type)
        if origin is None:
            return isinstance(value, expected_type)
        elif origin is Union:
            return any(self._is_instance_of_type(value, t) for t in get_args(expected_type))
        elif origin in {list, List}:
            return isinstance(value, list) and all(self._is_instance_of_type(v, get_args(expected_type)[0]) for v in value)
        elif origin in {dict, Dict}:
            key_type, value_type = get_args(expected_type)
            return (isinstance(value, dict) and 
                    all(self._is_instance_of_type(k, key_type) for k in value.keys()) and 
                    all(self._is_instance_of_type(v, value_type) for v in value.values()))
        else:
            return isinstance(value, origin)

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
            'title':  model.encode(self.title),
            'year': self.year,
            # 'genre': self.genre, # to reinstate this you need to update the get_average_movie method to handle strings, or convert from a string to a vector
            'rating': None, # this should be R, PG-13, etc.
            'director': None,
            'actors': None,
            'plot':   model.encode(self.plot),
            'budget': self.budget,
            'box_office': self.box_office,
            'duration': self.duration,
            'country': None,
            'language': None,
            'awards': None,
            'imdb_rating': self.imdb_rating, # you don't want to fit to rating, you want to maximize it
            'imdb_votes': None, # imdb_rating should be log(imdb_votes) * imdb_rating, where each are separately normalized - want the product but not either individually
            # 'imdb_id' is not a useful vector for recommendation
        }
        self.attribute_vectors = vectors
        self.validate_attribute_vectors()
        # vectors = self.normalize_vectors(vectors) # TODO: move this to the MovieDatabase class
    
    def get_actor_vectors(self, movies, actors):
        """Return the average movie vector for each actor."""
        movie_vectors = []
        for actor in self.actors:
            actor_movies = []
            for movie in actors[actor].movies:
                    vector = movies[movie].attribute_vectors
                    vector.pop('actors', None)
                    actor_movies.append(vector)
            movie_vectors.append(calculate_average_movie(actor_movies))
        self.attribute_vectors['actors'] = calculate_average_movie(movie_vectors)
        self.validate_attribute_vectors()
    
    # TODO: is this normalization necessary? We are already normalizing the vectors in the distance calculation
    def get_normalized_vector(self, key):
        """Return the normalized vector for a specific attribute."""
        if key in ['title', 'genre', 'plot', 'actors']:
            return self.attribute_vectors[key]
        elif key in self.attribute_vectors:
            if isNoneType(self.attribute_vectors[key]):
                return None
            min_val = self.aggregated_metrics.attribute_vectors[key]['min']
            max_val = self.aggregated_metrics.attribute_vectors[key]['max']
            if max_val != min_val:
                return (self.attribute_vectors[key] - min_val) / (max_val - min_val)
            else:
                return 0.5 # this value should have no effect on output
        raise Exception(f"Attribute '{key}' not found in the attribute vectors.")
        
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