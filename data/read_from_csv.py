import pandas as pd
import os

from classes import Movie, MovieDatabase
from parse_utils import convert_to_type, parse_actors
import time

# Get the current directory
current_directory = os.path.dirname(os.path.abspath(__file__))

# Build the full path to the CSV file
file_path = os.path.join(current_directory, 'imdb.csv')

# Read data from CSV file
data = pd.read_csv(file_path)

# Create an array of dictionaries where each element represents a row in the dataset
data = data.to_dict(orient='records')

# Create Movie instances
movieDatabase = MovieDatabase()
for data_dict in data:

    actors_list = parse_actors(data_dict['Actors'])

    movie = Movie(
        title=data_dict['Title'],
        year=convert_to_type(data_dict['Year'], int),
        genre=data_dict['Genre'],
        rating=None, # No rating information in the data_dict
        director=data_dict['Director'],
        actors=actors_list,
        plot=data_dict['Description'],
        budget=None,  # No budget information in the data_dict
        box_office=convert_to_type(data_dict['Revenue (Millions)'], int),
        duration=convert_to_type(data_dict['Runtime (Minutes)'], float),
        country=None,  # No country information in the data_dict
        language=None,  # No language information in the data_dict
        awards=None,  # No awards information in the data_dict
        imdb_rating=convert_to_type(data_dict['Rating'], float),
        imdb_votes=convert_to_type(data_dict['Votes'], int),
        imdb_id=None  # No IMDb ID information in the data_dict
    )
    movieDatabase.add_movie(movie)

    total_movies = len(data)
    current_movie_index = data.index(data_dict) + 1
    proportion_added = current_movie_index / total_movies

# Print the first 5 movies in the database
for movie in list(movieDatabase.get_movies())[:5]:
    print(f"Movie: {movie.title}")
    for attribute, vector in movie.attribute_vectors.items():
        if vector is not None:
            print(f"Attribute: {attribute}, Vector: {vector}, Value: {getattr(movie, attribute)}")
    print("\n")

print(list(movieDatabase.attribute_vectors.keys()))

# # Print the first 5 actors in the database
# for actor in list(movieDatabase.actors.values())[:5]:
#     print(repr(actor))

# IN PROGRESS:  Why is 'year' not normalized? Fix this
# Then fix normalization of non 1-sized vectors. CUrrent implementation breaks other things? DO we also normalize these across movies?