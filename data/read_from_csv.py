import pandas as pd
import os

from data.classes import Actor, Movie, MovieMetrics
from data.parse_utils import convert_to_type, parse_actors

def load_movie_data():
    # Get the current directory
    current_directory = os.path.dirname(os.path.abspath(__file__))

    # Build the full path to the CSV file
    file_path = os.path.join(current_directory, 'imdb.csv')

    # Read data from CSV file
    data = pd.read_csv(file_path)

    # Create an array of dictionaries where each element represents a row in the dataset
    data = data.to_dict(orient='records')
    data = data[:50]  # Limit the number of movies to 10 for faster processing

    # Create Movie instances
    movies = {}
    actors = {}
    metrics = MovieMetrics()
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
            imdb_id=None,  # No IMDb ID information in the data_dict
            aggregated_metrics = metrics,
        )

        movies[movie.id] = movie
        for actor_name in actors_list:
            if actor_name in actors:
                actors[actor_name].add_movie(movie.id)
            else:
                actor = Actor(name=actor_name)
                actor.add_movie(movie.id)
                actors[actor_name] = actor

        metrics.update_attribute_vectors(movie)
    
    return movies, actors, metrics