import requests
import pathlib
import zipfile
import pandas as pd

class MovieRecommendationLearner:
    def __init__(self):
        self.ratings = None
        self.movies = None
        self.users = None
        self.user_item_matrix = None
        self.tarined_model = {}
    
    def step1_download_data(self):
        # Code to download and load the dataset
        print("\n" + "="*60)
        print("STEP 1: Downloading and loading the dataset...")
        print("="*60)
        url = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"

        try:
            if pathlib.Path("ml-100k.zip").exists():
                print("Dataset already exists. Skipping download.")
                return True
            else:
                respose = requests.get(url)
                with open("ml-100k.zip", "wb") as file:
                    file.write(respose.content) 
                with zipfile.ZipFile("ml-100k.zip", 'r') as zip_ref:
                    zip_ref.extractall("data/")

                print("Dataset downloaded and saved as 'ml-100k.zip'. Please extract it manually.")
                return True
        except Exception as e:
            print(f"An error occurred while downloading the dataset: {e}")
            return False
    
    def step2_load_data(self):
        # Code to load the dataset into memory
        print("\n" + "="*60)
        print("STEP 2: Loading the dataset into memory...")
        print("="*60)
        try:
            self.ratings = pd.read_csv("data/ml-100k/u.data", sep="\t", names=["user_id", "movie_id", "rating", "timestamp"])
            self.movies = pd.read_csv("data/ml-100k/u.item", sep="|", names=["movie_id", "title", "release_date", "video_release_date", "imdb_url"] + [f"genre_{i}" for i in range(19)], encoding='latin-1')
            self.users = pd.read_csv("data/ml-100k/u.user", sep="|", names=["user_id", "age", "gender", "occupation", "zip_code"])
            print("Dataset loaded successfully.")
            # Display dataset info
            print("\nRatings DataFrame Info:")
            print(self.ratings.info())
            print("\nMovies DataFrame Info:")
            print(self.movies.info())
            print("\nUsers DataFrame Info:")
            print(self.users.info())

            # show sample data
            print("\nSample Ratings Data:")
            print(self.ratings.head())
        except Exception as e:
            print(f"An error occurred while loading the dataset: {e}")
            return False
        
        # calculate sparsity
        try:
            num_users = self.ratings['user_id'].nunique()
            num_movies = self.ratings['movie_id'].nunique()
            sparsity = 1 - (len(self.ratings) / (num_users * num_movies))
            print(f"\nSparsity of the user-item matrix: {sparsity:.4f}")
            return True
        except Exception as e:
            print(f"An error occurred while calculating sparsity: {e}")
            return False
        

if __name__ == "__main__":
    learner = MovieRecommendationLearner()
    if learner.step1_download_data():
        if learner.step2_load_data():
            print("Data preparation steps completed successfully.")
        else:
            print("Data loading failed.")
    else:
        print("Data download failed.")

        

    
