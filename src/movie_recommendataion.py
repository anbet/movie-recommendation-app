import requests
import pathlib
import zipfile
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

class MovieRecommendationLearner:
    def __init__(self):
        self.ratings = None
        self.movies = None
        self.users = None
        self.user_item_matrix = None
        self.trained_model = {}
    
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
    
    def step3_visualize_data(self):
        # Code to visualize the dataset
        print("\n" + "="*60)
        print("STEP 3: Visualizing the dataset...")
        print("="*60)

        if self.ratings is None:
            print("Ratings data not loaded. Please run step2_load_data() first.")
            return 
        
        if self.users is None:
            print("Users data not loaded. Please run step2_load_data() first.")
            return 
        
        if self.movies is None:
            print("Movies data not loaded. Please run step2_load_data() first.")
            return

        # 1. Rating Distribution
        # 2. Number of Ratings per User
        # 3. Movie Popularity
        # 4. Age Distribution of Users
        # 5. Gender Distribution of Users
        # 6. Top Genres

        try:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            # Rating distribution, histogram
            axes[0].hist(self.ratings['rating'], bins=range(1, 7), align='left', color='skyblue', edgecolor='black')
            axes[0].set_title('Rating Distribution')
            axes[0].set_xlabel('Rating')
            axes[0].set_ylabel('Frequency')

            # Number of ratings per user, histogram
            ratings_per_user = self.ratings.groupby('user_id').size()
            axes[1].hist(ratings_per_user, bins=30, color='salmon', edgecolor='black')
            axes[1].set_title('Number of Ratings per User')
            axes[1].set_xlabel('Number of Ratings')
            axes[1].set_ylabel('Number of Users')

            plt.tight_layout()
            plt.show()

            # Movie popularity histogram
            plt.figure(figsize=(7, 4))
            movie_popularity = self.ratings.groupby('movie_id').size()
            plt.hist(movie_popularity, bins=30, color='lightgreen', edgecolor='black')
            plt.title('Movie Popularity')
            plt.xlabel('Ratings per Movie')
            plt.ylabel('Number of Movies')
            plt.tight_layout()
            plt.show()

            # Age distribution histogram
            plt.figure(figsize=(7, 4))
            plt.hist(self.users['age'], bins=30, color='orange', edgecolor='black')
            plt.title('Age Distribution of Users')
            plt.xlabel('Age')
            plt.ylabel('Number of Users')
            plt.tight_layout()
            plt.show()

            # Gender distribution pie chart
            plt.figure(figsize=(5, 5))
            gender_counts = self.users['gender'].value_counts()
            plt.pie(gender_counts.to_numpy(), labels=gender_counts.index.tolist(), autopct='%1.1f%%', startangle=140)
            plt.title("Gender Distribution of Users")
            plt.tight_layout()
            plt.show()

            # Top Genres horizontal bar plot
            genre_columns = [col for col in self.movies.columns if col.startswith('genre_')]
            genre_names = ['Unknown', 'Action', 'Adventure', 'Animation', 'Children', 
                          'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                          'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 
                          'Sci-Fi', 'Thriller', 'War', 'Western']
            genre_counts = self.movies[genre_columns].sum().values
            top_genres = sorted(zip(genre_names, genre_counts), key=lambda x: x[1], reverse=True)[:10]

            plt.figure(figsize=(8, 5))
            plt.barh([x[0] for x in reversed(top_genres)], [x[1] for x in reversed(top_genres)], color='purple')
            plt.title('Top Movie Genres')
            plt.xlabel('Number of Movies')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"An error occurred while visualizing the dataset: {e}")
            return
    
    def step4_create_user_item_matrix(self):
        # Code to create user-item matrix
        print("\n" + "="*60)
        print("STEP 4: Creating user-item matrix...")
        print("="*60)

        # Create user and movie matrix with ratings, this helps in building recommendation systems and understanding user preferences.
        if self.ratings is None:
            print("Ratings data not loaded. Please run step2_load_data() first.")
            return
        try:
            self.user_item_matrix = self.ratings.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)
            print("User-item matrix created successfully.")
            print(self.user_item_matrix.head())

            # calculate memory usage
            memory_usage = self.user_item_matrix.memory_usage(deep=True).sum() / (1024 ** 2)  # in MB
            print(f"User-item matrix memory usage: {memory_usage:.2f} MB")

            # calculate sparsity
            sparsity = 1 - (np.count_nonzero(self.user_item_matrix) / self.user_item_matrix.size)
            print(f"Sparsity of the user-item matrix: {sparsity:.4f}")
            return True
        except Exception as e:
            print(f"An error occurred while creating the user-item matrix: {e}")
            return False

    def step5_user_based_collaborative_filtering(self):
        # Code for user-based collaborative filtering
        print("\n" + "="*60)
        print("STEP 5: User-based Collaborative Filtering...")
        print("="*60)

        if self.user_item_matrix is None:
            print("User-item matrix not created. Please run step4_create_user_item_matrix() first.")
            return 

        
        print("Calculating user-user similarity matrix using cosine similarity matrix...")
        user_similarity = cosine_similarity(self.user_item_matrix)
        user_similarity_df = pd.DataFrame(user_similarity, index=self.user_item_matrix.index, columns=self.user_item_matrix.index)
        print("User-user similarity matrix calculated successfully.")
        print(user_similarity_df.head())

        def get_user_recommendations(user_id, num_recommendations=5):
            if user_id not in user_similarity_df.index:
                print(f"User ID {user_id} not found in the dataset.")
                return []

            if self.ratings is None:
                print("Ratings data not loaded. Please run step2_load_data() first.")
                return []

            # Get similar users
            similar_users = user_similarity_df[user_id].sort_values(ascending=False).index[1:]

            # Get movies rated by the target user
            user_rated_movies = set(self.ratings[self.ratings['user_id'] == user_id]['movie_id'])

            recommendations = {}
            for sim_user in similar_users:
                sim_user_ratings = self.ratings[self.ratings['user_id'] == sim_user]
                for _, row in sim_user_ratings.iterrows():
                    if row['movie_id'] not in user_rated_movies:
                        if row['movie_id'] not in recommendations:
                            recommendations[row['movie_id']] = 0
                        recommendations[row['movie_id']] += row['rating'] * user_similarity_df.at[user_id, sim_user]

            # Sort recommendations by score
            sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:num_recommendations]
            recommended_movie_ids = [movie_id for movie_id, _ in sorted_recommendations]

            # Get movie titles
            if self.movies is None:
                print("Movies data not loaded. Please run step2_load_data() first.")
                return []
            recommended_movies = self.movies[self.movies['movie_id'].isin(recommended_movie_ids)][['movie_id', 'title']]
            # Print details of generated recommendations
            print(f"\nTop {num_recommendations} recommendations for User ID {user_id}:")
            for idx, row in recommended_movies.iterrows():
                print(f"{row['movie_id']}: {row['title']}")
            return recommended_movies

        # store the function in the class for later use
        self.trained_model['user_based'] = get_user_recommendations
        print("User-based collaborative filtering model is ready to provide recommendations.")
        print("You can get recommendations by calling the method 'trained_model['user_based'](user_id, num_recommendations)'.")
        print("For example: trained_model['user_based'](1, 5) to get top 5 recommendations for user with ID 1.")
        
        return get_user_recommendations

if __name__ == "__main__":
    learner = MovieRecommendationLearner()
    if learner.step1_download_data():
        if learner.step2_load_data():
            print("Data preparation steps completed successfully.")
            learner.step3_visualize_data()
        else:
            print("Data loading failed.")
        if learner.step4_create_user_item_matrix():
            print("User-item matrix created successfully.")
        else:
            print("User-item matrix creation failed.")
        learner.step5_user_based_collaborative_filtering()
        # print sample recommendations for user_id 1
        if 'user_based' in learner.trained_model:
            print(f"Recommended Movie for user id 1: {learner.trained_model['user_based'](1, 5)}")
            

    else:
        print("Data download failed.")




