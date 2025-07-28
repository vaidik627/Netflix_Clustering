import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import joblib
import re
import warnings
warnings.filterwarnings('ignore')

class NetflixClusteringModel:
    def __init__(self):
        self.df = None
        self.scaler = StandardScaler()
        self.kmeans = None
        self.label_encoder = LabelEncoder()
        self.features = None
        self.genre_columns = None
        self.pca = PCA(n_components=2)
        self.rating_map = {
            'PG-13': 0, 'TV-MA': 1, 'PG': 2, 'TV-14': 3, 'TV-PG': 4, 
            'TV-Y': 5, 'TV-Y7': 6, 'R': 7, 'TV-G': 8, 'G': 9, 
            'NC-17': 10, 'NR': 11, 'TV-Y7-FV': 12, 'UR': 13
        }
        self.reverse_rating_map = {v: k for k, v in self.rating_map.items()}
        
    def load_and_preprocess_data(self, csv_path='netflix.csv'):
        """Load and preprocess the Netflix dataset"""
        try:
            # Load data
            self.df = pd.read_csv(csv_path)
            
            # Select relevant columns
            self.df = self.df[['type', 'title', 'rating', 'duration', 'description', 'listed_in', 'release_year']]
            
            # Drop rows with missing values in critical columns
            self.df = self.df.dropna(subset=['rating', 'listed_in', 'duration'])
            
            # Extract numeric duration
            self.df['duration'] = self.df['duration'].str.extract(r'(\d+)').astype(float)
            
            # Map ratings to numeric values
            self.df['rating'] = self.df['rating'].map(self.rating_map).fillna(1)  # Default to TV-MA
            
            # One-hot encode genres
            genres = self.df['listed_in'].str.get_dummies(',')
            self.df = pd.concat([self.df, genres], axis=1)
            
            # Store genre columns
            self.genre_columns = genres.columns.tolist()
            
            # Select features for clustering
            self.features = ['release_year', 'duration', 'rating'] + self.genre_columns
            
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def find_optimal_clusters(self, max_clusters=10):
        """Find optimal number of clusters using elbow method and silhouette score"""
        X = self.df[self.features]
        X_scaled = self.scaler.fit_transform(X)
        
        inertias = []
        silhouette_scores = []
        
        for k in range(2, max_clusters + 1):
            kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans_temp.fit_predict(X_scaled)
            
            inertias.append(kmeans_temp.inertia_)
            silhouette_scores.append(silhouette_score(X_scaled, cluster_labels))
        
        # Find optimal k using silhouette score
        optimal_k = 2 + np.argmax(silhouette_scores)
        
        return optimal_k, inertias, silhouette_scores
    
    def train_model(self, n_clusters=4):
        """Train the clustering model"""
        X = self.df[self.features]
        X_scaled = self.scaler.fit_transform(X)
        
        # Train KMeans
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.df['cluster'] = self.kmeans.fit_predict(X_scaled)
        
        # Fit PCA for visualization
        self.pca.fit(X_scaled)
        
        return self.kmeans.inertia_, silhouette_score(X_scaled, self.df['cluster'])
    
    def predict_cluster(self, user_input):
        """Predict cluster for user input"""
        if self.kmeans is None:
            raise ValueError("Model not trained yet. Please train the model first.")
        
        # Create feature vector from user input
        feature_vector = self._create_feature_vector(user_input)
        
        # Scale the features
        feature_vector_scaled = self.scaler.transform([feature_vector])
        
        # Predict cluster
        cluster = self.kmeans.predict(feature_vector_scaled)[0]
        
        # Get similar shows
        similar_shows = self._get_similar_shows(feature_vector_scaled[0], cluster, top_n=10)
        
        return cluster, similar_shows
    
    def _create_feature_vector(self, user_input):
        """Create feature vector from user input"""
        feature_vector = []
        
        # Release year
        feature_vector.append(user_input.get('release_year', 2020))
        
        # Duration
        feature_vector.append(user_input.get('duration', 90))
        
        # Rating
        rating = user_input.get('rating', 'TV-MA')
        feature_vector.append(self.rating_map.get(rating, 1))
        
        # Genres (one-hot encoding)
        user_genres = user_input.get('genres', [])
        for genre in self.genre_columns:
            feature_vector.append(1 if genre in user_genres else 0)
        
        return feature_vector
    
    def _get_similar_shows(self, feature_vector_scaled, cluster, top_n=10):
        """Get similar shows from the same cluster"""
        # Get shows from the same cluster
        cluster_shows = self.df[self.df['cluster'] == cluster].copy()
        
        if len(cluster_shows) == 0:
            # If no shows in cluster, return random shows
            return self.df[['title', 'type', 'rating', 'duration', 'listed_in', 'description']].head(top_n)
        
        # Calculate distances to find most similar shows
        distances = []
        for idx, row in cluster_shows.iterrows():
            show_features = self.df.loc[idx, self.features].values
            show_features_scaled = self.scaler.transform([show_features])[0]
            
            # Calculate Euclidean distance
            distance = np.linalg.norm(feature_vector_scaled - show_features_scaled)
            distances.append((distance, idx))
        
        # Sort by distance and get top N
        distances.sort()
        similar_indices = [idx for _, idx in distances[:top_n]]
        
        # Get the similar shows and convert rating back to string
        similar_shows = self.df.loc[similar_indices, ['title', 'type', 'rating', 'duration', 'listed_in', 'description']].copy()
        similar_shows['rating'] = similar_shows['rating'].map(self.reverse_rating_map)
        
        return similar_shows
    
    def get_cluster_statistics(self):
        """Get statistics for each cluster"""
        if self.kmeans is None:
            return None
        
        cluster_stats = {}
        for cluster_id in range(self.kmeans.n_clusters):
            cluster_data = self.df[self.df['cluster'] == cluster_id]
            
            if len(cluster_data) == 0:
                continue
                
            stats = {
                'count': len(cluster_data),
                'avg_release_year': cluster_data['release_year'].mean(),
                'avg_duration': cluster_data['duration'].mean(),
                'most_common_rating': self.reverse_rating_map.get(cluster_data['rating'].mode().iloc[0], 'TV-MA') if len(cluster_data['rating'].mode()) > 0 else 'TV-MA',
                'top_genres': self._get_top_genres(cluster_data),
                'sample_titles': cluster_data['title'].head(5).tolist()
            }
            
            cluster_stats[cluster_id] = stats
        
        return cluster_stats
    
    def _get_top_genres(self, cluster_data, top_n=5):
        """Get top genres for a cluster"""
        genre_counts = {}
        for genre in self.genre_columns:
            genre_counts[genre] = cluster_data[genre].sum()
        
        # Sort by count and get top N
        sorted_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)
        return [genre for genre, count in sorted_genres[:top_n] if count > 0]
    
    def get_all_genres(self):
        """Get all available genres"""
        return sorted(self.genre_columns)
    
    def get_rating_options(self):
        """Get all available rating options"""
        return list(self.rating_map.keys())
    
    def save_model(self, filepath='netflix_clustering_model.pkl'):
        """Save the trained model"""
        model_data = {
            'scaler': self.scaler,
            'kmeans': self.kmeans,
            'features': self.features,
            'genre_columns': self.genre_columns,
            'pca': self.pca,
            'df': self.df,
            'rating_map': self.rating_map,
            'reverse_rating_map': self.reverse_rating_map
        }
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath='netflix_clustering_model.pkl'):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        self.scaler = model_data['scaler']
        self.kmeans = model_data['kmeans']
        self.features = model_data['features']
        self.genre_columns = model_data['genre_columns']
        self.pca = model_data['pca']
        self.df = model_data['df']
        self.rating_map = model_data.get('rating_map', self.rating_map)
        self.reverse_rating_map = model_data.get('reverse_rating_map', self.reverse_rating_map)

# Example usage
if __name__ == "__main__":
    # Initialize model
    model = NetflixClusteringModel()
    
    # Load and preprocess data
    if model.load_and_preprocess_data():
        print("Data loaded successfully!")
        
        # Find optimal clusters
        optimal_k, inertias, silhouette_scores = model.find_optimal_clusters()
        print(f"Optimal number of clusters: {optimal_k}")
        
        # Train model
        inertia, silhouette = model.train_model(n_clusters=optimal_k)
        print(f"Model trained! Inertia: {inertia:.2f}, Silhouette Score: {silhouette:.3f}")
        
        # Save model
        model.save_model()
        print("Model saved successfully!")
        
        # Example prediction
        user_input = {
            'release_year': 2020,
            'duration': 120,
            'rating': 'TV-MA',
            'genres': ['Dramas', 'Thrillers']
        }
        
        cluster, similar_shows = model.predict_cluster(user_input)
        print(f"Predicted cluster: {cluster}")
        print("Similar shows:")
        print(similar_shows[['title', 'type', 'listed_in']].head())
    else:
        print("Failed to load data!") 