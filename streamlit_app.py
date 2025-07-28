import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from netflix_clustering_model import NetflixClusteringModel
import time
import os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Page configuration
st.set_page_config(
    page_title="Netflix Content Clustering",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #E50914;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #141414;
        margin-bottom: 1rem;
    }
    .stButton > button {
        background-color: #E50914;
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
    }
    .stButton > button:hover {
        background-color: #B2070F;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the Netflix clustering model"""
    model = NetflixClusteringModel()
    
    # Check if model file exists, if not train the model
    if os.path.exists('netflix_clustering_model.pkl'):
        try:
            model.load_model()
            st.success("✅ Model loaded successfully!")
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            return None
    else:
        with st.spinner("Training model for the first time..."):
            if model.load_and_preprocess_data():
               # optimal_k, _, _ = model.find_optimal_clusters()
                n_clusters = st.sidebar.slider("Number of Clusters", 3, 10, 5)
                inertia, silhouette = model.train_model(n_clusters=n_clusters)
                model.save_model()
                st.success(f"✅ Model trained successfully! Silhouette Score: {silhouette:.3f}")
            else:
                st.error("❌ Failed to load data!")
                return None
    
    return model

def main():
    # Header
    st.markdown('<h1 class="main-header">🎬 Netflix Content Clustering</h1>', unsafe_allow_html=True)
    st.markdown("### Discover your perfect Netflix content with AI-powered clustering")
    
    # Load model
    model = load_model()
    if model is None:
        st.error("Failed to load model. Please check your data file.")
        return
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "🎯 Content Recommender", "📊 Cluster Analysis", "🔍 Data Explorer"]
    )
    
    if page == "🏠 Home":
        show_home_page(model)
    elif page == "🎯 Content Recommender":
        show_recommender_page(model)
    elif page == "📊 Cluster Analysis":
        show_cluster_analysis_page(model)
    elif page == "🔍 Data Explorer":
        show_data_explorer_page(model)

def show_home_page(model):
    """Display the home page with overview"""
    st.markdown('<h2 class="sub-header">Welcome to Netflix Content Clustering</h2>', unsafe_allow_html=True)
    
    # Feature cards using native Streamlit components
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🎯 Smart Recommendations")
        st.markdown("Get personalized content recommendations based on your preferences using advanced clustering algorithms.")
    
    with col2:
        st.markdown("### 📊 Cluster Insights")
        st.markdown("Explore different content clusters and understand what makes each group unique.")
    
    with col3:
        st.markdown("### ⚡ Real-time Analysis")
        st.markdown("Get instant results as you change your preferences with our real-time clustering model.")
    
    # Quick stats
    st.markdown("### 📈 Quick Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Shows", f"{len(model.df):,}")
    
    with col2:
        st.metric("Content Types", f"{model.df['type'].nunique()}")
    
    with col3:
        st.metric("Genres", f"{len(model.genre_columns)}")
    
    with col4:
        st.metric("Clusters", f"{model.kmeans.n_clusters}")
    
    # Sample data preview
    st.markdown("### 📋 Sample Data")
    st.dataframe(model.df[['title', 'type', 'rating', 'duration', 'listed_in']].head(10))

def show_recommender_page(model):
    """Display the content recommender page"""
    st.markdown('<h2 class="sub-header">🎯 Content Recommender</h2>', unsafe_allow_html=True)
    st.markdown("Enter your preferences to get personalized Netflix recommendations!")
    
    # User input form
    with st.form("recommendation_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            release_year = st.slider("Release Year", 1940, 2024, 2020)
            duration = st.slider("Duration (minutes)", 30, 300, 120)
            rating = st.selectbox("Rating", model.get_rating_options())
        
        with col2:
            # Genre selection
            st.markdown("**Select Genres:**")
            available_genres = model.get_all_genres()
            selected_genres = st.multiselect(
                "Choose your preferred genres:",
                options=available_genres,
                default=['Dramas', 'Thrillers'] if 'Dramas' in available_genres else available_genres[:2]
            )
        
        submitted = st.form_submit_button("🎬 Get Recommendations", use_container_width=True)
    
    if submitted:
        if not selected_genres:
            st.warning("⚠️ Please select at least one genre for better recommendations.")
            return
            
        with st.spinner("Finding your perfect content..."):
            try:
                # Create user input
                user_input = {
                    'release_year': release_year,
                    'duration': duration,
                    'rating': rating,
                    'genres': selected_genres
                }
                
                # Get predictions
                cluster, similar_shows = model.predict_cluster(user_input)
                
                # Display results
                st.markdown(f"### 🎯 Your Content Cluster: **{cluster}**")
                
                # Cluster information
                cluster_stats = model.get_cluster_statistics()
                if cluster_stats and cluster in cluster_stats:
                    stats = cluster_stats[cluster]
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Shows in Cluster", stats['count'])
                    with col2:
                        st.metric("Avg Release Year", f"{stats['avg_release_year']:.0f}")
                    with col3:
                        st.metric("Avg Duration", f"{stats['avg_duration']:.0f} min")
                    
                    st.markdown(f"**Top Genres in this cluster:** {', '.join(stats['top_genres'][:5])}")
                
                # Recommendations
                st.markdown("### 🎬 Recommended for You")
                
                if len(similar_shows) > 0:
                    for idx, row in similar_shows.iterrows():
                        with st.container():
                            # Create a card-like container using columns
                            col1, col2, col3 = st.columns([4, 1, 0.5])
                            
                            with col1:
                                # Use native Streamlit components instead of HTML
                                st.markdown(f"**{row['title']}**")
                                st.markdown(f"*{row['type']} • {row['rating']} • {row['duration']} min*")
                                st.markdown(f"**Genres:** {row['listed_in']}")
                                if pd.notna(row['description']) and len(str(row['description'])) > 0:
                                    st.markdown(f"*{str(row['description'])[:200]}...*")
                                st.divider()
                            
                            with col2:
                                if st.button(f"Add to List", key=f"btn_{idx}"):
                                    st.success("Added to your watchlist!")
                            
                            with col3:
                                st.write("")  # Empty space for alignment
                else:
                    st.warning("No similar shows found. Try adjusting your preferences.")
                    
            except Exception as e:
                st.error(f"❌ Error getting recommendations: {e}")
                st.info("💡 Try selecting different genres or adjusting your preferences.")

def show_cluster_analysis_page(model):
    """Display cluster analysis page"""
    st.markdown('<h2 class="sub-header">📊 Cluster Analysis</h2>', unsafe_allow_html=True)
    
    # Get cluster statistics
    cluster_stats = model.get_cluster_statistics()
    
    if cluster_stats:
        # Cluster overview
        st.markdown("### 📈 Cluster Overview")
        
        # Create cluster comparison chart
        cluster_data = []
        for cluster_id, stats in cluster_stats.items():
            cluster_data.append({
                'Cluster': f'Cluster {cluster_id}',
                'Count': stats['count'],
                'Avg Release Year': stats['avg_release_year'],
                'Avg Duration': stats['avg_duration']
            })
        
        cluster_df = pd.DataFrame(cluster_data)
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(cluster_df, x='Cluster', y='Count', 
                        title='Number of Shows per Cluster',
                        color='Cluster', color_discrete_sequence=px.colors.qualitative.Set3)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(cluster_df, x='Avg Release Year', y='Avg Duration', 
                           size='Count', color='Cluster',
                           title='Cluster Characteristics',
                           color_discrete_sequence=px.colors.qualitative.Set3)
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed cluster information
        st.markdown("### 🔍 Detailed Cluster Information")
        
        for cluster_id, stats in cluster_stats.items():
            with st.expander(f"Cluster {cluster_id} - {stats['count']} shows"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                    **Statistics:**
                    - Total shows: {stats['count']}
                    - Average release year: {stats['avg_release_year']:.0f}
                    - Average duration: {stats['avg_duration']:.0f} minutes
                    - Most common rating: {stats['most_common_rating']}
                    """)
                
                with col2:
                    st.markdown("**Top Genres:**")
                    for genre in stats['top_genres'][:5]:
                        st.markdown(f"- {genre}")
                
                st.markdown("**Sample Titles:**")
                for title in stats['sample_titles']:
                    st.markdown(f"- {title}")

def show_data_explorer_page(model):
    """Display data exploration page"""
    st.markdown('<h2 class="sub-header">🔍 Data Explorer</h2>', unsafe_allow_html=True)
    
    # Data overview
    st.markdown("### 📋 Dataset Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        **Dataset Information:**
        - Total records: {len(model.df):,}
        - Features: {len(model.features)}
        - Genres: {len(model.genre_columns)}
        """)
    
    with col2:
        st.markdown(f"""
        **Data Types:**
        - Movies: {len(model.df[model.df['type'] == 'Movie']):,}
        - TV Shows: {len(model.df[model.df['type'] == 'TV Show']):,}
        """)
    
    # Interactive filters
    st.markdown("### 🔍 Interactive Filters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_type = st.selectbox("Content Type", ['All'] + model.df['type'].unique().tolist())
    
    with col2:
        year_range = st.slider("Release Year Range", 
                              int(model.df['release_year'].min()), 
                              int(model.df['release_year'].max()),
                              (int(model.df['release_year'].min()), int(model.df['release_year'].max())))
    
    with col3:
        selected_genres = st.multiselect("Filter by Genres", model.genre_columns)
    
    # Apply filters
    filtered_df = model.df.copy()
    
    if selected_type != 'All':
        filtered_df = filtered_df[filtered_df['type'] == selected_type]
    
    filtered_df = filtered_df[
        (filtered_df['release_year'] >= year_range[0]) & 
        (filtered_df['release_year'] <= year_range[1])
    ]
    
    if selected_genres:
        for genre in selected_genres:
            filtered_df = filtered_df[filtered_df[genre] == 1]
    
    # Display filtered results
    st.markdown(f"### 📊 Filtered Results ({len(filtered_df)} shows)")
    
    if len(filtered_df) > 0:
        # Convert rating back to string for display
        display_df = filtered_df[['title', 'type', 'rating', 'duration', 'release_year', 'listed_in']].copy()
        display_df['rating'] = display_df['rating'].map(model.reverse_rating_map)
        
        st.dataframe(display_df.head(20))
        
        # Download button
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="📥 Download filtered data as CSV",
            data=csv,
            file_name='filtered_netflix_data.csv',
            mime='text/csv'
        )
    else:
        st.warning("No shows match your current filters. Try adjusting your criteria.")

if __name__ == "__main__":
    main() 