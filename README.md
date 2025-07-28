>  Stream live link -"https://netflixclustering-axuw97urcfptbkdgfk4c5d.streamlit.app/"
# 🎬 Netflix Content Clustering with Real-time Recommendations

A comprehensive Netflix content clustering and recommendation system built with machine learning and Streamlit. This application provides real-time content recommendations based on user preferences using K-means clustering.

## ✨ Features

- **🎯 Real-time Recommendations**: Get personalized Netflix content recommendations instantly
- **📊 Advanced Clustering**: K-means clustering with optimal cluster detection
- **⚡ Interactive UI**: Beautiful Streamlit interface with real-time updates
- **📈 Performance Analytics**: Model performance metrics and visualizations
- **🔍 Data Explorer**: Interactive data filtering and exploration
- **📱 Responsive Design**: Modern, Netflix-inspired UI design

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Netflix dataset (`netflix.csv`)

### Installation

1. **Clone or download the project files**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure your Netflix dataset is in the project directory**:
   - File should be named `netflix.csv`
   - Should contain columns: `show_id`, `type`, `title`, `director`, `cast`, `country`, `date_added`, `release_year`, `rating`, `duration`, `listed_in`, `description`

4. **Run the Streamlit application**:
   ```bash
   streamlit run streamlit_app.py
   ```

5. **Open your browser** and navigate to the URL shown in the terminal (usually `http://localhost:8501`)

## 📋 Application Structure

### 🏠 Home Page
- Overview of the application
- Quick statistics about the dataset
- Sample data preview

### 🎯 Content Recommender
- **Real-time input**: Adjust your preferences and see instant results
- **Personalized recommendations**: Get shows based on your selected criteria
- **Cluster information**: Learn about your content cluster
- **Watchlist management**: Add recommended shows to your list

### 📊 Cluster Analysis
- **Cluster overview**: Visual comparison of all clusters
- **Detailed statistics**: In-depth analysis of each cluster
- **Interactive charts**: Explore cluster characteristics

### 📈 Model Performance
- **Performance metrics**: Silhouette score, inertia, and cluster count
- **Elbow method analysis**: Optimal cluster selection visualization
- **Performance interpretation**: Understand model quality

### 🔍 Data Explorer
- **Interactive filters**: Filter by type, year, and genres
- **Data download**: Export filtered results as CSV
- **Dataset overview**: Comprehensive data statistics

## 🎯 How to Use the Recommender

1. **Navigate to "🎯 Content Recommender"** in the sidebar

2. **Set your preferences**:
   - **Release Year**: Choose your preferred year range
   - **Duration**: Select preferred content length
   - **Rating**: Choose content rating (TV-MA, PG-13, etc.)
   - **Genres**: Select multiple genres you enjoy

3. **Click "🎬 Get Recommendations"** to see personalized results

4. **Explore your cluster**: Learn about your content cluster and similar shows

5. **Add to watchlist**: Click "Add to List" for shows you want to watch

## 🔧 Model Details

### Clustering Algorithm
- **Algorithm**: K-means clustering
- **Features**: Release year, duration, rating, and genre one-hot encoding
- **Preprocessing**: StandardScaler for feature normalization
- **Optimal clusters**: Automatically determined using elbow method

### Feature Engineering
- **Numeric features**: Release year, duration
- **Categorical features**: Rating (encoded), genres (one-hot encoded)
- **Text features**: Genres extracted and converted to binary features

### Performance Metrics
- **Silhouette Score**: Measures cluster quality (0-1, higher is better)
- **Inertia**: Sum of squared distances to cluster centers
- **Cluster separation**: Visual analysis of cluster boundaries

## 📊 Data Requirements

Your `netflix.csv` file should contain the following columns:

| Column | Type | Description |
|--------|------|-------------|
| show_id | string | Unique identifier for each show |
| type | string | Content type (Movie/TV Show) |
| title | string | Show title |
| director | string | Director name(s) |
| cast | string | Cast members |
| country | string | Country of origin |
| date_added | string | Date added to Netflix |
| release_year | integer | Year of release |
| rating | string | Content rating |
| duration | string | Duration (e.g., "90 min", "2 Seasons") |
| listed_in | string | Genres (comma-separated) |
| description | string | Show description |

## 🎨 Customization

### Model Parameters
You can modify the clustering model in `netflix_clustering_model.py`:

```python
# Change number of clusters
model.train_model(n_clusters=5)

# Modify feature selection
features = ['release_year', 'duration', 'rating'] + selected_genres
```

### UI Customization
The Streamlit app styling can be customized in `streamlit_app.py`:

```python
# Modify CSS styles
st.markdown("""
<style>
    .main-header {
        color: #E50914;  # Netflix red
        font-size: 3rem;
    }
</style>
""", unsafe_allow_html=True)
```

## 🚀 Performance Tips

1. **First Run**: The model will train automatically on first run (may take a few minutes)
2. **Caching**: Results are cached for faster subsequent runs
3. **Large Datasets**: For datasets >10,000 records, consider sampling for faster processing
4. **Memory**: Ensure sufficient RAM for large datasets

## 🔍 Troubleshooting

### Common Issues

1. **"Failed to load data"**
   - Ensure `netflix.csv` is in the project directory
   - Check file format and column names

2. **"Model not trained"**
   - The model trains automatically on first run
   - Check console for training progress

3. **Slow performance**
   - Reduce dataset size for testing
   - Check available system memory

4. **Import errors**
   - Install all requirements: `pip install -r requirements.txt`
   - Check Python version (3.8+ required)

### Getting Help

If you encounter issues:

1. Check the console output for error messages
2. Verify your data file format
3. Ensure all dependencies are installed
4. Check Python version compatibility

## 📈 Future Enhancements

- **Advanced algorithms**: DBSCAN, hierarchical clustering
- **Content-based filtering**: TF-IDF for description analysis
- **User feedback**: Rating system for recommendations
- **A/B testing**: Compare different clustering approaches
- **API integration**: Connect to Netflix API for real-time data

## 🤝 Contributing

Feel free to contribute to this project by:

1. Reporting bugs
2. Suggesting new features
3. Improving documentation
4. Optimizing performance

## 📄 License

This project is open source and available under the MIT License.

---

**Enjoy discovering your perfect Netflix content! 🎬✨** 
