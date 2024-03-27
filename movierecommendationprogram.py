import tkinter as tk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud

# I want to make this model more accurate by using more of the dataset.

# Load the dataset
movies = pd.read_csv('top_1000_popular_movies_tmdb.csv', lineterminator='\n', index_col = 0)

# Dropping Duplicate titles
print("Before dropping duplicates: ", movies.shape)
# Identify and drop rows with non-unique titles
movies_unique_titles = movies.drop_duplicates(subset=['title', 'genres'])
print("After Dropping Duplicates: ", movies_unique_titles.shape)

# Function to recommend movies based on user input using cosine similarity
def recommend_movies(user_input, movies, top_n=5):
    
    # Create a copy of the DataFrame to avoid modifying the original DataFrame
    movies_copy = movies.copy()
    
    # Combine text fields 
    movies_copy.loc[:, 'combined'] = movies_copy['title'] + ' ' + movies_copy['genres'] + ' ' + movies_copy['overview'] + ' ' + movies_copy['tagline'] + ' ' + movies_copy['production_companies']

    # Vectorize on FULL dataset 
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies_copy['combined'].fillna(''))

    # Compute cosine sim on FULL matrix
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)  

    # Get index of movie matching input
    idx = movies_copy[movies_copy['title'].str.contains(user_input, case=False)].index.tolist()
    print("User input", user_input)
    print("index:", idx)

    if not idx:
        idx = movies_copy[movies_copy['genres'].str.contains(user_input, case=False)].index.tolist()

    if not idx:
        return pd.DataFrame()

    # Calculate similarity scores
    try:
        sim_scores = list(enumerate(cosine_sim[idx][0]))
    except IndexError:
        print(f"Error: Index {idx} is out of bounds for axis 0 with size {len(cosine_sim)}")
        return pd.DataFrame()

    # Sort movies by similarity
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get top N indices
    sim_scores = sim_scores[1:top_n+1]
  
    # Get movie indices
    movie_indices = [i[0] for i in sim_scores]
    
    # Get movie titles and cosine similarity scores
    recommended_movies = movies_copy.iloc[movie_indices][['title', 'genres', 'overview', 'tagline', 'production_companies']]
    recommended_movies['overview'] = recommended_movies['overview'].apply(lambda x: truncate_text(x, max_words=20))
    cosine_similarity_scores = [i[1] for i in sim_scores]
  
    # Scale cosine similarity scores to percentages
    max_score = max(cosine_similarity_scores)
    cosine_similarity_percentages = [round(score / max_score, 2) * 100 for score in cosine_similarity_scores]
  
    # Add cosine similarity scores as percentages to the DataFrame
    recommended_movies['cosine_similarity_percentage'] = cosine_similarity_percentages

    # Return recommendations
    return recommended_movies

# Function to truncate text to a maximum number of words
def truncate_text(text, max_words=20):
    words = text.split()
    truncated_words = words[:max_words]
    truncated_text = ' '.join(truncated_words)
    if len(words) > max_words:
        truncated_text += '...'
    return truncated_text

# Function to perform descriptive analysis of the dataset
def descriptive_analysis(movies):
    
  
    # Visualize the correlation between score and movie count
    score_movie_count_plot = plt.figure(figsize=(8, 6))
    movies.groupby("vote_average").size().plot(kind="line")
    plt.title("Score and Movie Count")
    plt.xlabel("Score")
    plt.ylabel("Movie Count")
    plt.tight_layout()

    budget_revenue_plot = plt.figure(figsize=(8, 6))
    sns.scatterplot(data=movies, x='budget', y='revenue', color='skyblue')
    plt.title("Budget vs. Revenue")
    plt.xlabel("Budget")
    plt.ylabel("Revenue")
    plt.tight_layout()
    plt.show()


    return score_movie_count_plot, budget_revenue_plot

def display_statistics(movies):
    # Calculate summary statistics
    summary_statistics = movies.describe()

    # Create a new window
    stats_window = tk.Toplevel()
    stats_window.title("Summary Statistics")

    # Create a text box in the new window
    summary_statistics_text = tk.Text(stats_window, wrap="word", height=20, width=200)
    summary_statistics_text.pack(padx=10, pady=10)

    # Display summary statistics in the text box
    summary_statistics_text.config(state=tk.NORMAL)
    
    # Add custom description
    summary_statistics_text.insert(tk.END, "Summary Statistics for Movie Dataset:\n\n")
    
    # Display count of missing values for each column
    missing_values = movies.isnull().sum()
    summary_statistics_text.insert(tk.END, "Missing Values:\n")
    summary_statistics_text.insert(tk.END, missing_values.to_string() + "\n\n")
    
    # Display number of unique values for categorical columns
    unique_values = movies.select_dtypes(include=['object']).nunique()
    summary_statistics_text.insert(tk.END, "Unique Values for Categorical Columns:\n")
    summary_statistics_text.insert(tk.END, unique_values.to_string() + "\n\n")
    
    # Display summary statistics
    summary_statistics_text.insert(tk.END, "Descriptive Statistics:\n")
    summary_statistics_text.insert(tk.END, summary_statistics.to_string())

    summary_statistics_text.config(state=tk.DISABLED)

# Function to display visualizations
def display_visualizations():
    score_movie_count_plot, budget_revenue_plot = descriptive_analysis(movies)
    
    # Display vote average vs revenue scatter plot
    canvas3 = FigureCanvasTkAgg(score_movie_count_plot)
    canvas3.get_tk_widget()

        # Display vote average vs revenue scatter plot
    canvas4 = FigureCanvasTkAgg(budget_revenue_plot)
    canvas4.get_tk_widget()



# Function to generate word cloud from movie taglines
def generate_tagline_wordcloud():
    # Concatenate all taglines into a single string
    all_taglines = ' '.join(movies['tagline'].dropna())
    
    # Generate word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_taglines)
    
    # Plot word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title("Tagline Word Cloud")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Only allow input if it contains only letters and spaces to prevent SQL injection or XSS attacks
def validate_input(text):
    return all(char.isalpha() or char.isspace() for char in text)


# Function to handle the recommendation process
def recommend_movies_process():
    def validate_input(text):
        # Only allow input if it contains only letters and spaces
        return all(char.isalpha() or char.isspace() for char in text)
    
    user_input = entry.get().strip()  # Strip leading and trailing whitespace

    if len(user_input) == 0:
        result_label.config(text="Please enter a movie or genre.")
        return
    
    recommendations = recommend_movies(user_input, movies_unique_titles)
    
    if recommendations.empty:
        result_label.config(text="Sorry, no recommendations found.")
    else:
        # Concatenate movie information with line breaks
        formatted_recommendations = "\n\n".join(
            f"Title: {row['title']}\nGenres: {row['genres']}\nOverview: {row['overview']}\nTaglines: {row['tagline']}\nProduction Company: {row['production_companies']}\nCosine Simularity: {row['cosine_similarity_percentage']}%" 
            for index, row in recommendations.iterrows()
        )
        result_label.config(text=formatted_recommendations)

# Create the main application window
window = tk.Tk()
window.title("Movie Recommendation System")

# Set maximum width for the window
window.maxsize(width=800, height=1000)

window.config(bg="#272a2a")

# Header
header_label = tk.Label(window, text="filmBOT", font=("Courier", 24, "bold"), bg="#FF715B", fg="black")
header_label.grid(row=0, column=0, columnspan=6, padx=10, pady=10)

# Create labels, entry, and buttons
instruction_label = tk.Label(window, bg=window.cget('bg'), fg="white", highlightbackground=window.cget('bg'), highlightcolor=window.cget('bg'), text="Enter your favorite movie or genre:")
instruction_label.grid(row=1, column=0, padx=10, pady=5)

entry = tk.Entry(window, width=20)
entry.grid(row=1, column=2, padx=10, pady=5)

# Validate user input to allow only text
validate_text = window.register(validate_input)
entry.config(validate="key", validatecommand=(validate_text, "%P"))

recommend_button = tk.Button(window, text="Recommend Movies", bg="#FFA85C", command=recommend_movies_process)
recommend_button.grid(row=2, column=0, columnspan=3, padx=10, pady=5)

statistics_button = tk.Button(window, text="Display Statistics", bg="#7FC8F8", command=lambda: display_statistics(movies))
statistics_button.grid(row=6, column=0, columnspan=3, padx=10, pady=5) 

visualizations_button = tk.Button(window, text="Display Visualizations", bg="#78F3CE", command=display_visualizations)
visualizations_button.grid(row=5, column=0, columnspan=3, padx=10, pady=5)

# Create button to generate word cloud
wordcloud_button = tk.Button(window, text="Generate Word Cloud", bg="#CFA6E2", command=generate_tagline_wordcloud)
wordcloud_button.grid(row=4, column=0, columnspan=3, padx=10, pady=5)

result_label = tk.Label(window, bg=window.cget('bg'), fg="white", highlightbackground=window.cget('bg'), highlightcolor=window.cget('bg'), text="", justify='left', anchor='w')
result_label.grid(row=7, column=0, columnspan=3, padx=10, pady=5, sticky='w')
result_label.config(wraplength=700, anchor='nw')


# Run the main event loop
window.mainloop()