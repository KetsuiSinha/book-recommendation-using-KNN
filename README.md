# Book Recommendation System: K-Nearest Neighbors

This project implements a book recommendation algorithm using the K-Nearest Neighbors (KNN) algorithm. The dataset used is the Book-Crossings dataset, which contains 1.1 million ratings (scale of 1-10) for 270,000 books by 90,000 users.

## Objective
- Build a recommendation system that suggests books similar to a given book title.
- Use the KNN algorithm to measure the closeness of books based on user ratings.

## Dataset Details
- The dataset includes ratings from users for various books.
- To ensure statistical significance, the following filters are applied:
  - Remove users with fewer than 200 ratings.
  - Remove books with fewer than 100 ratings.

## Implementation Steps

### 1. Data Preparation
- **Data Cleaning**: Filter the dataset to remove users and books with insufficient ratings.
- **Feature Matrix**: Create a matrix where rows represent books and columns represent users. The values in the matrix are user ratings for books.

### 2. Model Creation
- Use the `NearestNeighbors` algorithm from `sklearn.neighbors`.
- Fit the KNN model on the feature matrix.
- Use distance metrics to find the closest books.

### 3. Recommendation Function
- Define a function `get_recommends` that takes a book title as input.
- Find the book's index in the dataset and retrieve the top 5 nearest neighbors.
- Return the original book title along with a list of recommended books and their distances.

### 4. Example Output
The function `get_recommends` should return recommendations in the following format:

```python
get_recommends("The Queen of the Damned (Vampire Chronicles (Paperback))")
```
Output:
```python
[
  'The Queen of the Damned (Vampire Chronicles (Paperback))',
  [
    ['Catch 22', 0.793983519077301], 
    ['The Witching Hour (Lives of the Mayfair Witches)', 0.7448656558990479], 
    ['Interview with the Vampire', 0.7345068454742432],
    ['The Tale of the Body Thief (Vampire Chronicles (Paperback))', 0.5376338362693787],
    ['The Vampire Lestat (Vampire Chronicles, Book II)', 0.5178412199020386]
  ]
]
```

### 5. Graphical Analysis (Optional)
- Visualize the dataset to understand the distribution of ratings and sparsity of the matrix.
- Plot relationships between books and users.

## Example Usage

```python
from sklearn.neighbors import NearestNeighbors
import pandas as pd

# Data Preparation
filtered_users = dataset[dataset['user_rating_count'] > 200]
filtered_books = filtered_users[filtered_users['book_rating_count'] > 100]
ratings_matrix = filtered_books.pivot(index='book_title', columns='user_id', values='rating').fillna(0)

# Model Creation
model = NearestNeighbors(metric='cosine', algorithm='brute')
model.fit(ratings_matrix)

# Recommendation Function
def get_recommends(book_title):
    book_index = ratings_matrix.index.tolist().index(book_title)
    distances, indices = model.kneighbors(ratings_matrix.iloc[book_index, :].values.reshape(1, -1), n_neighbors=6)
    recommendations = [
        [ratings_matrix.index[i], distances[0][idx]]
        for idx, i in enumerate(indices[0][1:])
    ]
    return [book_title, recommendations]

# Example Call
get_recommends("The Queen of the Damned (Vampire Chronicles (Paperback))")
```
