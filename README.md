# Book Recommendation System Using K-Nearest Neighbors

This project implements a book recommendation algorithm using the K-Nearest Neighbors (KNN) algorithm. The model is trained on the [Book-Crossings dataset](http://www2.informatik.uni-freiburg.de/~cziegler/BX/), which contains 1.1 million ratings (scale of 1-10) for 270,000 books by 90,000 users.

---

## Project Overview

The goal is to create a recommendation system that suggests books similar to a given book title. The recommendations are based on user rating patterns, and the algorithm uses KNN to measure distances between books to determine their similarity.

### Features

- Filters out users with fewer than 200 ratings and books with fewer than 100 ratings to ensure statistical significance.
- Uses the `NearestNeighbors` class from `sklearn.neighbors` to build the recommendation model.
- Provides a function `get_recommends()` that outputs five similar books along with their distances from the input book.

---

## Dataset Structure

The dataset contains the following:

- **Users:** 90,000 unique users.
- **Books:** 270,000 unique book titles.
- **Ratings:** 1.1 million ratings on a scale of 1 to 10.

---

## Implementation Details

### Data Preparation

1. **Filter the dataset:**
   - Remove users with fewer than 200 ratings.
   - Remove books with fewer than 100 ratings.
2. **Create a user-book matrix:** Rows represent users, columns represent books, and values represent ratings.

### Model Development

The recommendation model is built using the KNN algorithm from scikit-learn's `NearestNeighbors` class. The model identifies the 5 most similar books to any given book based on their distance in the user-book rating space.

### `get_recommends` Function

The `get_recommends()` function accepts a book title and returns a list with:
1. The book title passed as input.
2. A list of 5 recommended books with their similarity scores.

---

## Usage

### Example

```python
# Call the function with a book title from the dataset
get_recommends("The Queen of the Damned (Vampire Chronicles (Paperback))")
