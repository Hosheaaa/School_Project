# Recommender System Projects

## Overview
This folder contains both individual and group projects for the Recommender Systems course.

## Individual Project
### Matrix Factorization Techniques for Movie Recommendation

**Implemented Models**:
1. **WMF (Weighted Matrix Factorization)**
2. **BPR (Bayesian Personalized Ranking)**
3. **BiVAE (Bilateral Variational Autoencoder)**

**Dataset**: MovieLens with ratings data

**Evaluation**: Generated top-50 movie recommendations for each user

**Technologies**:
- Python
- Collaborative Filtering
- Matrix Factorization algorithms

## Group Project (Group 1)
### Hybrid Movie Recommendation System

**Objective**: Build a comprehensive recommendation system combining multiple approaches.

**Dataset**:
- MovieLens dataset (movies.csv, ratings.csv)
- TMDb 5000 Movies dataset
- Merged dataset with rich movie metadata

**Features**:
- Content-based filtering using movie features
- Collaborative filtering using user ratings
- Hybrid approach combining both methods
- LibFM integration for factorization machines

**Key Files**:
- Feature engineering with matched dataset
- Overview and keywords extraction
- Full data and filtered data in LibFM format

**Technologies**:
- Python
- Pandas, NumPy
- LibFM
- Scikit-learn
- Feature engineering techniques

## Requirements
See `requirements.txt` in project directories for dependencies:
- numpy
- scipy
- Cython
- tqdm
- powerlaw
