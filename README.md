# Movie-Recommender-System

In this repository, I used two approaches.

1. XGBoost
   - Pre-processed data based on genres and created a One Hot Encoding of every movie in the movie dataset
   - Merged both the datasets on movie ID
   - Implemented XGBRegressor to find out best parameters for our model
   - From the best parameters, implemented a function to get top 5 movies

2. Colaborative Filtering Approach
   - Implemented a user-user based collaborative filtering based approach using PySpark for fast processing of data
   - This approach finds similar movies using the pearson correlation and also handles the cold start problem
   - This approach gives excellent results with a RMSE value of 0.78 however, it is not able to output use case based on algorithm structure
