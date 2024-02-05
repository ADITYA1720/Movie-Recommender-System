import streamlit as st
st.write("In this application, I used two approaches.")

st.write("1. XGBoost")
st.write("- Pre-processed data based on genres and created a One Hot Encoding of every movie in the movie dataset")
st.write("- Merged both the datasets on movie ID")
st.write("- Implemented XGBRegressor to find out best parameters for our model")
st.write("- From the best parameters, implemented a function to get top 5 movies")

st.write("2. Colaborative Filtering Approach")
st.write("- Implemented a user-user based collaborative filtering based approach using PySpark for fast processing of data")
st.write("- This approach finds similar movies using the pearson correlation and also handles the cold start problem")
st.write("- This approach gives excellent results with a RMSE value of 0.78 however, it is not able to output use case based on algorithm structure")
