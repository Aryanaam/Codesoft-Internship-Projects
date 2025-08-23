IMDb Movie Rating Predictor

The IMDb Movie Rating Predictor is an interactive web application that predicts the IMDb rating of a movie based on its details. Using machine learning and visual analytics, this app allows users to explore how different factors—like genre, director, actors, and release year—affect movie ratings. It also provides insightful visualizations to help understand patterns in the dataset.

This project is ideal for movie enthusiasts, data science learners, and anyone interested in exploring predictive modeling and data visualization with real-world datasets.

Key Features

Predict Movie Ratings:
The app predicts IMDb ratings based on user input for genre, director, actors, and release year.

Interactive Visualizations:

Rating Distribution: Shows how ratings are distributed across the dataset with the predicted rating highlighted.

Genre Analysis: Displays average ratings by genre to identify which genres typically receive higher ratings.

Director Insights: Highlights the top directors by average rating, giving insight into influential filmmakers.

Handles Missing Data:
The model automatically handles missing or unknown values, ensuring reliable predictions.

Web-based Interface:
Users can access the app via a simple, interactive web interface powered by Gradio, without needing to install heavy software or coding experience.

Technologies Used

Python 3 – Core programming language for the project.

pandas & numpy – For efficient data handling and preprocessing.

scikit-learn – Machine learning library used for training the Random Forest regression model.

Matplotlib & Seaborn – For creating high-quality data visualizations.

Gradio – For building an interactive and user-friendly web interface.

Joblib – For saving and loading trained models and encoders.

How It Works

The user enters movie details such as genre, director, lead actors, and release year.

The machine learning model predicts the IMDb rating based on historical data patterns.

Visualizations show the predicted rating in the context of overall movie ratings, as well as genre and director trends.

Users gain both a numerical prediction and a visual understanding of how factors influence movie ratings.

Benefits & Applications

For Movie Enthusiasts: Quickly estimate the potential rating of a movie before release or exploration.

For Data Science Learners: Demonstrates real-world application of machine learning, data preprocessing, feature engineering, and visualization.

For Industry Insights: Helps analysts understand trends and patterns in the Indian movie industry dataset.

Future Enhancements

Dropdown menus for selecting genres, directors, and actors for easier input.

Incorporation of additional features like movie duration, votes, budget, and language for improved predictions.

Batch predictions for multiple movies at once.

Implementation of more advanced machine learning models to improve accuracy.

Deployment as a cloud-hosted app for global access.

License

This project is open-source and free to use. Contributions and improvements are welcome.
