# Python Project 1 - PySpark (Data Processing & Machine Learning)
This Python (PySpark) project demonstrates Data Processing and Machine Learning tasks using PySpark. 

The code provides a comprehensive overview of data loading, manipulation(filtering), exploration, and building a Machine Learning Pipeline using PySpark for Logistic Regression. 
It covers basic DataFrame operations, schema adjustments, joins, feature engineering using transformers, constructing a pipeline for consistency, and training a logistic regression model with cross-validation for hyperparameter tuning and evaluation. 
Each step demonstrates best practices for working with large-scale data in a distributed computing environment provided by Spark.

# Project Structure
-Datasets: Directory containing sample datasets used in the project (airports.csv, flights_small.csv, planes.csv).

-Scripts: Directory containing the main Python script (Airlines_Project.py) that executes the entire project.

# Requirements
· Python 3.11.2

· PySpark (ensure it's installed and configured correctly)

· numpy, pandas (for Data Manipulation)

# Key Components
‣ Data Loading: Datasets (airports.csv, flights_small.csv, planes.csv) are loaded into Spark DataFrames.

‣ Data Manipulation: Various operations such as filtering, aggregation, and schema adjustments are performed.

‣ Feature Engineering: Categorical variables are transformed using StringIndexer and OneHotEncoder, and features are assembled into vectors.

‣ Machine Learning: Logistic Regression model is trained using cross-validation for hyperparameter tuning.

‣ Evaluation: Model performance is evaluated using the area under the ROC curve (areaUnderROC).

# Files Description
1. pyspark_project.py: Main script file containing the entire pipeline from data loading to model evaluation.

2. airports.csv: Sample dataset containing airport information.
   
3. flights_small.csv: Sample dataset containing flight information.
   
4. planes.csv: Sample dataset containing aircraft information.

# Results 
We used the Evaluator in order to measure the performance of our Binary Classification Model, (such as Logistic Regression), specifically by calculating the area under the Receiver Operating Characteristic (ROC) curve (areaUnderROC). 
tHE 'test_results' is a DataFrame that contains the predictions made by our best logistic regression model (best_lr) on the test dataset. 
It includes predicted probabilities and possibly other columns like prediction (binary predictions) and label (actual labels).
The evaluation score (0.678187114993716) gives us an indication of how well our logistic regression model is performing in terms of classifying flights as either delayed (is_late) or not, based on the features we provided. 

# Next Steps
-Explore additional datasets or customize the pipeline for your specific use case.

-Experiment with different machine learning algorithms or add more features to improve model performance.

-Refactor code for scalability and efficiency, especially for larger datasets.

# Resources
[PySpark Documentation](https://spark.apache.org/docs/latest/api/python/index.html)

[PySpark MLlib Guide](https://spark.apache.org/docs/latest/ml-guide.html)
