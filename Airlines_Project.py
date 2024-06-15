# Python (PySpark) Project: Data Processing & Machine Learning 
# Import Libraries
import numpy as np
import pandas as pd 
import pyspark as spark
from pyspark.sql import SparkSession
from pyspark.context import SparkContext
import pyspark.sql.functions as F
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
import pyspark.ml.evaluation as evals 
import pyspark.ml.tuning as tune

# SparkContext: Create the connection to the Spark Cluster 
sc = spark.SparkContext.getOrCreate()
print(sc)
# SparkSession: Create My-Spark (catalog that lists all the data inside our cluster)
spark = SparkSession.builder.getOrCreate()
print(spark)
# Create Dataframe
pd_temp = pd.DataFrame(np.random.random(10)) # 10 random numbers
# Create spark_temp for pd_temp (create Spark DF from Pandas DF)
spark_temp = spark.createDataFrame(pd_temp)
# Print the Tables in the Catalog 
print(spark.catalog.listTables()) # empty
# Add spark_temp to the Catalog
spark_temp.createOrReplaceTempView("temp") # updated the existing Table (Catalog)


# DATASETS
# 1. Airports Dataset 
file_path = '/Users/evgenia/Desktop/Python/airports.csv'
airports = spark.read.csv(file_path, header=True)
airports.show()
type(airports)

# spark.catalog.listDatabases()
# spark.catalog.listTables()

# 2. Flights Dataset 
flights = spark.read.csv('/Users/evgenia/Desktop/Python/flights_small.csv', header=True)
flights.show()
flights.name=flights.createOrReplaceTempView("flights")
spark.catalog.listTables()
# Create the Dataframe for Flights
flights_df = spark.table('flights')
print(flights_df.show())
# New Column: 'duration_hours'
flights = flights.withColumn('duration_hours', flights.air_time/60) 
flights.show()
flights.describe().show() # summary statistics

# Data Processing
# How to filter:
# Filter Flights with a SQL strings
long_flights1 = flights.filter('distance > 100') # string 
long_flights1.show()
# Filter Flights with a boolean column
long_flights2 = flights.filter(flights.distance > 100) # boolean 
long_flights2.show()

# How to select columns:
# Select the first set of columns as a string
selected_1 = flights.select('tailnum','origin','dest')
# Select the second set of columns usinf df.col_name
temp = flights.select(flights.origin,flights.dest,flights.carrier)

# Define first filter to only keep flights from SEA to PDX 
FilterA = flights.origin == 'SEA' # from SEA
FilterB = flights.dest == 'PDX'   # to PDX 
# Filter the Data first by 1st filter & then by 2nd filter
selected_2 = temp.filter(FilterA).filter(FilterB)
selected_2.show()

# Create a Table of the Average Speed of each flight both ways.
# Calculate average speed by dividing the distance by the air_time (converted to hours).
# Use the .alias() method name
# Define avg_speed
avg_speed = (flights.distance/(flights.air_time/60)).alias('avg_speed') 
speed_1 = flights.select('origin','tailnum','dest', avg_speed)
speed_1.show()
# Using the Spark DataFrame method .selectExpr()
speed_2 = flights.selectExpr('origin','tailnum','dest','distance/(air_time/60) as avg_speed')
speed_2.show()
flights.describe()
#air_time & distance: sting, so to find min() and max() we need to convert these to float (cast)
flights = flights.withColumn('air_time',flights.air_time.cast('float'))
flights = flights.withColumn('distance', flights.distance.cast('float'))
flights.describe('air_time', 'distance').show()
flights.describe()
flights.printSchema() # check the data types

# Find the length of the shortest (in terms of distance) flight that left PDX
flights.filter(flights.origin == 'PDX').groupBy().min('distance').show()
# Find the length of the shortest (in terms of time) flight that left PDX
flights.filter(flights.origin == 'SEA').groupBy().max('air_time').show()
# Get the average air time of Delta Airlines flights that left SEA
flights.filter(flights.carrier == 'DL').filter(flights.origin == 'SEA').groupBy().avg('air_time').show()
# Get the total number of hours all planes in this dataset spent in the air by creating a column called duration_hrs
flights.withColumn('duration_hrs', flights.air_time/60).groupBy().sum('duration_hrs').show()
# Group by tailnum column
# flights.withColumn('duration_hrs', flights.air_time/60).groupBy('tailnum').sum('duration_hrs').show() 
by_plane = flights.groupBy('tailnum')
# Use the .count() method with no arguments to count the number of flights each plane made
by_plane.count().show()
# Group by origin column
by_origin = flights.groupBy('origin')
# Find the .avg() of the air_time column to find average duration of flights from PDX and SEA
by_origin.avg('air_time').show()

# Convert column 'dep_delay' to numeric 
flights = flights.withColumn('dep_delay', flights.dep_delay.cast('float'))
flights.describe('dep_delay')
flights.printSchema() # check the data types
# Group by month & dest
by_month_dest = flights.groupBy('month','dest')
# Average departure delay by month & destination
by_month_dest.avg('dep_delay').show()
# Rename the 'faa' column
airports.show()
airports = airports.withColumnRenamed('faa','dest')
print("DataFrame after renaming column:")
airports.show()
airports.printSchema()

# Join the DataFrames: Airports & Flights
flights.show()
airports.show()
flights_airports = flights.join(airports, on = 'dest', how = 'leftouter')
flights_airports.show()
flights_airports.describe()

# 3. Planes Dataset 
planes = spark.read.csv('/Users/evgenia/Desktop/Python/planes.csv', header = True)
planes.show()
# Rename 'year' column to 'plane_year' in order to avoid duplicate column name
planes = planes.withColumnRenamed('year', 'plane_year')
planes.show()

# Join the DataFrames: Flights & Planes
flights_planes = flights.join(planes, on = 'tailnum', how = 'leftouter')
flights_planes.show()
flights_planes.describe()

# Change Data Types of columns
flights_planes = flights_planes.withColumn('arr_delay', flights_planes.arr_delay.cast('Integer'))
flights_planes = flights_planes.withColumn('air_time', flights_planes.air_time.cast('integer'))
flights_planes = flights_planes.withColumn('month', flights_planes.month.cast('integer'))
flights_planes = flights_planes.withColumn('plane_year', flights_planes.plane_year.cast('integer'))
print("Updated Schema:")
flights_planes.printSchema()
flights_planes.describe('arr_delay','air_time','month','plane_year')
flights_planes.describe('arr_delay','air_time','month','plane_year').show()

# Create a New Column(s):
flights_planes = flights_planes.withColumn('plane_age', flights_planes.year - flights_planes.plane_year)
flights_planes = flights_planes.withColumn('is_late', flights_planes.arr_delay > 0) 
flights_planes = flights_planes.withColumn('label', flights_planes.is_late.cast('Integer')) 
flights_planes.show() 
flights_planes.filter("arr_delay is not NULL and dep_delay is not NULL and air_time is not NULL and plane_year is not NULL")
flights_planes.printSchema()

# Machine Learning 
# Convert categorical variables ('carrier' & 'dest' columns) into numerical indices ('carrier_index' & 'dest_index' columns).
# Create StringIndexer: maps a string column to an index column
carr_indexer = StringIndexer(inputCol='carrier', outputCol='carrier_index')
dest_indexer = StringIndexer(inputCol='dest', outputCol='dest_index')
# Converts the numerical indices generated by StringIndexer into binary vectors ('carr_fact' & 'dest_fact' columns).
# Create OneHotEncoder: transform categorical data into a more suitable format for ML (binary vector format)
carr_encoder = OneHotEncoder(inputCol='carrier_index', outputCol='carr_fact')
dest_encoder = OneHotEncoder(inputCol='dest_index', outputCol='dest_fact')

# Assemble Vector: merges multiple columns into a vector column 
# Assemble all features (month, air_time, carr_fact, dest_fact, plane_age) into a single vector column (features).
vec_assembler = VectorAssembler(inputCols=['month','air_time','carr_fact','dest_fact','plane_age'],
                                outputCol='features', handleInvalid="skip")

# Create the Pipeline: combines all the Estimators and Transformers that we've created!
# Combine all the transformation stages (StringIndexer, OneHotEncoder, VectorAssembler) into a single Pipeline for easier reuse and consistency.
flights_pipeline = Pipeline(stages=[dest_indexer, dest_encoder, carr_indexer, carr_encoder, vec_assembler])
# Fitting the Pipeline
piped_data = flights_pipeline.fit(flights_planes).transform(flights_planes)
piped_data.show()

# Splitting Data with weights 0.6 - 0.4 (60 - 40 %)
training, test = piped_data.randomSplit([.6,.4])

lr = LogisticRegression()

# Create the Evaluator 
# Our model is a Binary Classification Problem - so we will use 'BinaryClassificationEvaluator' 
# This Evaluator calculates the area under the ROC curve (False Positives & False Negatives)
evaluator = evals.BinaryClassificationEvaluator(metricName='areaUnderROC')

# Create the Parameter grid
grid = tune.ParamGridBuilder()
# Add the Hyperparameter 
grid = grid.addGrid(lr.regParam, np.arange(0, .1, .01))
grid = grid.addGrid(lr.elasticNetParam, [0,1])
# Build the grid
grid = grid.build()

# Create the Cross Validator
cv = tune.CrossValidator(estimator=lr,
                         estimatorParamMaps=grid,
                         evaluator=evaluator)

# Fit Cross-Validation models
models = cv.fit(training)
# Extract the best model
best_lr = models.bestModel

# Use the model to predict he test-set 
test_results = best_lr.transform(test)
# Evaluate the predictions
print(evaluator.evaluate(test_results))

