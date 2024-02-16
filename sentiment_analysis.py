from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.streaming import StreamingContext
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pyspark.sql.types import StructType, StructField, StringType
from pyspark.sql.functions import udf, col, count, window, avg, lit, ArrayType
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml import Pipeline
from pyspark.sql.types import DoubleType

import json
import time
import re

# Function to extract any references to users, posts, or urls
def extract_references(text):
    user_pattern = r'/u/(\w+)'
    post_pattern = r'/r/(\w+)'
    url_pattern = r'(https?://[^\s]+)'
    
    users = re.findall(user_pattern, text)
    posts = re.findall(post_pattern, text)
    urls = re.findall(url_pattern, text)
    
    references = []
    references.extend(users)
    references.extend(posts)
    references.extend(urls)
    
    return references

# Function to perform sentiment analysis on text
def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment = analyzer.polarity_scores(text)
    return sentiment["compound"]


# Create a SparkSession and StreamingContext
spark_conf = SparkConf().setAppName("reddit")
ss1 = SparkSession.builder.config(conf=spark_conf).getOrCreate()
ssc = StreamingContext(ss1.sparkContext, 5)

# Create a DStream
lines = ssc.socketTextStream("localhost", 9999)
comments = lines.map(lambda json_data: json.loads(json_data))


# Define the schema for the DataFrame
schema = StructType([
    StructField("comment",StringType(),True),
    StructField("prev_comment",StringType(),True),
    StructField("post",StringType(),True),
    StructField("author",StringType(),True),
    StructField("link_url",StringType(),True),
    StructField("link_permalink",StringType(),True),
    StructField("post_date",StringType(),True),
    StructField("ups",StringType(),True),
    StructField("likes",StringType(),True),
    StructField("post_img",StringType(),True)
    ])

# Storage for data collected
base_path = "./data/raw/reddit_v5"

# Convert each RDD in the DStream to a DataFrame
def process_rdd(time, rdd):
    if not rdd.isEmpty():
        df = ss1.createDataFrame(rdd, schema)
        
        #Extract references to users, posts, and external sites
        regex_extract_udf = udf(lambda text: extract_references(text), ArrayType(StringType()))
        df_with_refs = df.withColumn("refs", regex_extract_udf(col("comment")))
        
        #Count occurrences of references in 60-second windows every 5 seconds
        windowed_counts = df_with_refs \
            .withWatermark("post_date", "60 seconds") \
            .groupBy(window("post_date", "5 seconds")) \
            .agg(count("refs").alias("reference_count"))
        
        #Get top 10 important words in window using TF-IDF
        tokenizer = Tokenizer(inputCol="comment", outputCol="words")
        hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=10000)
        idf = IDF(inputCol="rawFeatures", outputCol="features")
        
        pipeline = Pipeline(stages=[tokenizer, hashingTF, idf])
        model = pipeline.fit(df)
        tf_idf_result = model.transform(df)

        # store the top 10 words in a list
        top10 = tf_idf_result.select("words", "features").rdd \
            .map(lambda x: x[0]).take(10)
        
        # Convert the list to a string representation
        top10_str = str(top10)

        # Add the top10 words to the dataframe
        df_with_refs = df_with_refs.withColumn("top10", lit(top10_str))
        
        #Get the time range of the data
        min_time = df.selectExpr("MIN(post_date)").first()[0]
        max_time = df.selectExpr("MAX(post_date)").first()[0]
        
        #Perform sentiment analysis of every post and calculate average sentiment
        sentiment_udf = udf(lambda text: analyze_sentiment(text), DoubleType())
        sentiment_scores = df.withColumn("sentiment", sentiment_udf(col("comment")))
        average_sentiment = sentiment_scores.select(avg("sentiment")).first()[0]

        # Add the time, min_time, max_time, average_sentiment to the dataframe
        df_with_refs = df_with_refs.withColumn("time", lit(time))
        df_with_refs = df_with_refs.withColumn("min_time", lit(min_time))
        df_with_refs = df_with_refs.withColumn("max_time", lit(max_time))
        df_with_refs = df_with_refs.withColumn("average_sentiment", lit(average_sentiment))
        
        #Save the processed data to disk
        output_path = f"{base_path}/{time.strftime('%Y%m%d%H%M%S')}"
        df_with_refs.write.json(output_path)      
        
        # Print some information for verification
        # print(f"Time: {time}, Data Range: {min_time} - {max_time}, Average Sentiment: {average_sentiment}")
        # Show the output
        df_with_refs.show()

comments.foreachRDD(process_rdd)

# Start the streaming context
ssc.start()  
# no ssc.awaitTermination() added here to make the cell non blocking and to use other cell in parallel.