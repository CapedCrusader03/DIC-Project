
from pydantic import BaseModel
import pandas as pd
import pickle
import mysql.connector
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import sqlite3

# Initialize the FastAPI app
# app = FastAPI()

# Load the saved model
with open('Kshitij_Hypothesis1.pkl', 'rb') as file:
    model = pickle.load(file)

# Define input schema
class PaperInput(BaseModel):
    authors: str
    title: str
    categories: str
    abstract: str
    year: int
    persist: bool  # Whether to save the data to the database or not

# Database connection settings
# db_config = {
#     'host': "dicproject.c3qgwuokuqml.us-east-2.rds.amazonaws.com",
#     'user': "admin",
#     'password': "kshitij123",
#     'database': "dicproject"
# }

db_config = "neurips_arxiv.db"

def create_entry(table, data):
    with sqlite3.connect(db_config) as conn:
        columns = ", ".join(data.keys())
        placeholders = ", ".join("?" * len(data))
        sql = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
        conn.execute(sql, list(data.values()))
        conn.commit()

# def read_entries(table, condition=""):
#     with sqlite3.connect(db_config) as conn:
#         sql = f"SELECT * FROM {table} {condition}"
#         return pd.read_sql(sql, conn)


# Simplify category function (matches the training logic)
def simplify_category(categories):
    return categories.split('-')[0].split('.')[0]

# Train the model on new data
def retrain_model():

    #sqlite3 db
    conn = sqlite3.connect(db_config)
    query = "SELECT * FROM neurips_arxiv"

    df = pd.read_sql(query,conn)

    # # Connecting to the database
    # conn = mysql.connector.connect(**db_config)
    # cursor = conn.cursor(dictionary=True)

    # # Fetch data from the database
    # query = "SELECT * FROM neuiprs_arxiv_data"
    # cursor.execute(query)
    # rows = cursor.fetchall()

    # Close the connection
    conn.close()

    # Convert to DataFrame
    # df = pd.DataFrame(rows)

    # Add simplified_category
    df['simplified_category'] = df['categories'].apply(simplify_category)

    # Process data similar to your initial code
    grouped_data = df.groupby('simplified_category').size().reset_index(name='count')
    data_count = (
        df.groupby(['year', 'simplified_category'])
        .size()
        .reset_index(name='paper_count')
    )
    data_count['growth_rate'] = data_count.groupby('simplified_category')['paper_count'].pct_change() * 100
    volatility_df = data_count.groupby('simplified_category')['growth_rate'].std().reset_index()
    volatility_df.columns = ['simplified_category', 'volatility']
    data_count = data_count.merge(volatility_df, on='simplified_category', how='left')
    data_count['growth_rate_last_year'] = data_count.groupby('simplified_category')['growth_rate'].shift(1)
    data_count['volatility_last_year'] = data_count.groupby('simplified_category')['volatility'].shift(1)
    data_count = data_count.dropna().reset_index(drop=True)

    # Split data: training up to 2022
    train_data = data_count[data_count['year'] <= 2022]
    X_train = train_data[['paper_count', 'growth_rate_last_year']]
    y_train = train_data['growth_rate']

    # Retrain the model
    new_model = RandomForestRegressor(random_state=42)
    new_model.fit(X_train, y_train)

    with open('Kshitij_Hypothesis1.pkl', 'wb') as file:
        pickle.dump(new_model, file)

    return new_model

def predict_trend(input_data: PaperInput):
    simplified_category = simplify_category(input_data.get("categories"))

    if input_data.get("persist"):
        # conn = mysql.connector.connect(**db_config)
        # cursor = conn.cursor()

        # query = """
        #     INSERT INTO neuiprs_arxiv_data (authors, title, categories, abstract, year)
        #     VALUES (%s, %s, %s, %s, %s)
        # """
        # query = "INSERT INTO neurips_arxiv (authors, title, categories, abstract, year) VALUES (%s, %s, %s, %s, %s)"
        query = "INSERT INTO neurips_arxiv (authors, title, categories, abstract, year) VALUES (?, ?, ?, ?, ?)"
        conn = sqlite3.connect(db_config)
        cursor = conn.cursor()
        values = [input_data.get("authors"), input_data.get("title"), input_data.get("categories"), input_data.get("abstract"), input_data.get("year")]
        cursor.execute(query, values)
        conn.commit()
        conn.close()

        # Retrain the model with new data
        updated_model = retrain_model()
    else:
        updated_model = model

    # Prepare input data for prediction
    paper_data = {
        'year': [input_data.get("year")],
        'simplified_category': [simplified_category],
        'paper_count': [1],  # Assume 1 paper for user input
        'growth_rate_last_year': [0],  # Use default values
    }
    paper_df = pd.DataFrame(paper_data)

    # Make prediction
    predicted_growth_rate = updated_model.predict(paper_df[['paper_count', 'growth_rate_last_year']])[0]

    # Classify trend
    growing_threshold = 3
    declining_threshold = -3
    if predicted_growth_rate >= growing_threshold:
        trend = "Growing"
    elif predicted_growth_rate <= declining_threshold:
        trend = "Declining"
    else:
        trend = "Stable"

    return {"trend": trend, "predicted_growth_rate": predicted_growth_rate}

# input_data = {
#     "authors": "John Doe",
#     "title": "Deep Learning in Practice",
#     "categories": "cs.AI",
#     "abstract": "Exploring advanced AI techniques.",
#     "year": 2023,
#     "persist": True
# }

# print(predict_trend(input_data))

# DB_FILE = "neurips_arxiv.db"
# def upload_data_in_db():
#     """
#     Uploads CSV files into the database.
#     """
#     with sqlite3.connect(DB_FILE) as conn:
#         neurips_arxiv = pd.read_csv('neurips_arxiv.csv')

#         neurips_arxiv.to_sql('neurips_arxiv', conn, if_exists='replace', index=False)

#         print("Raw data has been uploaded successfully!")
#         conn.commit()

# upload_data_in_db()