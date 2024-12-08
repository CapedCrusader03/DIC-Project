import streamlit as st
# import Kshitij_Hypo_1 as kk
# import shakyahypo1 as shakya_badmaash
import sqlite3
import pandas as pd
import re  # Import the re module
from sklearn.cluster import KMeans
from sklearn.preprocessing import MultiLabelBinarizer
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.ensemble import RandomForestRegressor
from pydantic import BaseModel

from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import re
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
from nltk.stem import PorterStemmer
import itertools
import numpy as np

st.set_page_config(
    page_title="Study on Research papers published in NeurIPS and ArXiv",
    page_icon=":book:",
    layout="centered",
    initial_sidebar_state="auto"
)

import os

# Function to establish connection to the SQLite database
@st.cache_resource
def get_database_connection(db_file: str):
    """
    Establish and cache the SQLite database connection.
    """
    return sqlite3.connect(db_file, check_same_thread=False)

# Function to check if a table exists
def table_exists(conn, table_name: str):
    """
    Check if a table exists in the SQLite database.
    """
    query = f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';"
    result = conn.execute(query).fetchone()
    return result is not None

# Function to load CSV into the SQLite database
def load_csv_to_db(csv_file: str, conn, table_name: str):
    """
    Load a CSV file into a SQLite database table.
    """
    try:
        # Read the CSV file into a Pandas DataFrame
        df = pd.read_csv(csv_file)
        # Load the DataFrame into the SQLite table
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        st.write(f"CSV file '{csv_file}' loaded into database as table '{table_name}'.")
    except Exception as e:
        st.error(f"Error loading CSV into database: {e}")

# Main app logic
st.title("A study on research papers published in NeurIPS and ArXiv.")

DB_FILE = "neurips_arxiv.db"  # SQLite database file
CSV_FILE = "neurips_arxiv.csv"       # CSV file to load
TABLE_NAME = "neurips_arxiv"   # Table name for the CSV data

# Establish database connection
conn = get_database_connection(DB_FILE)

# Check and load CSV into the database
if not table_exists(conn, TABLE_NAME):
    st.write(f"Table '{TABLE_NAME}' not found in database. Loading CSV...")
    load_csv_to_db(CSV_FILE, conn, TABLE_NAME)
else:
    st.write(f"Table '{TABLE_NAME}' already exists in the database.")

# Example usage: Show the first few rows of the table
if table_exists(conn, TABLE_NAME):
    query = f"SELECT * FROM {TABLE_NAME} LIMIT 5;"
    df = pd.read_sql_query(query, conn)
    st.write("Preview of the data:", df)


# Define a function to load each page
def load_page(page_name):
    if page_name == "Add":
        # Import and run the add_page.py
        import add_page
        add_page.show()
    elif page_name == "Delete":
        # Import and run the delete_page.py
        import delete_page
        delete_page.show()
    elif page_name == "Modify":
        # Import and run the modify_page.py
        import modify_page
        modify_page.show()
    elif page_name=="View":
        import lookUp_page
        lookUp_page.show()

# Dropdown for operations
operation = st.selectbox(
    "Choose an operation to perform on the database:",
    ["View", "Add", "Delete", "Modify"]
)

# Load the corresponding page based on selection
if operation:
    load_page(operation)

# Hypothesis testing section (no functionality added yet)
st.title("Hypothesis Testing Questions")
questions = [
    "Majority of authors publish research articles in a single specialized field, but a section of authors engages in interdisciplinary work.",
    "How is the research paper trending over time? Is it growing, declining, or remaining stable?",
    "Research growth in the listed category for the last decade vs previous year",
    "Research Growth between interconnected category in last decade"
]

def generate_graph_data():
    # Connect to the SQLite database
    conn = sqlite3.connect(DB_FILE)
    query = "SELECT title, authors, categories, abstract, year FROM neurips_arxiv"
    df = pd.read_sql(query, conn)
    conn.close()

    # Preprocess data
    df['simplified_category'] = df['categories'].str.split('-').str[0].str.split('.').str[0]
    df['author_count'] = df['authors'].apply(lambda x: len(re.split(r',| and ', x)) if isinstance(x, str) else 0)
    df['authors'] = df['authors'].str.split(r',| and ')

    # Explode data for authors and categories (fix the function name here)
    df_exploded = df.explode('authors')
    df_exploded['authors'] = df_exploded['authors'].str.strip()

    # Select top categories
    top_categories = (
        df.explode('simplified_category')
        .groupby('simplified_category')['author_count']
        .sum()
        .nlargest(10)
        .index
    )
    df_exploded = df_exploded[df_exploded['simplified_category'].isin(top_categories)]

    # Process author-category mapping
    author_categories = df_exploded.groupby('authors')['simplified_category'].apply(set)
    mlb = MultiLabelBinarizer()
    author_category_matrix = pd.DataFrame(
        mlb.fit_transform(author_categories),
        index=author_categories.index,
        columns=mlb.classes_
    )

    # KMeans clustering
    kmeans = KMeans(n_clusters=5, random_state=0)
    author_category_matrix['Cluster'] = kmeans.fit_predict(author_category_matrix)

    # Generate heatmap
    plt.figure(figsize=(10, 6))
    heatmap_data = author_category_matrix.groupby('Cluster').mean()
    sns.heatmap(heatmap_data, cmap="YlGnBu", annot=True)
    plt.title("Average Category Involvement per Cluster (Top Categories by Author Count)")
    plt.xlabel("Simplified Category")
    plt.ylabel("Cluster")

    # Streamlit-compatible plot
    st.pyplot(plt.gcf())  # Return the figure for display in Streamlit

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

db_config = "neurips_arxiv.db"

def create_entry(table, data):
    with sqlite3.connect(db_config) as conn:
        columns = ", ".join(data.keys())
        placeholders = ", ".join("?" * len(data))
        sql = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
        conn.execute(sql, list(data.values()))
        conn.commit()

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

nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

stemmer = PorterStemmer()
stopwords = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

import warnings
warnings.filterwarnings("ignore")

tfidf = TfidfVectorizer(ngram_range=(2,2))

arxiv_taxonomy_mapping = {
    "cs": "Computer Science",
    "econ": "Economics",
    "eess": "Electrical Engineering and Systems Science",
    "math": "Mathematics",
    "astro-ph": "Astrophysics",
    "cond-mat": "Condensed Matter",
    "gr-qc": "General Relativity and Quantum Cosmology",
    "hep-ex": "High Energy Physics - Experiment",
    "hep-lat": "High Energy Physics - Lattice",
    "hep-ph": "High Energy Physics - Phenomenology",
    "hep-th": "High Energy Physics - Theory",
    "math-ph": "Mathematical Physics",
    "nlin": "Nonlinear Sciences",
    "nucl-ex": "Nuclear Experiment",
    "nucl-th": "Nuclear Theory",
    "physics": "Physics",
    "quant-ph": "Quantum Physics",
    "q-bio": "Quantitative Biology",
    "q-fin": "Quantitative Finance",
    "stat": "Statistics",
    "cs.AI": "Artificial Intelligence",
    "cs.AR": "Architecture",
    "cs.CC": "Computational Complexity",
    "cs.CE": "Computational Engineering, Finance, and Science",
    "cs.CG": "Computational Geometry",
    "cs.CL": "Computation and Language",
    "cs.CR": "Cryptography and Security",
    "cs.CV": "Computer Vision and Pattern Recognition",
    "cs.CY": "Computational Biology and Bioinformatics",
    "cs.DB": "Databases",
    "cs.DC": "Distributed, Parallel, and Cluster Computing",
    "cs.DL": "Digital Libraries",
    "cs.DM": "Discrete Mathematics",
    "cs.DS": "Data Structures and Algorithms",
    "cs.ET": "Emerging Technologies",
    "cs.FL": "Formal Languages and Automata Theory",
    "cs.GL": "General Literature",
    "cs.GR": "Graphics",
    "cs.GT": "Game Theory",
    "cs.HC": "Human-Computer Interaction",
    "cs.IR": "Information Retrieval",
    "cs.IT": "Information Theory",
    "cs.LG": "Machine Learning",
    "cs.LO": "Logic in Computer Science",
    "cs.MA": "Multimedia",
    "cs.MM": "Multimedia Systems",
    "cs.MS": "Mathematical Software",
    "cs.NA": "Numerical Analysis",
    "cs.NE": "Neural and Evolutionary Computing",
    "cs.NI": "Networking and Internet Architecture",
    "cs.OH": "Other Computer Science",
    "cs.OS": "Operating Systems",
    "cs.PF": "Performance",
    "cs.PL": "Programming Languages",
    "cs.RO": "Robotics",
    "cs.SC": "Scientific Computing",
    "cs.SD": "Social and Information Networks",
    "cs.SE": "Software Engineering",
    "cs.SI": "Social Computing",
    "cs.SY": "Systems and Control",
    "econ.EM": "Econometrics",
    "econ.GN": "General Economics",
    "econ.TH": "Theoretical Economics",
    "eess.AS": "Audio and Speech Processing",
    "eess.IV": "Image and Video Processing",
    "eess.SP": "Signal Processing",
    "eess.SY": "Systems and Control",
    "math.AC": "Algorithmic Combinatorics",
    "math.AG": "Algebraic Geometry",
    "math.AP": "Analysis of PDEs",
    "math.AT": "Algebraic Topology",
    "math.CA": "Classical Analysis and ODEs",
    "math.CO": "Combinatorics",
    "math.CT": "Category Theory",
    "math.CV": "Complex Variables",
    "math.DG": "Differential Geometry",
    "math.DS": "Dynamical Systems",
    "math.FA": "Functional Analysis",
    "math.GM": "General Mathematics",
    "math.GN": "General Topology",
    "math.GR": "Group Theory",
    "math.GT": "Geometric Topology",
    "math.HO": "History and Overview",
    "math.IT": "Information Theory",
    "math.KT": "K-Theory and Homology",
    "math.LO": "Logic",
    "math.MG": "Mathematical Physics",
    "math.MP": "Mathematical Programming",
    "math.NA": "Numerical Analysis",
    "math.NT": "Number Theory",
    "math.OA": "Operator Algebras",
    "math.OC": "Optimization and Control",
    "math.PR": "Probability",
    "math.QA": "Quantum Algebra",
    "math.RA": "Rings and Algebras",
    "math.RT": "Representation Theory",
    "math.SG": "Symplectic Geometry",
    "math.SP": "Spectral Theory",
    "math.ST": "Statistics Theory",
    "astro-ph.CO": "Cosmology and Nongalactic Astrophysics",
    "astro-ph.EP": "Earth and Planetary Astrophysics",
    "astro-ph.GA": "Galaxy Astrophysics",
    "astro-ph.HE": "High Energy Astrophysical Phenomena",
    "astro-ph.IM": "Instrumentation and Methods for Astrophysics",
    "astro-ph.SR": "Solar and Stellar Astrophysics",
    "cond-mat.dis-nn": "Disordered Systems and Neural Networks",
    "cond-mat.mes-hall": "Mesoscale and Nanoscale Physics",
    "cond-mat.mtrl-sci": "Materials Science",
    "cond-mat.other": "Other Condensed Matter",
    "cond-mat.quant-gas": "Quantum Gases",
    "cond-mat.soft": "Soft Condensed Matter",
    "cond-mat.stat-mech": "Statistical Mechanics",
    "cond-mat.str-el": "Strongly Correlated Electrons",
    "cond-mat.supr-con": "Superconductivity",
    "nlin.AO": "Adaptation and Self-Organizing Systems",
    "nlin.CD": "Cellular Automata and Lattice Gases",
    "nlin.CG": "Chaos and Nonlinear Dynamics",
    "nlin.PS": "Pattern Formation and Solitons",
    "nlin.SI": "Statistical Mechanics",
    "physics.acc-ph": "Accelerator Physics",
    "physics.ao-ph": "Atmospheric and Oceanic Physics",
    "physics.app-ph": "Applied Physics",
    "physics.atm-clus": "Atomic and Molecular Clusters",
    "physics.atom-ph": "Atomic Physics",
    "physics.bio-ph": "Biological Physics",
    "physics.chem-ph": "Chemical Physics",
    "physics.class-ph": "Classical Physics",
    "physics.comp-ph": "Computational Physics",
    "physics.data-an": "Data Analysis, Statistics, and Probability",
    "physics.ed-ph": "Physics Education",
    "physics.flu-dyn": "Fluid Dynamics",
    "physics.gen-ph": "General Physics",
    "physics.geo-ph": "Geophysics",
    "physics.hist-ph": "History and Philosophy of Physics",
    "physics.ins-det": "Instrumentation and Detectors",
    "physics.med-ph": "Medical Physics",
    "physics.optics": "Optics",
    "physics.plasm-ph": "Plasma Physics",
    "physics.pop-ph": "Popular Physics",
    "physics.soc-ph": "Society and Physics",
    "physics.space-ph": "Space Physics",
    "q-bio.BM": "Biomolecules",
    "q-bio.CB": "Cell Behavior",
    "q-bio.GN": "Genomics",
    "q-bio.MN": "Molecular Networks",
    "q-bio.NC": "Neurons and Cognition",
    "q-bio.OT": "Other Quantitative Biology",
    "q-bio.PE": "Populations and Evolution",
    "q-bio.QM": "Quantitative Methods",
    "q-bio.SC": "Subcellular Processes",
    "q-bio.TO": "Tissues and Organs",
    "q-fin.CP": "Computational Finance",
    "q-fin.EC": "Economics",
    "q-fin.GN": "General Finance",
    "q-fin.MF": "Mathematical Finance",
    "q-fin.PM": "Portfolio Management",
    "q-fin.PR": "Pricing of Financial Instruments",
    "q-fin.RM": "Risk Management",
    "q-fin.ST": "Statistical Finance",
    "q-fin.TR": "Trading and Market Microstructure",
    "stat.AP": "Applications",
    "stat.CO": "Computation",
    "stat.ME": "Methodology",
    "stat.ML": "Machine Learning",
    "stat.OT": "Other Statistics",
    "stat.TH": "Theory"
}


db_config = "neurips_arxiv.db"
cnx = sqlite3.connect(db_config)


def get_data(query):
    cur = cnx.cursor()
    arxiv_df = pd.read_sql(query,cnx)
    return arxiv_df

def get_query(cat_list):
    cond = " OR ".join([ f"categories LIKE '%{cat}%'" for cat in cat_list])
    query = f"""select * from neurips_arxiv nad
    where  
    {cond};"""
    return query

#to return unique categories from a df
def get_all_unique_cat(filtered_cat_df):
    unique_cat = set()
    for cat in filtered_cat_df["categories"].to_list():
        unique_cat.update(cat.split(" "))
        
    return unique_cat

## Pandas apply function : to find category present in list
def is_present(cat_list , curr_cat):
    if curr_cat in cat_list.split(" "):
        return True
    return False 

# Add status of each category as individual column
def add_category_status(unique_cat , all_df , cat_list):
    for col in cat_list:
        all_df[f"cat_{col}"] = all_df["categories"].apply(is_present,args=(col , ))
    return all_df
    
# add column of each category
def filtered_category_to_column(all_data , cat_list):
    unique_cat = get_all_unique_cat(all_data)
    all_data = add_category_status(unique_cat , all_data , cat_list)
    return all_data

def get_slope(year_count_df , col):
    slope, _ = np.polyfit(year_count_df.index, year_count_df[col].to_list() , 1)
    return round(slope.item() , 4)


def get_plot(arxiv_df , cat_list):
    growth_list = []

    fig, ax = plt.subplots(figsize=(15,8))

    for sd_cat in cat_list:
        col_name = f"cat_{sd_cat}"
        cat_count_df = arxiv_df[arxiv_df[col_name]][[col_name, "year"]].groupby("year").count()
        plt.plot(cat_count_df.index , np.log(cat_count_df[col_name].to_list()) , label = arxiv_taxonomy_mapping[sd_cat] )
        prev_slope , last_slope = 0 , 0
        #print(cat_count_df)
        if len(cat_count_df)  > 0:
            before_df = cat_count_df[cat_count_df.index < 2014]
            after_df = cat_count_df[cat_count_df.index >= 2014]
            before_2014 = 0 if len(before_df) == 0 else get_slope(before_df , col_name)
            after_2014 = 0 if len(after_df) == 0 else get_slope(after_df , col_name)
        growth_list.append(( arxiv_taxonomy_mapping[sd_cat] , round(before_2014,2) , round(after_2014,2)))
    
    ax.axvline(x=2014, color='r', linestyle='--', linewidth=2) 
    custom_ticks = [ year for year in range(2007 , 2025)]       # Positions for the ticks
    custom_labels = [ str(year) for year in range(2007 , 2025)]
    ax.set_xticks(ticks=custom_ticks, labels=custom_labels)
    ax.set_xlabel("Year")
    ax.set_ylabel("log(number of research)")
    ax.set_title("Growth of research in sub domain of self driving car with respect to Year")
    ax.grid()
    ax.legend()
    return growth_list , fig

def get_top_subdomain(growth_list):
    growth_df = pd.DataFrame(growth_list , columns=["category" , "before14" , "after14"])
    growth_df.replace(0,1 , inplace=True)
    growth_df["growth_factor"] = round(growth_df["after14"]/growth_df["before14"] , 2)
    growth_df = growth_df[["category" , "growth_factor"]].sort_values("growth_factor" ,ascending=False)
    return growth_df 


def get_hypo1(cat_list):
    query = get_query(cat_list)
    arxiv_df = get_data(query)
    arxiv_df = filtered_category_to_column(arxiv_df , cat_list)
    growth_list , fig = get_plot(arxiv_df , cat_list)
    top_domain_df = get_top_subdomain(growth_list)
    
    return top_domain_df , fig




def get_sorted_edges(arxiv_df , cat_list):
    pairs = list(itertools.combinations(cat_list,3 ))

    edges = [] 
    for n1 , n2 , n3 in pairs:
        col1 = f"cat_{n1}"
        col2 = f"cat_{n2}"
        col3 = f"cat_{n3}"
        collab_after_2014 = sum(arxiv_df[(arxiv_df[col1]) & (arxiv_df[col2]) & (arxiv_df[col3])  & (arxiv_df["year"] >= 2014)][[col1  , "year" ]].groupby("year").count()[col1].to_list())
        edges.append([n1, n2 , n3 , round(collab_after_2014,5)])
        
    sorted_edges = sorted( edges , key=lambda x : x[3] ,reverse= True )[:10]
    return sorted_edges

def get_interplot(arxiv_df , sorted_edges):
    edges = []
    fig, ax = plt.subplots(figsize=(20,8))
    for n1 , n2 , n3 , weight in sorted_edges:
        col1 = f"cat_{n1}"
        col2 = f"cat_{n2}"
        col3 = f"cat_{n3}"
        edge_df = arxiv_df[ (arxiv_df[col1]) & (arxiv_df[col2]) & (arxiv_df[col3]) ][[col1  , "year" ]].groupby("year").count()
        #edges.append( (f"{n1}-{n2}" , edge_df[col1].sum().item() ) )
        plt.plot(edge_df.index ,
                np.log(edge_df[col1].to_list()),
                label = f"{arxiv_taxonomy_mapping[n1]}|{arxiv_taxonomy_mapping[n2]}|{arxiv_taxonomy_mapping[n3]}" )
        
    ax.axvline(x=2014, color='r', linestyle='--', linewidth=2) 
    custom_ticks = [ year for year in range(2007 , 2025)]       # Positions for the ticks
    custom_labels = [ str(year) for year in range(2007 , 2025)]
    ax.set_xticks(ticks=custom_ticks, labels=custom_labels)
    ax.set_xlabel("Year")
    ax.set_ylabel("log(number of research)")
    ax.set_title("Growth of interdisciplinary research in sub domain of self driving car with respect to Year")
    ax.grid()
    ax.legend()
    return fig

def preprocessing_text_col(df ):
    df = df.str.lower()
    df = df.apply(lambda x: re.sub(r'[^\w\s]', '', x))
    df = df.apply(nltk.word_tokenize)
    df = df.apply(lambda x: [word for word in x if word not in stopwords])
    df = df.apply(lambda x: [lemmatizer.lemmatize(word ) for word in x])
    #df = df.apply(lambda x: [stemmer.stem(word) for word in x])
    df = df.apply(lambda x : " ".join(x))
    return df

def get_tfidf_scores(df ,  n1 , n2 ,n3 , start_year , end_year):
    col1 = f"cat_{n1}"
    col2 = f"cat_{n2}"
    col3 = f"cat_{n3}"
    filtered_titles = df[
        (df[col1] == True)  &
        (df[col2] == True) &
        (df[col3] == True) &
        (df["year"] <= end_year) &
        (df["year"] >= start_year) 
        ]["title"]
    #print(filtered_titles)
    title_clean = preprocessing_text_col(filtered_titles).to_list()
    #print(title_clean)
    if len(title_clean) == 0:
        title_clean = ["THis is dummuy value" , " this is not dummy" ]
    result = tfidf.fit_transform(title_clean)
    tfidf_df = pd.DataFrame(result.toarray(), columns=tfidf.get_feature_names_out())
    tfidf_scores = tfidf_df.sum().sort_values(ascending=False)
    return tfidf_scores

def get_topic_slope(df , n1 , n2 , n3):
    first_df = get_tfidf_scores(df , n1 , n2 ,n3 , 2014 , 2019)
    
    second_df = get_tfidf_scores(df ,  n1 , n2 ,n3  , 2020 , 2024)
    #year21_24_df = get_tfidf_scores(df ,  n1 , n2 ,n3  , 2021 , 2024)
    s1 = set(first_df[first_df > 1].index)
    s2 = set(second_df[second_df > 1].index)
    #s3 = set(year21_24_df[year21_24_df > 1].index)
    topic_with_slope = []
    for topic in s1.intersection(s2):
        topic_with_slope.append(
            [
                topic,
                first_df[topic],
                second_df[topic],
                #year21_24_df[topic]
            ]
        )
    return pd.DataFrame(topic_with_slope , columns=["topic" , "s1" , "s2" ])

def get_inter_domain_trend(arxiv_df ,  sorted_edges):

    inter_domain_trend = {}
    
    for n1 , n2 , n3 , w in sorted_edges:
        key = f" {arxiv_taxonomy_mapping[n1]} || {arxiv_taxonomy_mapping[n2]} || {arxiv_taxonomy_mapping[n3]}"
        slope_df = get_topic_slope(arxiv_df, n1, n2 , n3)
        slope_df["growth_factor"] = slope_df["s2"]/slope_df["s1"]
        trend_topics = slope_df.sort_values(by = "growth_factor" , ascending=False)["topic"].to_list()
        inter_domain_trend[key] = trend_topics
        
    return inter_domain_trend

def get_hypo2(cat_list):
    query = get_query(cat_list)
    arxiv_df = get_data(query)
    arxiv_df = filtered_category_to_column(arxiv_df , cat_list)
    sorted_edges = get_sorted_edges(arxiv_df , cat_list)
    fig = get_interplot(arxiv_df , sorted_edges)
    inter_domain_trend = get_inter_domain_trend(arxiv_df ,  sorted_edges)
    return inter_domain_trend , fig

# cat_list = ["cs.CV","cs.RO","cs.LG","cs.SY","cs.HC","eess.SY","eess.SP"]


# Create buttons and execute corresponding functions
st.markdown(f"### Hypothesis {1}: {questions[0]}")
if st.button("Get results for Hypothesis 1"):
    result = generate_graph_data()
    st.write(result)

st.markdown(f"### Hypothesis {2}: {questions[1]}")

# Define the function that processes the input
def process_input(input_data):
    if input_data["persist"]:
        return f"Data will be saved to the database and the model will be trained on it: {input_data}"
    else:
        return f"Data will not be saved. Using the pre-trained model for this input: {input_data}"

# Streamlit app layout
st.markdown("#### Input Data for Model")

# Create input fields
authors = st.text_input("Authors:", placeholder="Enter author names (e.g., John Doe)")
title = st.text_input("Title:", placeholder="Enter the paper title")
categories = st.text_input("Categories:", placeholder="Enter the category (e.g., cs.AI)")
abstract = st.text_area("Abstract:", placeholder="Enter the abstract")
year = st.number_input("Year:", min_value=1900, max_value=2100, value=2024)

# Choice for persist
persist = st.radio(
    "Do you want to save this data to the database and train the model?",
    options=["Yes, save and train", "No, just use pre-trained model"],
    index=1
)

# Convert user's choice to Boolean
persist_flag = True if persist == "Yes, save and train" else False

# Create dictionary
input_data = {
    "authors": authors,
    "title": title,
    "categories": categories,
    "abstract": abstract,
    "year": year,
    "persist": persist_flag
}

# Button to submit the data
if st.button("Submit and Get results for Hypothesis 2"):
    if not authors or not title or not categories or not abstract:
        st.error("Please fill in all the fields before submitting!")
    else:
        result = predict_trend(input_data)
        st.write(result)


st.markdown(f"### Hypothesis {3}: {questions[2]}")

# Categorized dictionary of arXiv taxonomy
arxiv_taxonomy_mapping_format = {
    "Computer Science": {
        "cs.AI": "Artificial Intelligence",
        "cs.AR": "Architecture",
        "cs.CC": "Computational Complexity",
        "cs.CE": "Computational Engineering, Finance, and Science",
        "cs.CG": "Computational Geometry",
        "cs.CL": "Computation and Language",
        "cs.CR": "Cryptography and Security",
        "cs.CV": "Computer Vision and Pattern Recognition",
        "cs.CY": "Computational Biology and Bioinformatics",
        "cs.DB": "Databases",
        "cs.DC": "Distributed, Parallel, and Cluster Computing",
        "cs.DL": "Digital Libraries",
        "cs.DM": "Discrete Mathematics",
        "cs.DS": "Data Structures and Algorithms",
        "cs.ET": "Emerging Technologies",
        "cs.FL": "Formal Languages and Automata Theory",
        "cs.GL": "General Literature",
        "cs.GR": "Graphics",
        "cs.GT": "Game Theory",
        "cs.HC": "Human-Computer Interaction",
        "cs.IR": "Information Retrieval",
        "cs.IT": "Information Theory",
        "cs.LG": "Machine Learning",
        "cs.LO": "Logic in Computer Science",
        "cs.MA": "Multimedia",
        "cs.MM": "Multimedia Systems",
        "cs.MS": "Mathematical Software",
        "cs.NA": "Numerical Analysis",
        "cs.NE": "Neural and Evolutionary Computing",
        "cs.NI": "Networking and Internet Architecture",
        "cs.OH": "Other Computer Science",
        "cs.OS": "Operating Systems",
        "cs.PF": "Performance",
        "cs.PL": "Programming Languages",
        "cs.RO": "Robotics",
        "cs.SC": "Scientific Computing",
        "cs.SD": "Social and Information Networks",
        "cs.SE": "Software Engineering",
        "cs.SI": "Social Computing",
        "cs.SY": "Systems and Control"
    },
    "Economics": {"econ.EM": "Econometrics",
        "econ.GN": "General Economics",
        "econ.TH": "Theoretical Economics"
    },
    "Electrical Engineering and Systems Science": {
        "eess.AS": "Audio and Speech Processing",
        "eess.IV": "Image and Video Processing",
        "eess.SP": "Signal Processing",
        "eess.SY": "Systems and Control",
    },
    "Astrophysics": {
         "astro-ph.CO": "Cosmology and Nongalactic Astrophysics",
        "astro-ph.EP": "Earth and Planetary Astrophysics",
        "astro-ph.GA": "Galaxy Astrophysics",
        "astro-ph.HE": "High Energy Astrophysical Phenomena",
        "astro-ph.IM": "Instrumentation and Methods for Astrophysics",
        "astro-ph.SR": "Solar and Stellar Astrophysics",
    },
    "Condensed Matter":{
        "cond-mat.dis-nn": "Disordered Systems and Neural Networks",
        "cond-mat.mes-hall": "Mesoscale and Nanoscale Physics",
        "cond-mat.mtrl-sci": "Materials Science",
        "cond-mat.other": "Other Condensed Matter",
        "cond-mat.quant-gas": "Quantum Gases",
        "cond-mat.soft": "Soft Condensed Matter",
        "cond-mat.stat-mech": "Statistical Mechanics",
        "cond-mat.str-el": "Strongly Correlated Electrons",
        "cond-mat.supr-con": "Superconductivity",
    },
    "Nonlinear Sciences": {
        "nlin.AO": "Adaptation and Self-Organizing Systems",
        "nlin.CD": "Cellular Automata and Lattice Gases",
        "nlin.CG": "Chaos and Nonlinear Dynamics",
        "nlin.PS": "Pattern Formation and Solitons",
        "nlin.SI": "Statistical Mechanics",
    },
    "Physics": {
        "physics.acc-ph": "Accelerator Physics",
        "physics.ao-ph": "Atmospheric and Oceanic Physics",
        "physics.app-ph": "Applied Physics",
        "physics.atm-clus": "Atomic and Molecular Clusters",
        "physics.atom-ph": "Atomic Physics",
        "physics.bio-ph": "Biological Physics",
        "physics.chem-ph": "Chemical Physics",
        "physics.class-ph": "Classical Physics",
        "physics.comp-ph": "Computational Physics",
        "physics.data-an": "Data Analysis, Statistics, and Probability",
        "physics.ed-ph": "Physics Education",
        "physics.flu-dyn": "Fluid Dynamics",
        "physics.gen-ph": "General Physics",
        "physics.geo-ph": "Geophysics",
        "physics.hist-ph": "History and Philosophy of Physics",
        "physics.ins-det": "Instrumentation and Detectors",
        "physics.med-ph": "Medical Physics",
        "physics.optics": "Optics",
        "physics.plasm-ph": "Plasma Physics",
        "physics.pop-ph": "Popular Physics",
        "physics.soc-ph": "Society and Physics",
        "physics.space-ph": "Space Physics",
    },
    "Quantitative Biology": {
        "q-bio.BM": "Biomolecules",
        "q-bio.CB": "Cell Behavior",
        "q-bio.GN": "Genomics",
        "q-bio.MN": "Molecular Networks",
        "q-bio.NC": "Neurons and Cognition",
        "q-bio.OT": "Other Quantitative Biology",
        "q-bio.PE": "Populations and Evolution",
        "q-bio.QM": "Quantitative Methods",
        "q-bio.SC": "Subcellular Processes",
        "q-bio.TO": "Tissues and Organs",
    },
    "Quantitative Finance": {
        "q-fin.CP": "Computational Finance",
        "q-fin.EC": "Economics",
        "q-fin.GN": "General Finance",
        "q-fin.MF": "Mathematical Finance",
        "q-fin.PM": "Portfolio Management",
        "q-fin.PR": "Pricing of Financial Instruments",
        "q-fin.RM": "Risk Management",
        "q-fin.ST": "Statistical Finance",
        "q-fin.TR": "Trading and Market Microstructure",
    },
    "Statistics": {
        "stat.AP": "Applications",
        "stat.CO": "Computation",
        "stat.ME": "Methodology",
        "stat.ML": "Machine Learning",
        "stat.OT": "Other Statistics",
        "stat.TH": "Theory",
    },
    "Mathematics": {
        "math.AC": "Algorithmic Combinatorics",
        "math.AG": "Algebraic Geometry",
        "math.AP": "Analysis of PDEs",
        "math.AT": "Algebraic Topology",
        "math.CA": "Classical Analysis and ODEs",
        "math.CO": "Combinatorics",
        "math.CT": "Category Theory",
        "math.CV": "Complex Variables",
        "math.DG": "Differential Geometry",
        "math.DS": "Dynamical Systems",
        "math.FA": "Functional Analysis",
        "math.GM": "General Mathematics",
        "math.GN": "General Topology",
        "math.GR": "Group Theory",
        "math.GT": "Geometric Topology",
        "math.HO": "History and Overview",
        "math.IT": "Information Theory",
        "math.KT": "K-Theory and Homology",
        "math.LO": "Logic",
        "math.MG": "Mathematical Physics",
        "math.MP": "Mathematical Programming",
        "math.NA": "Numerical Analysis",
        "math.NT": "Number Theory",
        "math.OA": "Operator Algebras",
        "math.OC": "Optimization and Control",
        "math.PR": "Probability",
        "math.QA": "Quantum Algebra",
        "math.RA": "Rings and Algebras",
        "math.RT": "Representation Theory",
        "math.SG": "Symplectic Geometry",
        "math.SP": "Spectral Theory",
        "math.ST": "Statistics Theory",
    },
}

# Helper function to flatten categories into a list of options
def get_flattened_options(category_dict):
    flat_list = []
    for key, value in category_dict.items():
        if isinstance(value, dict):  # If value is a subcategory
            for sub_key, sub_value in value.items():
                flat_list.append((f"{key}.{sub_key}", sub_value))
        else:
            flat_list.append((key, value))
    return flat_list

# Flattened options for the dropdown
flattened_options = get_flattened_options(arxiv_taxonomy_mapping_format)

# User-friendly dropdown grouping
categories_by_domain = {}
for key, value in arxiv_taxonomy_mapping_format.items():
    if isinstance(value, dict):  # Subcategories
        categories_by_domain[key] = list(value.items())
    else:
        categories_by_domain[key] = [(key, value)]

# Streamlit app layout
st.markdown("#### Select Categories for Model Input")

# Dropdowns for categories
selected_categories = []
for domain, options in categories_by_domain.items():
    with st.expander(f"{domain.upper()} ({arxiv_taxonomy_mapping.get(domain, domain)})"):
        selected = st.multiselect(
            f"Select categories under {domain.upper()}:",
            options=[key for key, _ in options],
            format_func=lambda x: dict(options).get(x, x),
            key=f"hypo1_{domain}"
        )
        selected_categories.extend(selected)

# Ensure at least 3 distinct selections
if st.button("Submit and Get results for Hypothesis 3"):
    if len(selected_categories) < 3:
        st.error("Please select at least 3 distinct categories.")
    else:
        st.success(f"Selected categories: {selected_categories}")
        result = get_hypo1(selected_categories)
        df, graph = result
        st.dataframe(df)
        st.pyplot(graph)


# Streamlit app layout
st.markdown(f"### Hypothesis {4}: {questions[3]}")
st.markdown("#### Select Categories for Model Input")

# Dropdowns for categories
selected_categories = []
for domain, options in categories_by_domain.items():
    with st.expander(f"{domain.upper()} ({arxiv_taxonomy_mapping.get(domain, domain)})"):
        selected = st.multiselect(
            f"Select categories under {domain.upper()}:",
            options=[key for key, _ in options],
            format_func=lambda x: dict(options).get(x, x),
            key=f"hypo2_{domain}"
        )
        selected_categories.extend(selected)

# Ensure at least 3 distinct selections
if st.button("Submit and Get results for Hypothesis 4"):
    if len(selected_categories) < 3:
        st.error("Please select at least 3 distinct categories.")
    else:
        st.success(f"Selected categories: {selected_categories}")
        result = get_hypo2(selected_categories)
        df, graph = result
        st.dataframe(df)
        st.pyplot(graph)
