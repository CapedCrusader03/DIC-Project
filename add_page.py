import streamlit as st
import sqlite3

def add_record(authors, title, categories, abstract, year):
    conn = sqlite3.connect('mainData.db')
    cursor = conn.cursor()
    
    # SQL query to insert a new record into the database
    query = """
    INSERT INTO Details (authors, title, categories, abstract, year)
    VALUES (?, ?, ?, ?, ?)
    """
    cursor.execute(query, (authors, title, categories, abstract, year))
    conn.commit()
    conn.close()

def show():
    st.title("Add New Record")

    # Input form to add a new record
    with st.form(key='add_form'):
        authors = st.text_input("Enter Authors (comma-separated)")
        title = st.text_input("Enter Title of the Paper")
        categories = st.text_input("Enter Categories")
        abstract = st.text_area("Enter Abstract")
        year = st.number_input("Enter Year of the Paper", min_value=1900, max_value=2100, step=1)
        
        submit_button = st.form_submit_button("Add Record")
        
        if submit_button:
            if authors and title and categories and abstract and year:
                # Call the function to add the record
                add_record(authors, title, categories, abstract, year)
                st.success(f"Record with Title '{title}' added successfully!")
            else:
                st.error("Please provide all the required details to add the record.")
