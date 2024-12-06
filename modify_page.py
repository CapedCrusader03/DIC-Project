import streamlit as st
import sqlite3

def modify_record(title, authors, year, new_title, new_authors, new_categories, new_abstract, new_year):
    conn = sqlite3.connect('mainData.db')
    cursor = conn.cursor()
    
    # SQL query to update a record
    query = """
    UPDATE Details
    SET title = ?, authors = ?, categories = ?, abstract = ?, year = ?
    WHERE title = ? AND authors = ? AND year = ?
    """
    cursor.execute(query, (new_title, new_authors, new_categories, new_abstract, new_year, title, authors, year))
    conn.commit()
    conn.close()

def show():
    st.title("Modify Record")

    # Input form to modify a record
    with st.form(key='modify_form'):
        authors = st.text_input("Enter Authors (comma-separated) to modify")
        title = st.text_input("Enter Title of the Paper to modify")
        year = st.number_input("Enter Year of the Paper to modify", min_value=1900, max_value=2100, step=1)
        
        new_title = st.text_input("Enter New Title of the Paper")
        new_authors = st.text_input("Enter New Authors (comma-separated)")
        new_categories = st.text_input("Enter New Categories")
        new_abstract = st.text_area("Enter New Abstract")
        new_year = st.number_input("Enter New Year of the Paper", min_value=1900, max_value=2100, step=1)

        submit_button = st.form_submit_button("Modify Record")
        
        if submit_button:
            if title and authors and year and new_title and new_authors and new_categories and new_abstract and new_year:
                # Call the function to modify the record
                modify_record(title, authors, year, new_title, new_authors, new_categories, new_abstract, new_year)
                st.success(f"Record with Title '{title}' modified successfully!")
            else:
                st.error("Please provide all the required details to modify the record.")
