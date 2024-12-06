import streamlit as st
import sqlite3

def delete_record(title, authors, year):
    conn = sqlite3.connect('mainData.db')
    cursor = conn.cursor()
    
    # Query to delete record based on title, authors, and year
    query = """
    DELETE FROM Details
    WHERE title = ? AND authors = ? AND year = ?
    """
    cursor.execute(query, (title, authors, year))
    conn.commit()
    conn.close()

def show():
    st.title("Delete Record")

    # Input form to delete a record
    with st.form(key='delete_form'):
        authors = st.text_input("Enter Authors (comma-separated)")
        title = st.text_input("Enter Title of the Paper")
        year = st.number_input("Enter Year of the Paper", min_value=1900, max_value=2100, step=1)
        
        submit_button = st.form_submit_button("Delete Record")
        
        if submit_button:
            if title and authors and year:
                # Call the function to delete the record
                delete_record(title, authors, year)
                st.success(f"Record with Title '{title}' deleted successfully!")
            else:
                st.error("Please provide Title, Authors, and Year to delete the record.")
