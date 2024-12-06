import streamlit as st
import sqlite3

def search_record(title, author, year):
    conn = sqlite3.connect('mainData.db')
    cursor = conn.cursor()
    
    # SQL query to search for records by title, author, and year
    query = """
    SELECT authors, title, categories, abstract, year 
    FROM Details 
    WHERE (title LIKE ? OR ? = '') 
      AND (authors LIKE ? OR ? = '') 
      AND (year = ? OR ? = 0)
    """
    cursor.execute(query, (f"%{title}%", title, f"%{author}%", author, year, year))
    records = cursor.fetchall()
    conn.close()
    
    return records

def show():
    st.title("Search Records")

    # Input form to search for records
    with st.form(key='search_form'):
        search_title = st.text_input("Enter Title or Part of Title to Search (optional)")
        search_author = st.text_input("Enter Author Name or Part of Name to Search (optional)")
        search_year = st.number_input("Enter Year to Search (optional, leave blank for all)", min_value=0, step=1)
        
        search_button = st.form_submit_button("Search")
        
        if search_button:
            # Call the function to search for records
            results = search_record(search_title, search_author, search_year if search_year > 0 else 0)
            if results:
                st.write(f"Found {len(results)} record(s):")
                for record in results:
                    authors, title, categories, abstract, year = record
                    st.markdown(f"### Title: {title}")
                    st.write(f"- **Authors:** {authors}")
                    st.write(f"- **Categories:** {categories}")
                    st.write(f"- **Abstract:** {abstract}")
                    st.write(f"- **Year:** {year}")
                    st.write("---")
            else:
                st.warning("No records found matching the search criteria.")
