# DIC-Project
Extracting hot trending research topics from the NeurIPS and ArXiv publication dataset in a given period.

| Name        | Email ID           | UB Number  |
| ------------- |:-------------:| -----:|
| Kshitij Kumar      | kkumar8@buffalo.edu | 50610480 |
| Amritesh Kuraria      | akuraria@buffalo.edu      |   50598180 |
| Shubham Shubham | shubham@buffalo.edu      | 50596116    |

Link to the dataset: https://drive.google.com/file/d/1BXNsl8h2SQzQJdsYJM7g7GEhTm0P50cJ/view?usp=sharing


# Research Analysis Web App

This repository contains the source code for a Streamlit-based web application that allows users to perform CRUD operations (Create, Read, Update, Delete) on a research database and analyze data based on predefined hypotheses.

---

## Features

- **View Data:** Using `lookUp.py`, users can view the data in the database.
- **Add Data:** Using `add.py`, users can add new entries to the database.
- **Modify Data:** Using `modify.py`, users can update existing entries in the database.
- **Delete Data:** Using `delete.py`, users can delete entries from the database.
- **Hypothesis Analysis:** The app uses a machine learning model to provide insights into various research hypotheses.

---

## Requirements

Before running the application, ensure you have the following installed:

- Python 3.11 or above
- SQLite3
- All dependencies listed in `requirements.txt`

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/CapedCrusader03/DIC-Project.git
   cd DIC-Project
   ```

2. Install the dependencies:
    ```
    pip install -r requirements.txt
    ```

3. Run the the application:
    ```bash
    streamlit run app.py
    ```

4. Open the app in your web browser (Streamlit will usually open it automatically).

5. Use the intuitive interface to:

    Add, modify, delete, or view database records.
    Analyze data trends and validate hypotheses.
