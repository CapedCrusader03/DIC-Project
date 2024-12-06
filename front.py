import streamlit as st
import Kshitij_Hypo_1 as kk
import shakyahypo1 as shakya_badmaash
# Set Streamlit page config
st.set_page_config(page_title="Interactive Questions App", layout="wide")

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

# Dropdown for operations
operation = st.selectbox(
    "Choose an operation to perform on the database:",
    ["", "Add", "Delete", "Modify"]
)

# Load the corresponding page based on selection
if operation:
    load_page(operation)

# Hypothesis testing section (no functionality added yet)
st.markdown("## Hypothesis Testing Questions")
questions = [
    "Majority of authors publish research articles in a single specialized field, but a section of authors engages in interdisciplinary work.",
    "How is the research paper trending over time? Is it growing, declining, or remaining stable?",
    "Research growth in the listed category for the last decade vs previous year",
    "What is your favorite hobby?"
]

# Display hypotheses
for i, question in enumerate(questions, start=1):
    st.markdown(f"### Hypothesis {i}: {question}")
    if st.button(f"Show Result for Hypothesis {i}"):
        if i == 2:
            input_data = {
                "authors": "John Doe",
                "title": "Deep Learning in Practice",
                "categories": "cs.AI",
                "abstract": "Exploring advanced AI techniques.",
                "year": 2023,
                "persist": True
            }
            res = kk.predict_trend(input_data)
            if res:
                st.success("Prediction Successful!")
                st.write(f"**Trend:** {res['trend']}")
                st.write(f"**Predicted Growth Rate:** {res['predicted_growth_rate']:.2f}%")
            else:
                st.error("Error running the prediction!")
        if i==3:
            cat_list = ["cs.CV","cs.RO","cs.LG","cs.SY","cs.HC","eess.SY","eess.SP"]
            res = shakya_badmaash.get_hypo1(cat_list)
            if res:
                print(res)
                # st.dataframe(res["top_domain_df"])
                # st.write(res['top_domain_df'])
                st.plotly_chart(res['fig'], use_container_width=True)
        if i==4:
            cat_list = ["cs.CV","cs.RO","cs.LG","cs.SY","cs.HC","eess.SY","eess.SP"]
            res = shakya_badmaash.get_hypo2(cat_list)
            if res:
                print(res)
                # st.write(res["inter_domain_trend"])
                st.plotly_chart(res["fig"], use_container_width=True)
        st.write(f"The result for Hypothesis {i} will be displayed here.")


