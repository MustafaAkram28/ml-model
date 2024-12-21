import streamlit as st
import pandas as pd
import pickle

# Load the model and encoders
model = pickle.load(open(r'C:\Users\HP\OneDrive\Desktop\akram\model (2).pkl', 'rb'))
lb1 = pickle.load(open(r'C:\Users\HP\OneDrive\Desktop\akram\lb1 (1).pkl', 'rb'))
lb2 = pickle.load(open(r'C:\Users\HP\OneDrive\Desktop\akram\lb2 (1).pkl', 'rb'))
lb3 = pickle.load(open(r'C:\Users\HP\OneDrive\Desktop\akram\lb3 (1).pkl', 'rb'))
lb4 = pickle.load(open(r'C:\Users\HP\OneDrive\Desktop\akram\lb4 (1).pkl', 'rb'))

# Load the smartphone dataset
data_path = r'C:\Users\HP\OneDrive\Desktop\akram\realistic_smartphones.xlsx'  # Update path as needed
df_smartphones = pd.read_excel(data_path)

# Define feature names in the exact order used during training
feature_names = [
    'Mobile Name', 'Brand', 'OS', 'CPU Performance (Score)',
    'GPU Performance (Score)', 'Camera Quality Rating (out of 5)',
    'Gaming Score (out of 5)', 'Multitasking Score (out of 5)', 'Price Range'
]

# Page configuration
st.set_page_config(page_title="Smartphone Rating Predictor", page_icon="📱", layout="centered")

# Title and description
st.title("Smartphone Rating Predictor")
st.write("Predict customer ratings and get smartphone suggestions based on your inputs!")

# Input fields
mobile_name = st.selectbox("Select Mobile Name", lb1.classes_)
brand = st.selectbox("Select Brand", lb2.classes_)
os = st.selectbox("Select OS", lb3.classes_)
cpu_score = st.number_input("Enter CPU Performance (Score)", min_value=0, max_value=100, step=1, value=50)
gpu_score = st.number_input("Enter GPU Performance (Score)", min_value=0, max_value=100, step=1, value=50)
camera_rating = st.number_input("Enter Camera Quality Rating (out of 5)", min_value=0.0, max_value=5.0, step=0.1, value=3.0)
gaming_score = st.number_input("Enter Gaming Score (out of 5)", min_value=0.0, max_value=5.0, step=0.1, value=3.0)
multitasking_score = st.number_input("Enter Multitasking Score (out of 5)", min_value=0.0, max_value=5.0, step=0.1, value=3.0)
price_range = st.number_input("Enter Price Range (0=Low, 1=Medium, 2=High)", min_value=0, max_value=2, step=1, value=1)

# Predict and suggest
if st.button("Predict Rating"):
    # Encode the inputs
    mobile_name_encoded = lb1.transform([mobile_name])[0]
    brand_encoded = lb2.transform([brand])[0]
    os_encoded = lb3.transform([os])[0]

    # Create input DataFrame
    input_data = pd.DataFrame([{
        'Mobile Name': mobile_name_encoded,
        'Brand': brand_encoded,
        'OS': os_encoded,
        'CPU Performance (Score)': cpu_score,
        'GPU Performance (Score)': gpu_score,
        'Camera Quality Rating (out of 5)': camera_rating,
        'Gaming Score (out of 5)': gaming_score,
        'Multitasking Score (out of 5)': multitasking_score,
        'Price Range': price_range
    }])

    # Ensure the feature names match the training dataset
    input_data = input_data[feature_names]

    # Predict
    try:
        predicted_rating = model.predict(input_data)[0]
        st.success(f"Predicted Customer Rating: {predicted_rating:.2f}")

        # Suggest smartphones from the dataset
        st.write("### Suggested Smartphones")
        suggested_smartphones = df_smartphones[
            (df_smartphones['Customer Rating (out of 5)'] >= predicted_rating - 0.5) &
            (df_smartphones['Customer Rating (out of 5)'] <= predicted_rating + 0.5)
        ].sort_values(by='Customer Rating (out of 5)', ascending=False)

        if not suggested_smartphones.empty:
            st.table(suggested_smartphones[['Mobile Name', 'Brand', 'Customer Rating (out of 5)']])
        else:
            st.write("No smartphones found with similar ratings.")
    except ValueError as e:
        st.error(f"Error: {e}")
