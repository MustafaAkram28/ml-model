import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import ipywidgets as widgets
from IPython.display import display
import pickle

# Load the dataset
data = pd.read_excel("/content/realistic_smartphones.xlsx")

# Preprocessing: Label Encoding
lb1 = LabelEncoder()
data["Mobile Name"] = lb1.fit_transform(data["Mobile Name"])
lb2 = LabelEncoder()
data["Brand"] = lb2.fit_transform(data["Brand"])
lb3 = LabelEncoder()
data["OS"] = lb3.fit_transform(data["OS"])
lb4 = LabelEncoder()
data["Price Range"] = lb4.fit_transform(data["Price Range"])

# Features and Target
X = data.drop('Customer Rating (out of 5)', axis=1)  # Independent variables
Y = data['Customer Rating (out of 5)']               # Target variable

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, random_state=42)

# Model training (RandomForestRegressor)
model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train, Y_train)

# Function to predict customer rating
def predict_customer_rating(mobile_name, brand, os, battery_power, ram):
    input_data = pd.DataFrame([{
        'Mobile Name': int(mobile_name),
        'Brand': int(brand),
        'OS': int(os),
        'Battery Power': float(battery_power),
        'RAM': float(ram),
        'CPU Performance (Score)': float(X['CPU Performance (Score)'].mean()),
        'GPU Performance (Score)': float(X['GPU Performance (Score)'].mean()),
        'Camera Quality Rating (out of 5)': float(X['Camera Quality Rating (out of 5)'].mean()),
        'Gaming Score (out of 5)': float(X['Gaming Score (out of 5)'].mean()),
        'Multitasking Score (out of 5)': float(X['Multitasking Score (out of 5)'].mean()),
        'Price Range': int(X['Price Range'].mean())
    }])
    input_data = input_data[X.columns]
    predicted_rating = model.predict(input_data)[0]
    return predicted_rating

# Function to suggest a smartphone
def suggest_smartphone(predicted_rating):
    suggestions = data[data['Customer Rating (out of 5)'] >= predicted_rating - 0.5]
    if not suggestions.empty:
        suggestions['Mobile Name'] = lb1.inverse_transform(suggestions['Mobile Name'])
        suggestions['Brand'] = lb2.inverse_transform(suggestions['Brand'])
        suggestions['OS'] = lb3.inverse_transform(suggestions['OS'])
        return suggestions[['Mobile Name', 'Brand', 'OS', 'Customer Rating (out of 5)']].head()
    else:
        return "No suitable smartphone found."

# Widgets for user input
mobile_name_widget = widgets.Dropdown(options=lb1.classes_, description='Mobile Name:')
brand_widget = widgets.Dropdown(options=lb2.classes_, description='Brand:')
os_widget = widgets.Dropdown(options=lb3.classes_, description='OS:')
battery_power_widget = widgets.IntText(value=1000, description='Battery Power:')
ram_widget = widgets.IntText(value=2, description='RAM:')
predict_button = widgets.Button(description="Predict")

# Handle prediction when button is clicked
def on_button_clicked(b):
    mobile_name = mobile_name_widget.value
    brand = brand_widget.value
    os = os_widget.value
    battery_power = battery_power_widget.value
    ram = ram_widget.value

    mobile_name_encoded = lb1.transform([mobile_name])[0]
    brand_encoded = lb2.transform([brand])[0]
    os_encoded = lb3.transform([os])[0]

    predicted_rating = predict_customer_rating(mobile_name_encoded, brand_encoded, os_encoded, battery_power, ram)
    print(f"Predicted Customer Rating: {predicted_rating:.2f}")

    suggestions = suggest_smartphone(predicted_rating)
    print("Suggested Smartphones:")
    print(suggestions)

predict_button.on_click(on_button_clicked)

# Display widgets
display(mobile_name_widget, brand_widget, os_widget, battery_power_widget, ram_widget, predict_button)

# Save model and encoders
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(lb1, open('lb1.pkl', 'wb'))
pickle.dump(lb2, open('lb2.pkl', 'wb'))
pickle.dump(lb3, open('lb3.pkl', 'wb'))
pickle.dump(lb4, open('lb4.pkl', 'wb'))
