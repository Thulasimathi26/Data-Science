
Original file is located at
    https://colab.research.google.com/drive/1LHSouQeD_tA9J58hn8Q1-yEM77VgZi3U

# Upload the Dataset

from google.colab import files

uploaded = files.upload()

# Load the Dataset

import pandas as pd

# Read the dataset
df = pd.read_csv('student-mat.csv', sep=';')

# Data Exploration

# Display first few rows
df.head()

# Shape of the dataset
print("Shape:", df.shape)

# Column names
print("Columns:", df.columns.tolist())

# Data types and non-null values
df.info()

# Summary statistics for numeric features
df.describe()

# Check for Missing Values and Duplicates

# Check for missing values
print(df.isnull().sum())

# Check for duplicates
print("Duplicate rows:", df.duplicated().sum())

# Visualize a Few Features

import seaborn as sns
import matplotlib.pyplot as plt

# Distribution of final grades
sns.histplot(df['G3'], kde=True)
plt.title('Distribution of Final Grade (G3)')
plt.xlabel('Final Grade')
plt.show()

# Relationship between study time and final grade
sns.boxplot(x='studytime', y='G3', data=df)
plt.title('Study Time vs Final Grade')
plt.show()

# Identify Target and Features

target = 'G3'
features = df.columns.drop(target)
print("Features:", features)

#Convert Categorical Columns to Numerical

# Identify categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns
print("Categorical Columns:", categorical_cols.tolist())

#One-Hot Encoding

df_encoded = pd.get_dummies(df, drop_first=True)

# Feature Scaling

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_encoded.drop('G3', axis=1))
y = df_encoded['G3']

#Train-Test Split

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#Model Building

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

#Evaluation

# Evaluate
print("MSE:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

#Make Predictions from New Input

# Sample input (replace values with any other valid values from the original dataset)
new_student = {
    'school': 'GP',             # 'GP' or 'MS'
    'sex': 'F',                 # 'F' or 'M'
    'age': 17,                  # Integer
    'address': 'U',            # 'U' or 'R'
    'famsize': 'GT3',          # 'LE3' or 'GT3'
    'Pstatus': 'A',            # 'A' or 'T'
    'Medu': 4,                 # 0 to 4
    'Fedu': 3,                 # 0 to 4
    'Mjob': 'health',          # 'teacher', 'health', etc.
    'Fjob': 'services',
    'reason': 'course',
    'guardian': 'mother',
    'traveltime': 2,
    'studytime': 3,
    'failures': 0,
    'schoolsup': 'yes',
    'famsup': 'no',
    'paid': 'no',
    'activities': 'yes',
    'nursery': 'yes',
    'higher': 'yes',
    'internet': 'yes',
    'romantic': 'no',
    'famrel': 4,
    'freetime': 3,
    'goout': 3,
    'Dalc': 1,
    'Walc': 1,
    'health': 4,
    'absences': 2,
    'G1': 14,
    'G2': 15
}

#Convert to DataFrame and Encode

import numpy as np

# Convert to DataFrame
new_df = pd.DataFrame([new_student])

# Combine with original df to match columns
df_temp = pd.concat([df.drop('G3', axis=1), new_df], ignore_index=True)

# One-hot encode
df_temp_encoded = pd.get_dummies(df_temp, drop_first=True)

# Match the encoded feature order
df_temp_encoded = df_temp_encoded.reindex(columns=df_encoded.drop('G3', axis=1).columns, fill_value=0)

# Scale (if you used scaling)
new_input_scaled = scaler.transform(df_temp_encoded.tail(1))

#Predict the Final Grade

predicted_grade = model.predict(new_input_scaled)
print("🎓 Predicted Final Grade (G3):", round(predicted_grade[0], 2))

#Deployment-Building an Interactive App

!pip install gradio

#Create a Prediction Function

import gradio as gr

def predict_grade(school, sex, age, address, famsize, Pstatus, Medu, Fedu,
                  Mjob, Fjob, reason, guardian, traveltime, studytime,
                  failures, schoolsup, famsup, paid, activities, nursery,
                  higher, internet, romantic, famrel, freetime, goout,
                  Dalc, Walc, health, absences, G1, G2):

    # Create input dictionary
    input_data = {
        'school': school, 'sex': sex, 'age': int(age), 'address': address, 'famsize': famsize,
        'Pstatus': Pstatus, 'Medu': int(Medu), 'Fedu': int(Fedu), 'Mjob': Mjob, 'Fjob': Fjob,
        'reason': reason, 'guardian': guardian, 'traveltime': int(traveltime), 'studytime': int(studytime),
        'failures': int(failures), 'schoolsup': schoolsup, 'famsup': famsup, 'paid': paid,
        'activities': activities, 'nursery': nursery, 'higher': higher, 'internet': internet,
        'romantic': romantic, 'famrel': int(famrel), 'freetime': int(freetime), 'goout': int(goout),
        'Dalc': int(Dalc), 'Walc': int(Walc), 'health': int(health), 'absences': int(absences),
        'G1': int(G1), 'G2': int(G2)
    }

    # Create DataFrame
    input_df = pd.DataFrame([input_data])

    # Combine and encode
    df_temp = pd.concat([df.drop('G3', axis=1), input_df], ignore_index=True)
    df_temp_encoded = pd.get_dummies(df_temp, drop_first=True)
    df_temp_encoded = df_temp_encoded.reindex(columns=df_encoded.drop('G3', axis=1).columns, fill_value=0)

    # Scale and predict
    scaled_input = scaler.transform(df_temp_encoded.tail(1))
    prediction = model.predict(scaled_input)

    return round(prediction[0], 2)

#Create the Gradio Interface

inputs = [
    gr.Dropdown(['GP', 'MS'], label="School (GP=Gabriel Pereira, MS=Mousinho da Silveira)"),
    gr.Dropdown(['M', 'F'], label="Gender (M=Male, F=Female)"),
    gr.Number(label="Student Age"),
    gr.Dropdown(['U', 'R'], label="Residence Area (U=Urban, R=Rural)"),
    gr.Dropdown(['LE3', 'GT3'], label="Family Size (LE3=≤3, GT3=>3 members)"),
    gr.Dropdown(['A', 'T'], label="Parent Cohabitation Status (A=Apart, T=Together)"),
    gr.Number(label="Mother's Education Level (0-4)"),
    gr.Number(label="Father's Education Level (0-4)"),
    gr.Dropdown(['teacher', 'health', 'services', 'at_home', 'other'], label="Mother's Job"),
    gr.Dropdown(['teacher', 'health', 'services', 'at_home', 'other'], label="Father's Job"),
    gr.Dropdown(['home', 'reputation', 'course', 'other'], label="Reason for Choosing School"),
    gr.Dropdown(['mother', 'father', 'other'], label="Guardian"),
    gr.Number(label="Travel Time to School (1-4)"),
    gr.Number(label="Weekly Study Time (1-4)"),
    gr.Number(label="Past Class Failures (0-3)"),
    gr.Dropdown(['yes', 'no'], label="Extra School Support"),
    gr.Dropdown(['yes', 'no'], label="Family Support"),
    gr.Dropdown(['yes', 'no'], label="Extra Paid Classes"),
    gr.Dropdown(['yes', 'no'], label="Participates in Activities"),
    gr.Dropdown(['yes', 'no'], label="Attended Nursery"),
    gr.Dropdown(['yes', 'no'], label="Aspires Higher Education"),
    gr.Dropdown(['yes', 'no'], label="Internet Access at Home"),
    gr.Dropdown(['yes', 'no'], label="Currently in a Relationship"),
    gr.Number(label="Family Relationship Quality (1-5)"),
    gr.Number(label="Free Time After School (1-5)"),
    gr.Number(label="Going Out Frequency (1-5)"),
    gr.Number(label="Workday Alcohol Consumption (1-5)"),
    gr.Number(label="Weekend Alcohol Consumption (1-5)"),
    gr.Number(label="Health Status (1=Very Bad to 5=Excellent)"),
    gr.Number(label="Number of Absences"),
    gr.Number(label="Grade in 1st Period (G1: 0-20)"),
    gr.Number(label="Grade in 2nd Period (G2: 0-20)")
]

output = gr.Number(label="🎯 Predicted Final Grade (G3)")

# Launch the app
gr.Interface(
    fn=predict_grade,
    inputs=inputs,
    outputs=output,
    title="🎓 Student Performance Predictor",
    description="Enter academic and demographic info to predict the final grade (G3) of a student."
).launch()

