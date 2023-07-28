import pandas as pd
from flask import Flask, request, jsonify, render_template
from sklearn.tree import DecisionTreeClassifier

# Load the dataset and train the decision tree model
df_orig = pd.read_csv("C:\\Users\\Manoshi Raha\\OneDrive\\Desktop\\Secure Tomorrow\\GHSH_Pooled_Data1.csv")
df = df_orig.copy()
df['Bullied'] = df['Bullied'].fillna(df['Bullied'].median())
df['Smoke_cig_currently'] = df['Smoke_cig_currently'].fillna(df['Smoke_cig_currently'].mean())
columns = ['Smoke_cig_currently', 'Bullied', 'Country', 'Age Group', 'Sex', 'Have_Understanding_Parents']
df_selected = df[columns + ['Attempted_suicide']]
df_selected.dropna(inplace=True)
threshold = 0.5  # Adjust the threshold as needed
df_selected['Attempted_suicide'] = (df_selected['Attempted_suicide'] > threshold).astype(int)
df_encoded = pd.get_dummies(df_selected, drop_first=True)
X = df_encoded.drop('Attempted_suicide', axis=1)
y = df_encoded['Attempted_suicide']
model = DecisionTreeClassifier()
model.fit(X, y)

# Create the Flask app
app = Flask(__name__)

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    input_data = [data[col] for col in X.columns]
    prediction = model.predict([input_data])
    if prediction[0] == 1:
        result = 'Attempted Suicide'
    else:
        result = 'No Attempted Suicide'
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
