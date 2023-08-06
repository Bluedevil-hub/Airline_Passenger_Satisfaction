from flask import Flask, request, render_template
from flask_caching import Cache
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# Load the saved model file
model = joblib.load('model.pkl')

# Create a Flask app
app = Flask(__name__, static_folder='static')
cache = Cache(app, config={'CACHE_TYPE': 'simple'})


# Define an API endpoint for the home page


@app.route('/')
def home():
    return render_template('index.html')

# Define an API endpoint for making predictions


@app.route('/predict', methods=['POST'])
def predict():
    cache.clear()
    # Get the input data from the request
    input_data = request.form.to_dict()

    # Convert the input data to a pandas DataFrame
    data = {'Gender': [str(input_data['gender'])],
            'Customer Type': [str(input_data['ctype'])],
            'Age': [int(input_data['age'])],
            'Type of Travel': [str(input_data['ttype'])],
            'Class': [str(input_data['tclass'])],
            'Flight Distance': [int(input_data['tdistance'])],
            'Inflight wifi service': [int(input_data['inwififlight'])],
            'Departure/Arrival time convenient': [int(input_data['datime'])],
            'Ease of Online booking': [int(input_data['eob'])],
            'Gate location': [int(input_data['gloc'])],
            'Food and drink': [int(input_data['fooddrink'])],
            'Online boarding': [int(input_data['onboard'])],
            'Seat comfort': [int(input_data['scomfort'])],
            'Inflight entertainment': [int(input_data['inenter'])],
            'On-board service': [int(input_data['obs'])],
            'Leg room service': [int(input_data['lgserv'])],
            'Baggage handling': [int(input_data['baghan'])],
            'Checkin service': [int(input_data['cserv'])],
            'Inflight service': [int(input_data['infiserv'])],
            'Cleanliness': [int(input_data['clean'])],
            'Departure Delay in Minutes': [int(input_data['ddinmin'])],
            'Arrival Delay in Minutes': [int(input_data['adinmin'])]
            }
    struct_col = {'Age': {},
                  'Flight Distance': {},
                  'Inflight wifi service': {},
                  'Departure/Arrival time convenient': {},
                  'Ease of Online booking': {},
                  'Gate location': {},
                  'Food and drink': {},
                  'Online boarding': {},
                  'Seat comfort': {},
                  'Inflight entertainment': {},
                  'On-board service': {},
                  'Leg room service': {},
                  'Baggage handling': {},
                  'Checkin service': {},
                  'Inflight service': {},
                  'Cleanliness': {},
                  'Departure Delay in Minutes': {},
                  'Gender_Female': {},
                  'Gender_Male': {},
                  'Customer_Type_Loyal Customer': {},
                  'Customer_Type_disloyal Customer': {},
                  'Type_of_Travel_Business travel': {},
                  'Type_of_Travel_Personal Travel': {},
                  'Class_Business': {},
                  'Class_Eco': {},
                  'Class_Eco Plus': {}}

    Y_new = pd.DataFrame(struct_col)

    X_new = pd.DataFrame(data)

    X_new.drop(columns=['Arrival Delay in Minutes'], inplace=True)
   # X_new.iloc[:, 6:20] = X_new.iloc[:, 6:20].replace(0, 3)
    print(X_new)
    X_new = pd.concat([X_new, pd.get_dummies(
        X_new['Gender'], prefix='Gender')], axis=1)

    X_new = pd.concat([X_new, pd.get_dummies(
        X_new['Customer Type'], prefix='Customer_Type')], axis=1)

    X_new = pd.concat([X_new, pd.get_dummies(
        X_new['Type of Travel'], prefix='Type_of_Travel')], axis=1)

    X_new = pd.concat([X_new, pd.get_dummies(
        X_new['Class'], prefix='Class')], axis=1)

    X_new.drop(columns=['Gender', 'Customer Type',
               'Type of Travel', 'Class'], inplace=True)

    ms = set(Y_new.columns) - set(X_new.columns)

    x_C = X_new.copy()

    for col in ms:
        x_C[col] = 0

    X_new = X_new.iloc[:, :17]

    X_new['Gender_Female'], X_new['Gender_Male'], X_new['Customer_Type_Loyal Customer'], X_new['Customer_Type_disloyal Customer'], X_new['Type_of_Travel_Business travel'], X_new['Type_of_Travel_Personal Travel'], X_new['Class_Business'], X_new['Class_Eco'], X_new[
        'Class_Eco Plus'] = x_C['Gender_Female'], x_C['Gender_Male'], x_C['Customer_Type_Loyal Customer'], x_C['Customer_Type_disloyal Customer'], x_C['Type_of_Travel_Business travel'], x_C['Type_of_Travel_Personal Travel'], x_C['Class_Business'], x_C['Class_Eco'], x_C['Class_Eco Plus']
    print(X_new.to_dict())

    # Use the loaded model to make predictions on the input data
    predictions = model.predict(X_new)
    print(predictions)
    # Render the predictions on a new page
    return render_template('prediction.html', predictions=predictions)


# Run the app
if __name__ == '__main__':
    app.run()
