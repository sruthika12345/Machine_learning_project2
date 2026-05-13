import pickle
from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor

app = Flask(__name__)

# Load models and encoders
model = pickle.load(open('./models/Model.pkl', 'rb'))
Brand_Encoder = pickle.load(open('./models/Brand_Encoder.pkl', 'rb'))
Model_Encoder = pickle.load(open('./models/Model_Encoder.pkl', 'rb'))
OneHot_Encoder = pickle.load(open('./models/OneHot_Encoder.pkl', 'rb'))


@app.route('/', methods=['GET'])
def Home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    # Extract form data
    Brand = request.form['Brand']
    Model = request.form['Model']
    Fuel = request.form['Fuel']
    Transmission = request.form['Transmission']
    Year = int(request.form['Year'])
    EngineSize = float(request.form['EngineSize'])
    Mileage = int(request.form['Mileage'])
    Doors = int(request.form['Doors'])
    OwnerCount = int(request.form['OwnerCount'])

    # Create DataFrame
    input_df = pd.DataFrame({
        'Brand': [Brand],
        'Model': [Model],
        'Year': [Year],
        'EngineSize': [EngineSize],
        'Fuel': [Fuel],
        'Transmission': [Transmission],
        'Mileage': [Mileage],
        'Doors': [Doors],
        'OwnerCount': [OwnerCount]
    })

    # Encode Brand and Model
    input_df['Encoded_Brand'] = input_df['Brand'].map(Brand_Encoder)
    input_df['Encoded_Model'] = input_df['Model'].map(Model_Encoder)

    input_df['Encoded_Brand'].fillna(
        input_df['Encoded_Brand'].mean(),
        inplace=True
    )

    input_df['Encoded_Model'].fillna(
        input_df['Encoded_Model'].mean(),
        inplace=True
    )

    # Drop original columns
    input_df.drop(['Brand', 'Model'], axis=1, inplace=True)

    # One-hot encode categorical columns
    categorical_cols = ['Fuel', 'Transmission']

    encoded_array = OneHot_Encoder.transform(
        input_df[categorical_cols]
    )

    encoded_df = pd.DataFrame(
        encoded_array.toarray(),
        columns=OneHot_Encoder.get_feature_names_out(categorical_cols)
    )

    # Merge encoded columns
    input_df_encoded = input_df.drop(
        columns=categorical_cols
    ).reset_index(drop=True)

    input_data = pd.concat(
        [input_df_encoded, encoded_df],
        axis=1
    )

    # Prediction
    prediction = model.predict(input_data)

    output = round(prediction[0], 2)

    if output < 0:
        return render_template(
            'index.html',
            prediction_text="Sorry, you cannot sell this car"
        )

    return render_template(
        'index.html',
        prediction_text=f"Car is worth at: $ {output}"
    )


if __name__ == "__main__":
    app.run(debug=True)