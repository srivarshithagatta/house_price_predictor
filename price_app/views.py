from django.shortcuts import render
import joblib
import pandas as pd
import os
import json

# Load the trained model and encoder once
model_path = os.path.join(os.path.dirname(__file__), 'models', 'house_price_model.pkl')
encoder_path = os.path.join(os.path.dirname(__file__), 'models', 'house_price_encoder.pkl')

# Load model and encoder globally so they're available to all views
model = joblib.load(model_path)
encoder = joblib.load(encoder_path)

def predict_price(request):
    if request.method == 'POST':
        # Extract numeric features
        numeric_features = [
            float(request.POST.get('bedrooms', 0)),
            float(request.POST.get('bathrooms', 0)),
            float(request.POST.get('sqft_living', 0)),
            float(request.POST.get('sqft_lot', 0)),
            float(request.POST.get('floors', 0)),
            float(request.POST.get('sqft_above', 0)),
            float(request.POST.get('sqft_basement', 0)),
            int(request.POST.get('yr_built', 0)),
            int(request.POST.get('yr_renovated', 0))
        ]

        # Extract categorical features for encoding
        categorical_features = [
            int(request.POST.get('waterfront', 0)),
            int(request.POST.get('view', 0)),
            int(request.POST.get('condition', 0))
        ]

        # Transform categorical features
        encoded_categorical = encoder.transform([categorical_features])

        # Combine numeric and encoded categorical features
        input_features = pd.DataFrame(
            [numeric_features + list(encoded_categorical[0])],
            columns=[  # List all the feature names here
                'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
                'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated'
            ] + list(encoder.get_feature_names_out(['waterfront', 'view', 'condition']))
        )

        # Predict the price
        predicted_price = model.predict(input_features)[0]

        # Render the result page with the predicted price
        return render(request, 'result.html', {'predicted_price': round(predicted_price, 2)})

    # If it's a GET request, just render the form (index.html)
    return render(request, 'index.html')


def property_form(request):
    # Load the dataset (ensure correct file path) once per request
    df = pd.read_csv('price_app/data/data.csv')

    # Drop unnecessary columns
    df = df.drop(columns=['date', 'statezip', 'country'])

    # Extract unique values for state, city, and street from the dataset
    unique_states = df['state'].unique()

    # Create a mapping for state -> cities and state -> streets
    state_city_mapping = {}
    state_street_mapping = {}

    for state in unique_states:
        cities = df[df['state'] == state]['city'].unique()
        streets = df[df['state'] == state]['street'].unique()

        state_city_mapping[state] = cities.tolist()
        state_street_mapping[state] = streets.tolist()

    # Pass the unique values to the template
    context = {
        'unique_states': unique_states,
        'state_city_mapping': json.dumps(state_city_mapping),  # Convert to JSON string
        'state_street_mapping': json.dumps(state_street_mapping),  # Convert to JSON string
    }

    return render(request, 'index.html', context)
