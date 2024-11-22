from django.shortcuts import render
import joblib
import pandas as pd
import os

# Load the trained model and encoder
model_path = os.path.join(os.path.dirname(__file__), 'models', 'house_price_model.pkl')
encoder_path = os.path.join(os.path.dirname(__file__), 'models', 'house_price_encoder.pkl')
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

        # Load the encoder and transform categorical features
        encoder = joblib.load('price_app/models/house_price_encoder.pkl')
        encoded_categorical = encoder.transform([categorical_features])

        # Combine numeric and encoded categorical features
        input_features = pd.DataFrame(
            [numeric_features + list(encoded_categorical[0])],
            columns=[
                'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
                'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated'
            ] + list(encoder.get_feature_names_out(['waterfront', 'view', 'condition']))
        )

        # Load the trained model
        model = joblib.load('price_app/models/house_price_model.pkl')

        # Predict the price
        predicted_price = model.predict(input_features)[0]

        return render(request, 'result.html', {'predicted_price': predicted_price})

    return render(request, 'index.html')
