import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
import joblib
data = pd.read_csv('price_app/data/data.csv')
outputs = pd.read_csv('price_app/data/output.csv')
data['price'] = outputs['price']

data = data.drop(columns=['date', 'street', 'city', 'statezip', 'country'])

categorical_columns = ['waterfront', 'view', 'condition']

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_categorical_data = encoder.fit_transform(data[categorical_columns])

encoded_df = pd.DataFrame(encoded_categorical_data, columns=encoder.get_feature_names_out(categorical_columns))

data = data.drop(columns=categorical_columns)
data = pd.concat([data, encoded_df], axis=1)

X = data.drop('price', axis=1)  # Features
y = data['price']               # Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

joblib.dump(model, 'price_app/models/houcmdse_price_model.pkl')
joblib.dump(encoder, 'price_app/models/house_price_encoder.pkl')
