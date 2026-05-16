import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

print("Loading dataset...")
df = pd.read_csv('dataset.csv')

# Feature engineering
df['brand'] = df['name'].str.split().str[0]
df['miles_clean'] = pd.to_numeric(
    df['miles'].str.replace(',', '').str.replace(' miles', ''), errors='coerce'
)
df['price_clean'] = pd.to_numeric(
    df['price'].str.replace(r'[\$,]', '', regex=True), errors='coerce'
)
df = df.dropna(subset=['miles_clean', 'price_clean', 'brand', 'year'])
print(f"Clean dataset: {len(df)} rows, {df['brand'].nunique()} brands")

X = df[['brand', 'year', 'miles_clean']]
y = np.log(df['price_clean'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline = Pipeline([
    ('preprocessor', ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['brand'])
    ], remainder='passthrough')),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
])

print("Training Random Forest model...")
pipeline.fit(X_train, y_train)

preds = pipeline.predict(X_test)
r2 = r2_score(y_test, preds)
mae = mean_absolute_error(np.exp(y_test), np.exp(preds))
print(f"R² Score : {r2:.3f}")
print(f"Mean Absolute Error: ${mae:,.0f}")

with open('model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

brands = sorted(df['brand'].unique().tolist())
year_min, year_max = int(df['year'].min()), int(df['year'].max())
miles_max = int(df['miles_clean'].max())

meta = {
    'brands': brands,
    'year_min': year_min,
    'year_max': year_max,
    'miles_max': miles_max
}
with open('meta.pkl', 'wb') as f:
    pickle.dump(meta, f)

print("Saved model.pkl and meta.pkl")
print("\nDone! Now run: python app.py")
