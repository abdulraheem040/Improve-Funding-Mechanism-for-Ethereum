import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the datasets
train_data = pd.read_csv("C:/Users/abdul/Downloads/dataset/dataset.csv")
test_data = pd.read_csv("C:/Users/abdul/Downloads/dataset/test.csv")

# Check the first few rows of the dataset to understand its structure
print("Train Data:")
print(train_data.head())
print("Test Data:")
print(test_data.head())

# Feature Engineering: Encoding categorical variables
# Combine both train and test data to fit the label encoder for 'project_a' and 'project_b'
combined_projects = pd.concat([
    train_data['project_a'], train_data['project_b'], 
    test_data['project_a'], test_data['project_b']
])

# Initialize label encoder for project URLs
label_encoder = LabelEncoder()
label_encoder.fit(combined_projects)

# Encode the 'project_a' and 'project_b' columns in both train and test data
train_data['project_a_encoded'] = label_encoder.transform(train_data['project_a'])
train_data['project_b_encoded'] = label_encoder.transform(train_data['project_b'])

test_data['project_a_encoded'] = label_encoder.transform(test_data['project_a'])
test_data['project_b_encoded'] = label_encoder.transform(test_data['project_b'])

# Encode the 'funder' column using LabelEncoder
funder_encoder = LabelEncoder()
train_data['funder_encoded'] = funder_encoder.fit_transform(train_data['funder'])

# Handle unseen labels in the test data by assigning a default value (-1)
def encode_funder(funder_column, encoder):
    return funder_column.apply(lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1)

test_data['funder_encoded'] = encode_funder(test_data['funder'], funder_encoder)

# Handle Date columns (if any)
if 'date_column' in train_data.columns:
    train_data['date_column'] = pd.to_datetime(train_data['date_column'])
    train_data['date_column'] = train_data['date_column'].astype(np.int64) // 10**9  # Convert to seconds since epoch

if 'date_column' in test_data.columns:
    test_data['date_column'] = pd.to_datetime(test_data['date_column'])
    test_data['date_column'] = test_data['date_column'].astype(np.int64) // 10**9  # Convert to seconds since epoch

# Drop unnecessary columns and prepare training and test features
columns_to_drop = ['weight_a', 'weight_b', 'project_a', 'project_b', 'funder']
if 'date_column' in train_data.columns:
    columns_to_drop.append('date_column')

X_train = train_data.drop(columns=columns_to_drop + ['id'], axis=1)  # Ensure 'id' is dropped
y_train = train_data['weight_a']  # Target variable

X_test = test_data.drop(columns=['id', 'project_a', 'project_b', 'funder', 'date_column'], axis=1, errors='ignore')

# Ensure both train and test datasets have the same columns
common_columns = X_train.columns.intersection(X_test.columns)
X_train = X_train[common_columns]
X_test = X_test[common_columns]

# Debug: Check if columns now match
print("X_train columns:", X_train.columns)
print("X_test columns:", X_test.columns)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.select_dtypes(include=[np.number]))
X_test_scaled = scaler.transform(X_test.select_dtypes(include=[np.number]))

# Train-test split (for model evaluation during training)
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train_scaled, y_train, test_size=0.2, random_state=42)

# Initialize and train the model (RandomForestRegressor)
model = RandomForestRegressor(n_estimators=300, max_depth=15, random_state=42)
model.fit(X_train_split, y_train_split)

# Predict on the validation set
y_pred_val = model.predict(X_val_split)
mse = mean_squared_error(y_val_split, y_pred_val)
print(f"Validation MSE: {mse}")

# Generate predictions for the test set
predictions = model.predict(X_test_scaled)

# Ensure predictions are between 0 and 1
predictions = np.clip(predictions, 0, 1)

# Prepare the submission file
submission = pd.DataFrame({'id': test_data['id'], 'pred': predictions})
submission.to_csv('submission1.csv', index=False)

print("Submission file created successfully!")
