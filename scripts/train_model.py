import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

# Load dataset
data = pd.read_csv("/home/pavithra/k8s-failure-prediction/data/merged_data.csv")

# Convert datetime columns to numeric timestamps
for col in data.select_dtypes(include=['object', 'datetime']):
    try:
        data[col] = pd.to_datetime(data[col]).astype(int) / 10**9
    except:
        pass

# Handle categorical features
categorical_cols = data.select_dtypes(include=['object']).columns
data[categorical_cols] = data[categorical_cols].apply(LabelEncoder().fit_transform)

# Handle missing values
imputer = SimpleImputer(strategy='mean')
data.iloc[:, :] = imputer.fit_transform(data)

# Split into features and target
X = data.drop(columns=["target"])
y = data["target"]

# Handle Class Imbalance with SMOTE
smote = SMOTE(sampling_strategy=0.6, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_resampled)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

# Hyperparameter Tuning

param_grid = {
    'n_estimators': [400, 500, 600],  # More trees to learn better
    'max_depth': [10, 12, 15],        # Allow deeper trees
    'learning_rate': [0.1, 0.2, 0.3], # Increase learning rate
    'min_child_weight': [1, 2],       # Reduce constraints
    'subsample': [0.9, 1.0],          # Use more data per tree
    'colsample_bytree': [0.9, 1.0],   # Use more features per tree
    'gamma': [0, 0.1],                # Reduce penalty on splits
    'reg_lambda': [0, 1],             # Reduce L2 regularization
    'reg_alpha': [0, 1],              # Reduce L1 regularization
    'scale_pos_weight': [1]           # Balance class weights normally
}


xgb_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss')

search = RandomizedSearchCV(xgb_model, param_distributions=param_grid, n_iter=30, scoring='accuracy', cv=5, verbose=1, n_jobs=-1, random_state=42)
search.fit(X_train, y_train)

best_model = search.best_estimator_

# Predictions
y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)

# Accuracy Scores
train_accuracy = accuracy_score(y_train, y_train_pred) * 100
test_accuracy = accuracy_score(y_test, y_test_pred) * 100

print(f"\n🔥 Train Accuracy: {train_accuracy:.2f}%")
print(f"🔥 Test Accuracy: {test_accuracy:.2f}%")

# Classification Report
print("\n📊 Classification Report:")
print(classification_report(y_test, y_test_pred))

joblib.dump(best_model, "k8s_failure_model.pkl")
print("\nMODEL SAVED\n")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Feature Importance Graph
feature_importances = best_model.feature_importances_
features = data.drop(columns=["target"]).columns

# Sort feature importances
sorted_idx = np.argsort(feature_importances)[::-1]

plt.figure(figsize=(10, 5))
sns.barplot(x=feature_importances[sorted_idx][:10], y=[features[i] for i in sorted_idx[:10]], palette="coolwarm")
plt.xlabel("Feature Importance Score")
plt.ylabel("Top 10 Features")
plt.title("Feature Importance (Top 10)")
plt.show()

