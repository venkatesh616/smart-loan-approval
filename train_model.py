import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("loan.csv")   # update with your actual path

# Drop Loan_ID (not useful for prediction)
df = df.drop("Loan_ID", axis=1)

# Target variable
y = df["Loan_Status"]
X = df.drop("Loan_Status", axis=1)

# Encode target (Y=1, N=0)
y = LabelEncoder().fit_transform(y)

# Encode categorical columns
cat_cols = X.select_dtypes(include=["object"]).columns
for col in cat_cols:
    X[col] = X[col].astype(str)   # handle NaNs
    X[col] = LabelEncoder().fit_transform(X[col])

# Fill missing numeric values with median
for col in X.select_dtypes(include=["int64", "float64"]).columns:
    X[col] = X[col].fillna(X[col].median())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize + SVM pipeline
scaler = StandardScaler()
svm = SVC(kernel="rbf", probability=True, random_state=42)

pipeline = Pipeline([
    ("scaler", scaler),
    ("svm", svm)
])

# Train model
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model and column order
model_data = {
    "model": pipeline,
    "columns": X.columns.tolist()
}
joblib.dump(model_data, "loan_svm_model.pkl")

print("✅ Model saved as loan_svm_model.pkl")