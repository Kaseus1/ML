import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib

# Load data
df = pd.read_csv("hospitalLOS.csv")

# Define features
feature_cols = [
    'rcount', 'gender', 'dialysisrenalendstage', 'asthma', 'irondef',
    'pneum', 'substancedependence', 'psychologicaldisordermajor', 'depress',
    'psychother', 'fibrosisandother', 'malnutrition', 'hemo', 'hematocrit',
    'neutrophils', 'sodium', 'glucose', 'bloodureanitro', 'creatinine',
    'bmi', 'pulse', 'respiration', 'secondarydiagnosisnonicd9'
]

X = df[feature_cols].copy()

# Encode gender (F=0, M=1)
X['gender'] = X['gender'].map({'F': 0, 'M': 1})

# Bucket the target variable 'lengthofstay' into categories
def bucket_los(days):
    if days <= 3:
        return 'Short'
    elif days <= 7:
        return 'Medium'
    else:
        return 'Long'

y = df['lengthofstay'].apply(bucket_los)

# One-hot encode 'secondarydiagnosisnonicd9' categorical variable
X = pd.get_dummies(X, columns=['secondarydiagnosisnonicd9'], drop_first=True)

# Convert all columns to numeric and fill missing values with column mean
X = X.apply(pd.to_numeric, errors='coerce')
X.fillna(X.mean(), inplace=True)

# Encode target labels (Short, Medium, Long)
label_encoder = LabelEncoder()
y_enc = label_encoder.fit_transform(y)

# Save label encoder for future use
joblib.dump(label_encoder, "los_label_encoder.pkl")

# Split data into training and test sets with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

# Apply SMOTE only on the training set to balance classes
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Initialize and train XGBoost classifier
model = XGBClassifier(eval_metric='mlogloss', random_state=42)
model.fit(X_train_smote, y_train_smote)

# Save the trained model for future use
joblib.dump(model, "los_xgb_model.pkl")

# Predict on test set
y_pred = model.predict(X_test)

# Print classification report with class names
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))



