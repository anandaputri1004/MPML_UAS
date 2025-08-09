import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

# Load dataset
df = pd.read_csv("C:/Users/LENOVO/Downloads/mpml_uas_package/onlinefoods.csv")


# Clean Monthly Income
def clean_income(x):
    s = str(x).replace(',', '').replace('Rs', '').strip()
    return float(''.join(ch for ch in s if ch.isdigit() or ch=='.')) if any(ch.isdigit() for ch in s) else np.nan

df['Monthly Income (num)'] = df['Monthly Income'].apply(clean_income)
df['Monthly Income (num)'] = pd.to_numeric(df['Monthly Income (num)'], errors='coerce')
df['Monthly Income (num)'] = df['Monthly Income (num)'].fillna(df['Monthly Income (num)'].median())

# Target
df['target'] = df['Output'].map({'Yes':1, 'No':0}).astype(int)

# Features
numeric_feats = ['Age','Family size','Monthly Income (num)']
categorical_feats = ['Gender','Marital Status','Educational Qualifications','Feedback']

X = df[numeric_feats + categorical_feats].copy()
y = df['target']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Preprocessor
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_feats),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_feats)
])

# Models
models = {
    'logreg': (Pipeline([('preproc', preprocessor), ('clf', LogisticRegression(max_iter=1000, random_state=42))]),
               {'clf__C':[0.1,1.0]}),
    'rf': (Pipeline([('preproc', preprocessor), ('clf', RandomForestClassifier(random_state=42))]),
           {'clf__n_estimators':[100], 'clf__max_depth':[None,10]}),
    'gb': (Pipeline([('preproc', preprocessor), ('clf', GradientBoostingClassifier(random_state=42))]),
           {'clf__n_estimators':[100], 'clf__learning_rate':[0.1], 'clf__max_depth':[3]})
}

# Train & evaluate
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
best_model = None
best_f1 = -1

for name, (pipe, params) in models.items():
    gs = GridSearchCV(pipe, params, cv=skf, scoring='f1', n_jobs=1)
    gs.fit(X_train, y_train)
    y_pred = gs.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    print(f"{name} F1={f1:.4f} | params={gs.best_params_}")
    if f1 > best_f1:
        best_f1 = f1
        best_model = gs.best_estimator_

# Save best model
joblib.dump(best_model, "best_model_light.pkl")
print("Best model saved to best_model_light.pkl")
