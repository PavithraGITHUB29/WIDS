import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from sklearn.decomposition import PCA

# ------------------------- Utility Functions -------------------------

def load_solutions(path):
    return pd.read_csv(path, index_col=0)

def load_metadata(quant_file, cat_file):
    quantitative = pd.read_csv(quant_file).set_index('participant_id')
    categorical = pd.read_csv(cat_file).set_index('participant_id')
    return quantitative.join(categorical, how='inner')

def load_connectivity(file_path, matrix_size=128):
    df = pd.read_csv(file_path)
    participant_ids = df.iloc[:, 0]
    numeric_array = df.select_dtypes(include=['number']).to_numpy()

    n = numeric_array.shape[0]
    triu_idx = np.triu_indices(matrix_size, k=1)
    upper_tri_matrices = np.zeros((n, len(triu_idx[0])))

    for i in range(n):
        full_matrix = np.zeros((matrix_size, matrix_size))
        full_matrix[triu_idx] = numeric_array[i, :len(triu_idx[0])]
        full_matrix = full_matrix + full_matrix.T
        upper_tri_matrices[i] = full_matrix[triu_idx]

    col_names = [f"FC_{i}_{j}" for i, j in zip(*triu_idx)]
    final_df = pd.DataFrame(upper_tri_matrices, columns=col_names)
    final_df.insert(0, 'participant_id', participant_ids.values)
    return final_df

def impute_metadata(metadata_df):
    df = SimpleImputer(strategy='median').fit_transform(metadata_df)
    df = IterativeImputer(max_iter=10, random_state=42).fit_transform(df)
    return pd.DataFrame(df, columns=metadata_df.columns, index=metadata_df.index)

def final_imputation(df):
    non_numeric = df[['participant_id']]
    numeric = df.drop(columns=['participant_id'])
    imputed = SimpleImputer(strategy='mean').fit_transform(numeric)
    imputed_df = pd.DataFrame(imputed, columns=numeric.columns)
    return pd.concat([non_numeric.reset_index(drop=True), imputed_df], axis=1)

# ------------------------- Load and Prepare Training Data -------------------------

# Load files
solutions_df = load_solutions('archive/TRAIN/TRAINING_SOLUTIONS.csv')
metadata_df = load_metadata('archive/TRAIN/TRAIN_QUANTITATIVE_METADATA.csv', 
                            'archive/TRAIN/TRAIN_CATEGORICAL_METADATA.csv')
connectivity_df = load_connectivity('archive/TRAIN/TRAIN_FUNCTIONAL_CONNECTOME_MATRICES.csv')

# Impute metadata
metadata_df = impute_metadata(metadata_df)

# Combine data
combined_df = connectivity_df.set_index('participant_id').join(metadata_df, how='inner')
combined_df.reset_index(inplace=True)

# Merge with target
combined_df = combined_df.merge(solutions_df[['Sex_F']], on='participant_id', how='inner')
combined_df = final_imputation(combined_df)

# SMOTE oversampling
X = combined_df.drop(columns=['participant_id', 'Sex_F'])
y = combined_df['Sex_F']
X, y = SMOTE(random_state=42).fit_resample(X, y)

pca = PCA(n_components=0.95, random_state=42)  # Keep 95% of variance
X = pca.fit_transform(X)
# ------------------------- Train XGBoost Model -------------------------

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42,
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8
)

model.fit(X_train, y_train)

y_pred = model.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, y_pred))
print("Validation F1 Score:", f1_score(y_val, y_pred))
print("\nClassification Report:\n", classification_report(y_val, y_pred))

# ------------------------- Prepare Test Data & Predict -------------------------

test_connectivity_df = load_connectivity("archive/TEST/TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv")
test_metadata_df = load_metadata("archive/TEST/TEST_QUANTITATIVE_METADATA.csv",
                                 "archive/TEST/TEST_CATEGORICAL.csv")

test_metadata_df = impute_metadata(test_metadata_df)

test_combined_df = test_connectivity_df.set_index('participant_id').join(test_metadata_df, how='inner')
test_combined_df.reset_index(inplace=True)
test_combined_df = final_imputation(test_combined_df)

participant_ids = test_combined_df['participant_id']
X_test_final = test_combined_df.drop(columns=['participant_id'])
X_test_final = test_combined_df.drop(columns=['participant_id'])
X_test_final = pca.transform(X_test_final)


# Optionally scale features (if you used scaling during training)
# scaler = StandardScaler()
# X_test_final = scaler.fit_transform(X_test_final)

y_test_pred = model.predict(X_test_final)

# Save predictions
results_df = pd.DataFrame({
    "participant_id": participant_ids,
    "Sex_F": y_test_pred
})
results_df.to_csv("test_predictions_sex.csv", index=False)
print("✅ Predictions saved to test_predictions_sex.csv")
