import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, f1_score
from xgboost import XGBClassifier


# ------------------------- Load and Process Functional Connectome Matrices -------------------------
def process_functional_connectome(file_path, matrix_size=128):
    df = pd.read_csv(file_path)
    numeric_array = df.select_dtypes(include=['number']).to_numpy()

    n_participants = numeric_array.shape[0]
    triu_indices = np.triu_indices(matrix_size, k=1)
    upper_tri_matrices = np.zeros((n_participants, len(triu_indices[0])))

    for i in range(n_participants):
        full_matrix = np.zeros((matrix_size, matrix_size))
        full_matrix[triu_indices] = numeric_array[i, :len(triu_indices[0])]
        full_matrix = full_matrix + full_matrix.T  # Make it symmetric
        upper_tri_matrices[i] = full_matrix[triu_indices]

    column_names = [f"FC_{row}_{col}" for row, col in zip(*triu_indices)]
    return pd.DataFrame(upper_tri_matrices, columns=column_names)


# ------------------------- Load and Merge Metadata -------------------------
def load_and_merge_metadata(quant_file, cat_file):
    train_quan = pd.read_csv(quant_file).set_index('participant_id')
    train_cat = pd.read_csv(cat_file).set_index('participant_id')
    return train_quan.join(train_cat, how='inner')


# ------------------------- Handle Missing Values -------------------------
def impute_numeric_data(df):
    df_imputed = SimpleImputer(strategy='median').fit_transform(df)
    df_imputed = IterativeImputer(max_iter=10, random_state=42).fit_transform(df_imputed)
    return pd.DataFrame(df_imputed, columns=df.columns, index=df.index)


# ------------------------- Combine Data -------------------------
def combine_data(connectivity_df, metadata_df):
    connectivity_df.index = metadata_df.index
    return pd.concat([connectivity_df, metadata_df], axis=1)


# ------------------------- Load and Preprocess Training Data -------------------------
connectivity_df = process_functional_connectome('archive/TRAIN/TRAIN_FUNCTIONAL_CONNECTOME_MATRICES.csv')
metadata_df = load_and_merge_metadata('archive/TRAIN/TRAIN_QUANTITATIVE_METADATA.csv',
                                      'archive/TRAIN/TRAIN_CATEGORICAL_METADATA.csv')
metadata_df = impute_numeric_data(metadata_df)
combined_df = combine_data(connectivity_df, metadata_df)

# Load labels
y = pd.read_csv('archive/TRAIN/TRAINING_SOLUTIONS.csv', index_col=0)
combined_df = combined_df.reset_index().merge(y, on='participant_id', how='inner')

# ------------------------- Model Training -------------------------
X = combined_df.drop(columns=['participant_id', 'Sex_F', 'ADHD_Outcome'])
y = combined_df[['Sex_F', 'ADHD_Outcome']]

# Handle missing values & scale data
X = SimpleImputer(strategy='mean').fit_transform(X)
X = StandardScaler().fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train XGBoost Model
xgb_model = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=7, random_state=42)
xgb_model.fit(X_train, y_train)

# Predict & Evaluate
y_pred = xgb_model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# ------------------------- Load and Preprocess Test Data -------------------------
test_connectivity_df = process_functional_connectome("archive/TEST/TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv")
test_metadata_df = load_and_merge_metadata("archive/TEST/TEST_QUANTITATIVE_METADATA.csv",
                                           "archive/TEST/TEST_CATEGORICAL.csv")
test_metadata_df = impute_numeric_data(test_metadata_df)
test_combined_df = combine_data(test_connectivity_df, test_metadata_df)

participant_ids = test_combined_df.index.tolist()
X_test_final = StandardScaler().fit_transform(test_combined_df)

# Predict on Test Data
y_test_pred = xgb_model.predict(X_test_final)

# Save Predictions
results_df = pd.DataFrame({
    "participant_id": participant_ids,
"ADHD_Outcome": y_test_pred[:, 1],
    "Sex_F": y_test_pred[:, 0]
})
results_df.to_csv("test_predictions1.csv", index=False)
print("Predictions saved to test_predictions.csv")
