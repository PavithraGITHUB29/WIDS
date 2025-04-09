import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score
from imblearn.over_sampling import SMOTE
from sklearn.experimental import enable_iterative_imputer
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ------------------------- Data Loading Functions -------------------------
def load_solutions(path):
    solutions_df = pd.read_csv(path, index_col=0)
    return solutions_df

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

# ------------------------- Preprocessing Functions -------------------------
def impute_missing_data(df):
    median_imputer = SimpleImputer(strategy='median')
    df_imputed = median_imputer.fit_transform(df)
    return pd.DataFrame(df_imputed, columns=df.columns, index=df.index)

def combine_data(connectivity_df, metadata_df):
    connectivity_df = connectivity_df.set_index('participant_id')
    combined = pd.concat([connectivity_df, metadata_df], axis=1, join='inner')
    combined.reset_index(inplace=True)
    return combined

def final_imputation(df):
    non_numeric = df[['participant_id']]
    numeric = df.drop(columns=['participant_id'])
    imputer = SimpleImputer(strategy='mean')
    imputed_array = imputer.fit_transform(numeric)
    imputed_df = pd.DataFrame(imputed_array, columns=numeric.columns)
    return pd.concat([non_numeric.reset_index(drop=True), imputed_df], axis=1)

# ------------------------- Modeling Pipeline -------------------------
def prepare_balanced_data(df):
    X = df.drop(columns=['participant_id', 'ADHD_Outcome'])
    y = df['ADHD_Outcome']
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    balanced_df = pd.DataFrame(X_resampled, columns=X.columns)
    balanced_df['ADHD_Outcome'] = y_resampled
    return balanced_df

def train_xgboost(df):
    X = df.drop(columns=['ADHD_Outcome'])
    y = df['ADHD_Outcome']
    
    pca = PCA(n_components=0.95, random_state=42)
    X = pca.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

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
    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    return model, pca

def predict_test_data(model, pca):
    test_connectivity_df = load_connectivity("archive/TEST/TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv")
    test_metadata_df = load_metadata("archive/TEST/TEST_QUANTITATIVE_METADATA.csv",
                                     "archive/TEST/TEST_CATEGORICAL.csv")
    test_metadata_df = impute_missing_data(test_metadata_df)
    test_combined_df = combine_data(test_connectivity_df, test_metadata_df)

    participant_ids = test_combined_df['participant_id']
    test_combined_df = final_imputation(test_combined_df)

    X_test_final = test_combined_df.drop(columns=['participant_id'])

    # Apply PCA using the one trained on training data
    X_test_final = pca.transform(X_test_final)

    y_test_pred = model.predict(X_test_final)

    results_df = pd.DataFrame({
        "participant_id": participant_ids,
        "ADHD_Outcome": y_test_pred
    })
    results_df.to_csv("test_predictions.csv", index=False)
    print("✅ Predictions saved to test_predictions.csv")

# ------------------------- Run Pipeline -------------------------
def main():
    solutions_df = load_solutions('archive/TRAIN/TRAINING_SOLUTIONS.csv')
    metadata_df = load_metadata(
        'archive/TRAIN/TRAIN_QUANTITATIVE_METADATA.csv',
        'archive/TRAIN/TRAIN_CATEGORICAL_METADATA.csv')
    metadata_df = impute_missing_data(metadata_df)

    connectivity_df = load_connectivity('archive/TRAIN/TRAIN_FUNCTIONAL_CONNECTOME_MATRICES.csv')
    combined_df = combine_data(connectivity_df, metadata_df)

    # Merge target labels
    combined_df = combined_df.merge(solutions_df[['ADHD_Outcome']], on='participant_id', how='inner')

    # Final imputation
    combined_df_imputed = final_imputation(combined_df)

    # Balance with SMOTE
    balanced_df = prepare_balanced_data(combined_df_imputed)

    model, pca = train_xgboost(balanced_df)

    # Predict test data
    predict_test_data(model, pca)

if __name__ == '__main__':
    main()
