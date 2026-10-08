from src.data import generate_raw_tabular_data
from src.features import build_feature_pipeline, save_preprocessor


def main():
  print("Step 1: Ingesting Raw Tabular Dataset with Missing & Categorical Data...")
  X_train, X_test, y_train, y_test, num_cols, cat_cols = (
      generate_raw_tabular_data()
  )

  print(f"Raw Training Shape: {X_train.shape}")
  print(f"Missing Values in Train:\n{X_train.isnull().sum()}\n")

  print("Step 2: Fitting Automated Feature Engineering Pipeline...")
  preprocessor = build_feature_pipeline(num_cols, cat_cols)

  # Fit and transform raw training features
  X_train_transformed = preprocessor.fit_transform(X_train)
  X_test_transformed = preprocessor.transform(X_test)

  print(
      f"Transformed Training Feature Matrix Shape: {X_train_transformed.shape}"
  )

  print("Step 3: Serializing Fitted Feature Preprocessor...")
  save_preprocessor(preprocessor, "best_preprocessor.joblib")
  print("✅ Feature pipeline successfully executed and saved!\n")


if __name__ == "__main__":
  main()
