import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def generate_raw_tabular_data(n_samples: int = 1000, random_state: int = 42):
  """Generates a synthetic raw tabular DataFrame containing numerical features with missing values (NaNs) and categorical columns."""
  np.random.seed(random_state)

  data = {
      "age": np.random.choice(
          [25, 30, 35, 40, np.nan, 50, 60], size=n_samples
      ),
      "income": np.random.choice(
          [30000, 50000, 75000, np.nan, 120000], size=n_samples
      ),
      "credit_score": np.random.normal(loc=650, scale=50, size=n_samples),
      "education": np.random.choice(
          ["High School", "Bachelor", "Master", np.nan], size=n_samples
      ),
      "city_tier": np.random.choice(
          ["Tier_1", "Tier_2", "Tier_3"], size=n_samples
      ),
      "target": np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2]),
  }

  df = pd.DataFrame(data)

  num_features = ["age", "income", "credit_score"]
  cat_features = ["education", "city_tier"]

  X = df[num_features + cat_features]
  y = df["target"]

  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=random_state, stratify=y
  )

  return X_train, X_test, y_train, y_test, num_features, cat_features
