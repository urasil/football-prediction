import pandas as pd
from recency_feature_engineering.recency_features import RecencyFeatures
from recency_feature_engineering.feature_cleaning import FeatureCleaning
from recency_feature_engineering.data_preprocessing import DataPreprocessing
from recency_feature_engineering.model_training import RecencyModelTraining
import numpy as np

# Feature Engineering
print('Feature Engineering...')
recency = RecencyFeatures(max_x=5)
recency_df = recency.get_recency_features()
cleaning = FeatureCleaning(recency_df)
df = cleaning.clean_data()

# Preprocessing
print('Preprocessing Data...')
preprocessor = DataPreprocessing()
df = preprocessor.encode_teams(df)
numerical_features = [col for col in df.columns if 'Avg' in col]
df = preprocessor.standardize_features(df, numerical_features)

train_df, test_df = preprocessor.chronological_split(df)

X_train, y_train = train_df.drop('Match Outcome', axis=1), train_df['Match Outcome']
X_test, y_test = test_df.drop('Match Outcome', axis=1), test_df['Match Outcome']

# Model Training
print('Starting Training...')

trainer = RecencyModelTraining()

iterations = np.arange(0, 200, 10)
neighbors_range = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51]

metrics_logistic = trainer.train_with_iterations(X_train, y_train, X_test, y_test, iterations)
metrics_knn = trainer.train_and_evaluate_knn(X_train, y_train, X_test, y_test, neighbors_range)
# trainer.plot_logistic_metrics(metrics_logistic, iterations)
trainer.plot_knn_metrics(metrics_knn, neighbors_range=neighbors_range)