import pandas as pd
from recency_feature_engineering.recency_features import RecencyFeatures
from recency_feature_engineering.feature_cleaning import FeatureCleaning
from recency_feature_engineering.data_preprocessing import DataPreprocessing
from recency_feature_engineering.model_training import RecencyModelTraining



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

train_df, val_df, test_df = preprocessor.chronological_split(df)

X_train, y_train = train_df.drop('Match Outcome', axis=1), train_df['Match Outcome']
X_val, y_val = val_df.drop('Match Outcome', axis=1), val_df['Match Outcome']
X_test, y_test = test_df.drop('Match Outcome', axis=1), test_df['Match Outcome']


# Model Training
print('Starting Training...')
trainer = RecencyModelTraining()
trainer.train(X_train, y_train)

val_rmse = trainer.evaluate(X_val, y_val)
print(f"Validation RMSE: {val_rmse}")

trainer.save_model('best_model.pkl')
trainer.load_model('best_model.pkl')

# Evaluation
print('Evaluation...')
test_rmse = trainer.evaluate(X_test, y_test)
print(f"Test RMSE: {test_rmse}")