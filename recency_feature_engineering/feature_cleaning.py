import pandas as pd 

class FeatureCleaning:
    def __init__(self, df, recency_features_prefix='Avg'):
        self.df = df
        self.recency_features_prefix = recency_features_prefix

    def remove_first_x_matches(self):
        self.df = self.df[pd.notna(self.df['Avg Goals Scored Home Last 5 Games'])]


    def keep_only_recency_features(self):
        recency_columns = [col for col in self.df.columns if col.startswith(self.recency_features_prefix)]
        non_recency_columns = ['Home Team', 'Away Team', 'Date', 'Match Outcome'] 
        columns_to_keep = recency_columns + non_recency_columns
        self.df = self.df[columns_to_keep]

    def clean_data(self):
        self.remove_first_x_matches()
        self.keep_only_recency_features()
        return self.df
