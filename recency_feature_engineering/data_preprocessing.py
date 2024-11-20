import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

class DataPreprocessing:
    def __init__(self):
        self.encoder = OneHotEncoder(handle_unknown='ignore')
        self.scaler = StandardScaler()

    def chronological_split(self, df, train_ratio=0.7, val_ratio=0.15):
        df = df.sort_values(by='Date')
        df.drop(columns=['Date'], inplace=True)
        df.dropna(inplace=True)
        train_size = int(len(df) * train_ratio)
        val_size = int(len(df) * val_ratio)
        train_df = df[:train_size]
        val_df = df[train_size:train_size + val_size]
        test_df = df[train_size + val_size:]
        return train_df, val_df, test_df

    def encode_teams(self, df):
        encoded_teams = self.encoder.fit_transform(df[['Home Team', 'Away Team']]).toarray()
        encoded_team_columns = self.encoder.get_feature_names_out(['Home Team', 'Away Team'])
        
        encoded_df = pd.DataFrame(encoded_teams, columns=encoded_team_columns, index=df.index)
        df = pd.concat([df, encoded_df], axis=1)
        return df.drop(['Home Team', 'Away Team'], axis=1)


    def standardize_features(self, df, features):
        df[features] = self.scaler.fit_transform(df[features])
        return df
