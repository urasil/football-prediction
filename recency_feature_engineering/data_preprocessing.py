import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split


class DataPreprocessing:
    def __init__(self):
        self.encoder = OneHotEncoder(handle_unknown='ignore')

    def chronological_split(self, df, train_ratio=0.8, chronological=False):
        df = df.sort_values(by='Date')
        df.drop(columns=['Date'], inplace=True)
        df.dropna(inplace=True)
        if chronological:
            train_size = int(len(df) * train_ratio)
            train_df = df[:train_size]
            test_df = df[train_size:]
        else:
            train_df, test_df = train_test_split(df, train_size=train_ratio, random_state=31, shuffle=True)
        return train_df, test_df

    def encode_teams(self, df, encoding="one-hot"):
        if encoding == "one-hot":
            encoded_teams = self.encoder.fit_transform(df[['Home Team', 'Away Team']]).toarray()
            encoded_team_columns = self.encoder.get_feature_names_out(['Home Team', 'Away Team'])
            
            encoded_df = pd.DataFrame(encoded_teams, columns=encoded_team_columns, index=df.index)
            df = pd.concat([df, encoded_df], axis=1)
            return df.drop(['Home Team', 'Away Team'], axis=1)
        else:
            all_teams = pd.concat([df['Home Team'], df['Away Team']]).unique()        
            encoder = LabelEncoder()
            encoder.fit(all_teams)
            
            df['Home Team'] = encoder.transform(df['Home Team'])
            df['Away Team'] = encoder.transform(df['Away Team'])
            
            return df


    
    def standardize_features(self, df, features, scale_type='minmax'):
        if scale_type == 'minmax':
            scaler = MinMaxScaler()
        else: 
            scaler = StandardScaler()
        
        df[features] = scaler.fit_transform(df[features])
        return df