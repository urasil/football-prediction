import pandas as pd

class GeneralFeatures:
    def __init__(self):
        data_dir = "football-prediction/epl-training.csv"
        df = pd.read_csv(data_dir)
        df.columns = [
            'Date',               
            'Home Team',          
            'Away Team',
            'Full Time Home Goals',  
            'Full Time Away Goals',  
            'Full Time Result',       
            'Half Time Home Goals',  
            'Half Time Away Goals',  
            'Half Time Result',      
            'Referee',
            'Home Shots',             
            'Away Shots',             
            'Home Shots on Target',   
            'Away Shots on Target',   
            'Home Corners',          
            'Away Corners',           
            'Home Fouls',             
            'Away Fouls',             
            'Home Yellow Cards',     
            'Away Yellow Cards',     
            'Home Red Cards',     
            'Away Red Cards'      
        ]
        
        df['Home Goal Conversion Rate'] = df['Full Time Home Goals'] / df['Home Shots on Target'].replace(0, 1)
        df['Away Goal Conversion Rate'] = df['Full Time Away Goals'] / df['Away Shots on Target'].replace(0, 1)

        df['Home Attacking Intensity'] = 2 * df["Home Shots on Target"] + 1 * (df['Home Shots'] - df["Home Shots on Target"]) + 0.5 * df['Home Corners']
        df['Away Attacking Intensity'] = 2 * df["Away Shots on Target"] + 1 * (df['Away Shots'] - df["Away Shots on Target"]) + 0.5 * df['Away Corners']
        df['Attacking Intensity Difference'] = df['Home Attacking Intensity'] - df['Away Attacking Intensity']

        df['Home Disciplinary Pressure'] = df['Home Fouls'] + df['Home Yellow Cards'] + df['Home Red Cards']
        df['Away Disciplinary Pressure'] = df['Away Fouls'] + df['Away Yellow Cards'] + df['Away Red Cards']
        df['Disciplinary Pressure Difference'] = df['Home Disciplinary Pressure'] - df['Away Disciplinary Pressure']

        df['xG Home'] = df['Home Goal Conversion Rate'] * df['Home Attacking Intensity']
        df['xG Away'] = df['Away Goal Conversion Rate'] * df['Away Attacking Intensity']

        df['Match Outcome'] = df['Full Time Result'].map({'H': 1, 'D': 0, 'A': -1})
        
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
        df = df.sort_values(by='Date')
        df['Season'] = df['Date'].apply(self.get_season)
        self.df = df

    def compute_difference_using_home_team_as_reference(self):
        features_to_differentiate = [
            ('Full Time Home Goals', 'Full Time Away Goals', 'Full Time Goal Difference'),
            ('Half Time Home Goals', 'Half Time Away Goals', 'Half Time Goal Difference'),
            ('Home Shots', 'Away Shots', 'Shot Difference'),
            ('Home Shots on Target', 'Away Shots on Target', 'Shots on Target Difference'),
            ('Home Corners', 'Away Corners', 'Corner Difference'),
            ('Home Fouls', 'Away Fouls', 'Foul Difference'),
            ('Home Yellow Cards', 'Away Yellow Cards', 'Yellow Card Difference'),
            ('Home Red Cards', 'Away Red Cards', 'Red Card Difference')
        ]
        for home_feature, away_feature, new_feature in features_to_differentiate:
            self.df[new_feature] = self.df[home_feature] - self.df[away_feature]
    
    # Each season starts at 08 and ends at 05 of next year - 2000-2001 season will be the 2000 season
    def get_season(self, date):
        if date.month >= 8:  
            return (date.year)
        else:  
            return (date.year - 1)

    def get_df_with_features(self):
        return self.df

        