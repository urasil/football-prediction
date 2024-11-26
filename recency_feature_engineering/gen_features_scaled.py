import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


#home_possession,away_possession,home_passes_completed,home_passes_pct,
# home_progressive_passes,home_progressive_passing_distance,home_xg,
# home_take_ons_won,home_take_ons,home_interceptions,home_blocks,
# home_touches,home_touches_def_3rd,home_touches_mid_3rd,
# home_touches_att_3rd,home_carries,home_carries_progressive_distance,
# home_tackles,home_tackles_won,away_passes_completed,away_passes_pct,
# away_progressive_passes,away_progressive_passing_distance,away_xg,
# away_shots,away_shots_on_target,away_take_ons_won,away_take_ons,away_interceptions,
# away_blocks,away_touches,away_touches_def_3rd,away_touches_mid_3rd,
# away_touches_att_3rd,away_carries,away_carries_progressive_distance,away_tackles,away_tackles_won

class GeneralFeatures:
    def __init__(self):
        data_dir = "/Users/lukemciver/football/football-prediction/finalData.csv"
        self.df = pd.read_csv(data_dir)  
        df = self.df  
        
        print("\nActual columns in the CSV file:")
        print(df.columns.tolist())
        print(f"\nNumber of columns: {len(df.columns)}")
        
        #cjanged cols to match csv cols
        df.columns = [
            'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 
            'HTHG', 'HTAG', 'HTR', 'Referee', 'HS', 'AS', 
            'HST', 'AST', 'HC', 'AC', 'HF', 'AF', 
            'HY', 'AY', 'HR', 'AR', 
            'home_possession', 'away_possession',
            'home_passes_completed', 'home_passes_pct',
            'home_progressive_passes', 'home_progressive_passing_distance',
            'home_xg', 'home_take_ons_won', 'home_take_ons',
            'home_interceptions', 'home_blocks', 'home_touches',
            'home_touches_def_3rd', 'home_touches_mid_3rd',
            'home_touches_att_3rd', 'home_carries',
            'home_carries_progressive_distance', 'home_tackles',
            'home_tackles_won', 'away_passes_completed',
            'away_passes_pct', 'away_progressive_passes',
            'away_progressive_passing_distance', 'away_xg',
            'away_take_ons_won', 'away_take_ons', 'away_interceptions',
            'away_blocks', 'away_touches', 'away_touches_def_3rd',
            'away_touches_mid_3rd', 'away_touches_att_3rd',
            'away_carries', 'away_carries_progressive_distance',
            'away_tackles', 'away_tackles_won'
        ]
        
        df['Home Goal Conversion Rate'] = df['FTHG'] / df['HST'].replace(0, 1)
        df['Away Goal Conversion Rate'] = df['FTAG'] / df['AST'].replace(0, 1)

        attacking_features_home = self._get_feature_weights([
            'HST',  # Home Shots on Target
            'HS',   # Home Shots
            'HC'    # Home Corners
        ], df['FTR'])  # Full Time Result

        df['Home Attacking Intensity'] = (
            attacking_features_home[0] * df['HST'] +
            attacking_features_home[1] * (df['HS'] - df['HST']) +
            attacking_features_home[2] * df['HC']
        )
        
        attacking_features_away = self._get_feature_weights([
            'AST',  # Away Shots on Target
            'AS',   # Away Shots
            'AC'    # Away Corners
        ], df['FTR'])  # full time time res

        df['Away Attacking Intensity'] = (
            attacking_features_away[0] * df['AST'] +
            attacking_features_away[1] * (df['AS'] - df['AST']) +
            attacking_features_away[2] * df['AC']
        )

        disciplinary_features_home = self._get_feature_weights([
            'HF',  # Home Fouls
            'HY',  # Home Yellow Cards
            'HR'   # Home Red Cards
        ], df['FTR'])  

        df['Home Disciplinary Pressure'] = (
            disciplinary_features_home[0] * df['HF'] +
            disciplinary_features_home[1] * df['HY'] +
            disciplinary_features_home[2] * df['HR']
        )

        disciplinary_features_away = self._get_feature_weights([
            'AF',  # Away Fouls
            'AY',  # Away Yellow Cards
            'AR'   # Away Red Cards
        ], df['FTR'])  

        df['Away Disciplinary Pressure'] = (
            disciplinary_features_away[0] * df['AF'] +
            disciplinary_features_away[1] * df['AY'] +
            disciplinary_features_away[2] * df['AR']
        )

        df['Attacking Intensity Difference'] = df['Home Attacking Intensity'] - df['Away Attacking Intensity']
        df['Disciplinary Pressure Difference'] = df['Home Disciplinary Pressure'] - df['Away Disciplinary Pressure']

        df['xG Home'] = df['Home Goal Conversion Rate'] * df['Home Attacking Intensity']
        df['xG Away'] = df['Away Goal Conversion Rate'] * df['Away Attacking Intensity']

        df['Match Outcome'] = df['FTR'].map({'H': 1, 'D': 0, 'A': -1})
        
        df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
        df = df.sort_values(by='Date')
        df['Season'] = df['Date'].apply(self.get_season)

        attacking_features_home = self._get_feature_weights([
            'home_xg',
            'home_progressive_passes',
            'home_progressive_passing_distance',
            'home_take_ons_won',
            'home_touches_att_3rd',
            'home_carries_progressive_distance'
        ], df['FTR'])
        
        for feature, weight in zip([
            'home_xg',
            'home_progressive_passes',
            'home_progressive_passing_distance',
            'home_take_ons_won',
            'home_touches_att_3rd',
            'home_carries_progressive_distance'
        ], attacking_features_home):
            print(f"{feature}: {weight:.4f}")

        df['Home Attacking Strength'] = sum(
            weight * df[feature] for weight, feature in 
            zip(attacking_features_home, [
                'home_xg',
                'home_progressive_passes',
                'home_progressive_passing_distance',
                'home_take_ons_won',
                'home_touches_att_3rd',
                'home_carries_progressive_distance'
            ])
        )

        attacking_features_away = self._get_feature_weights([
            'away_xg',
            'away_progressive_passes',
            'away_progressive_passing_distance',
            'away_take_ons_won',
            'away_touches_att_3rd',
            'away_carries_progressive_distance'
        ], df['FTR'])
        
        print("\nAway Attacking Features Weights:")
        for feature, weight in zip([
            'away_xg',
            'away_progressive_passes',
            'away_progressive_passing_distance',
            'away_take_ons_won',
            'away_touches_att_3rd',
            'away_carries_progressive_distance'
        ], attacking_features_away):
            print(f"{feature}: {weight:.4f}")

        df['Away Attacking Strength'] = sum(
            weight * df[feature] for weight, feature in 
            zip(attacking_features_away, [
                'away_xg',
                'away_progressive_passes',
                'away_progressive_passing_distance',
                'away_take_ons_won',
                'away_touches_att_3rd',
                'away_carries_progressive_distance'
            ])
        )

        defensive_features_home = self._get_feature_weights([
            'home_interceptions',
            'home_blocks',
            'home_touches_def_3rd',
            'home_tackles',
            'home_tackles_won'
        ], df['FTR'])
        
        print("\nHome Defensive Features Weights:")
        for feature, weight in zip([
            'home_interceptions',
            'home_blocks',
            'home_touches_def_3rd',
            'home_tackles',
            'home_tackles_won'
        ], defensive_features_home):
            print(f"{feature}: {weight:.4f}")

        df['Home Defensive Strength'] = sum(
            weight * df[feature] for weight, feature in 
            zip(defensive_features_home, [
                'home_interceptions',
                'home_blocks',
                'home_touches_def_3rd',
                'home_tackles',
                'home_tackles_won'
            ])
        )

        defensive_features_away = self._get_feature_weights([
            'away_interceptions',
            'away_blocks',
            'away_touches_def_3rd',
            'away_tackles',
            'away_tackles_won'
        ], df['FTR'])
        
        print("\nAway Defensive Features Weights:")
        for feature, weight in zip([
            'away_interceptions',
            'away_blocks',
            'away_touches_def_3rd',
            'away_tackles',
            'away_tackles_won'
        ], defensive_features_away):
            print(f"{feature}: {weight:.4f}")

        df['Away Defensive Strength'] = sum(
            weight * df[feature] for weight, feature in 
            zip(defensive_features_away, [
                'away_interceptions',
                'away_blocks',
                'away_touches_def_3rd',
                'away_tackles',
                'away_tackles_won'
            ])
        )
        df['Attacking Strength Difference'] = df['Home Attacking Strength'] - df['Away Attacking Strength']
        df['Defensive Strength Difference'] = df['Home Defensive Strength'] - df['Away Defensive Strength']

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

    def _get_feature_weights(self, feature_names, target):
        """
        Calculate feature importance weights using Random Forest
        """
        X = self.df[feature_names]
        y = target
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # train rf 
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_scaled, y)
        
        # get feature importances and normalize 
        importances = rf.feature_importances_
        return importances / importances.sum()

if __name__ == "__main__":
    # Create an instance of GeneralFeatures
    gf = GeneralFeatures()
    
    # Get the dataframe
    df = gf.get_df_with_features()
    
    print("\nDataset Information:")
    print(f"Number of rows: {len(df)}")
    print(f"Number of columns: {len(df.columns)}")
    print("\nFirst few rows of key features:")
    print(df[['Date', 'HomeTeam', 'AwayTeam', 'FTR', 
              'Home Attacking Strength', 'Away Attacking Strength']].head())

        