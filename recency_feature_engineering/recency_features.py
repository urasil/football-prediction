import pandas as pd
from recency_feature_engineering.general_features import GeneralFeatures

class RecencyFeatures:
    def __init__(self, max_x=5):
        self.max_x = max_x
        general = GeneralFeatures()
        self.df = general.get_df_with_features().sort_values(by='Date')

    def avg_stats_last_x_games(self):
    
        def calculate_running_avg(stats_list, x):
            if len(stats_list) < x:
                return None
            return sum(stats_list[-x:]) / x

        metrics = ['Goals Scored', 'Goals Conceded', 'xG', 'Disciplinary Pressure', 'Recent Performance']
        for metric in metrics:
            for x in range(1, self.max_x + 1):
                self.df[f'Avg {metric} Home Last {x} Games'] = None
                self.df[f'Avg {metric} Away Last {x} Games'] = None
        running_stats = {}

        for idx, row in self.df.iterrows():
            for team_type in ['Home Team', 'Away Team']:
                team = row[team_type]
                stat_type = team_type.split()[0]  # Home or Away

                if team not in running_stats:
                    running_stats[team] = {
                        'Goals Scored': [],
                        'Goals Conceded': [],
                        'xG': [],
                        'Disciplinary Pressure': [],
                        'Recent Performance': []
                    }

                team_stats = running_stats[team]

                for x in range(1, self.max_x + 1):
                    for metric in metrics:
                        avg_stat = calculate_running_avg(team_stats[metric], x)
                        self.df.at[idx, f'Avg {metric} {stat_type} Last {x} Games'] = avg_stat

                team_stats['Goals Scored'].append(row['Full Time Home Goals'] if stat_type == 'Home' else row['Full Time Away Goals'])
                team_stats['Goals Conceded'].append(row['Full Time Away Goals'] if stat_type == 'Home' else row['Full Time Home Goals'])
                team_stats['xG'].append(row['xG Home'] if stat_type == 'Home' else row['xG Away'])
                team_stats['Disciplinary Pressure'].append(row['Home Disciplinary Pressure'] if stat_type == 'Home' else row['Away Disciplinary Pressure'])
                team_stats['Recent Performance'].append(1 if (row['Full Time Result'] == 'H' and stat_type == 'Home') or (row['Full Time Result'] == 'A' and stat_type == 'Away') else 0.5 if row['Full Time Result'] == 'D' else 0)
        
    def get_recency_features(self):
        self.avg_stats_last_x_games()
        return self.df
