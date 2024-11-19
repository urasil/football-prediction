import pandas as pd
from bs4 import BeautifulSoup
import requests
import re
import os
from datetime import datetime
import time

def extract_team_stats(soup, team_id):
    stats = {}
    
    # Get the stats tables for the team
    summary_table = soup.find('div', id=f'div_stats_{team_id}_summary')
    possession_table = soup.find('div', id=f'div_stats_{team_id}_possession') 
    passing_table = soup.find('div', id=f'div_stats_{team_id}_passing')
    
    if not all([summary_table, possession_table, passing_table]):
        print(f"Missing tables for team {team_id}")
        return None
        
    # footer rows for totals
    summary_footer = summary_table.find('tfoot').find('tr') if summary_table else None
    possession_footer = possession_table.find('tfoot').find('tr') if possession_table else None
    passing_footer = passing_table.find('tfoot').find('tr') if passing_table else None

    
    if not all([summary_footer, possession_footer, passing_footer]):
        print(f"Missing footer rows for team {team_id}")
        return None

    # make sure tables are acc there 
    def safe_extract(footer, stat_name, convert_type=int):
        try:
            cell = footer.find('td', {'data-stat': stat_name})
            if cell and cell.text.strip():
                return convert_type(cell.text.strip())
            return None
        except Exception as e:
            print(f"Error extracting {stat_name}: {str(e)}")
            return None

    # Passing stats
    stats['passes_completed'] = safe_extract(passing_footer, 'passes_completed')
    stats['passes_pct'] = safe_extract(passing_footer, 'passes_pct', float)
    stats['progressive_passes'] = safe_extract(passing_footer, 'progressive_passes')
    stats['progressive_passing_distance'] = safe_extract(passing_footer, 'passes_progressive_distance')

    # Summary stats
    stats['xg'] = safe_extract(summary_footer, 'xg', float)
    stats['shots'] = safe_extract(summary_footer, 'shots')
    stats['shots_on_target'] = safe_extract(summary_footer, 'shots_on_target')
    stats['take_ons_won'] = safe_extract(summary_footer, 'take_ons_won')
    stats['take_ons'] = safe_extract(summary_footer, 'take_ons')
    stats['interceptions'] = safe_extract(summary_footer, 'interceptions')
    stats['blocks'] = safe_extract(summary_footer, 'blocks')

    # Possession stats
    stats['touches'] = safe_extract(possession_footer, 'touches')
    stats['touches_def_3rd'] = safe_extract(possession_footer, 'touches_def_3rd')
    stats['touches_mid_3rd'] = safe_extract(possession_footer, 'touches_mid_3rd')
    stats['touches_att_3rd'] = safe_extract(possession_footer, 'touches_att_3rd')
    stats['touches_def_pen'] = safe_extract(possession_footer, 'touches_def_pen')
    stats['touches_att_pen'] = safe_extract(possession_footer, 'touches_att_pen')
    stats['carries'] = safe_extract(possession_footer, 'carries')
    stats['carries_total_distance'] = safe_extract(possession_footer, 'carries_total_distance')
    stats['carries_progressive_distance'] = safe_extract(possession_footer, 'carries_progressive_distance')

    return stats
    
def scrape_match(match_url):
    """Scrape stats for a single match"""
    print(f"Fetching match page: {match_url}")
    
    response = requests.get(match_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    team_tables = soup.find_all('div', id=re.compile(r'div_stats_[a-f0-9]{8}_summary'))
    team_ids = [table['id'].split('_')[2] for table in team_tables]
    
    # Extract stats for both teams
    home_stats = extract_team_stats(soup, team_ids[0])
    away_stats = extract_team_stats(soup, team_ids[1])
    
    #is it getting btoh?????????
    if not home_stats or not away_stats:
        print("Failed to extract stats for one or both teams")
        return None
        
    match_stats = {
        'match_url': match_url,
        'home_team_id': team_ids[0],
        'away_team_id': team_ids[1]
    }
    
    # add prefixes home/away
    for k,v in home_stats.items():
        match_stats[f'home_{k}'] = v
    for k,v in away_stats.items():
        match_stats[f'away_{k}'] = v
        
    return match_stats


def get_match_urls(season_url=None):
    if season_url is None:
        season_url = 'https://fbref.com/en/comps/9/2017-2018/schedule/2017-2018-Premier-League-Scores-and-Fixtures'
        
    response = requests.get(season_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    matches = []
    for row in soup.select('table.stats_table tbody tr'):
        report_cell = row.select_one('td[data-stat="match_report"] a')
        if report_cell and 'href' in report_cell.attrs:
            match_url = f"https://fbref.com{report_cell['href']}"
            matches.append(match_url)
            
    return matches

def scrape_match_stats(match_url):
    """Scrape detailed stats from a match report"""
    print(f"Scraping {match_url}")
    
    response = requests.get(match_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    match_stats = {
        'match_url': match_url
    }

    # Get match date and change to  dd/mm/yyyy
    date_element = soup.select_one('.scorebox_meta div:first-child strong a')
    if date_element:
        try:
            date_str = date_element.text
            date_obj = datetime.strptime(date_str, '%A %B %d, %Y')
            match_stats['date'] = date_obj.strftime('%d/%m/%Y')
        except Exception as e:
            print(f"Error parsing date: {e}")
            match_stats['date'] = date_element.text

    # Get team names
    home_team = soup.select_one('.scorebox > div:first-child strong a')
    away_team = soup.select_one('.scorebox > div:nth-child(2) strong a')
    if home_team and away_team:
        match_stats['home_team'] = home_team.text
        match_stats['away_team'] = away_team.text

    # Get possession stats
    try:
        possession_row = soup.find('th', text='Possession').parent.find_next_sibling('tr')
        if possession_row:
            home_possession = possession_row.find_all('strong')[0].text.strip('%')
            away_possession = possession_row.find_all('strong')[1].text.strip('%')
            match_stats['home_possession'] = int(home_possession)
            match_stats['away_possession'] = int(away_possession)
    except Exception as e:
        print(f"Error extracting possession stats: {e}")

    # Get team stats
    team_ids = []
    stats_divs = soup.find_all('div', id=lambda x: x and x.startswith('div_stats_') and x.endswith('_summary'))
    
    for div in stats_divs:
        team_id = div['id'].split('_')[2]
        if team_id not in team_ids:
            team_ids.append(team_id)

    if len(team_ids) >= 2:
        home_id = team_ids[0]
        away_id = team_ids[1]
        
        home_stats = extract_team_stats(soup, home_id)
        if home_stats:
            home_stats = {f'home_{k}': v for k, v in home_stats.items()}
            match_stats.update(home_stats)
            
        away_stats = extract_team_stats(soup, away_id)
        if away_stats:
            away_stats = {f'away_{k}': v for k, v in away_stats.items()}
            match_stats.update(away_stats)

    if 'home_team' in match_stats and 'away_team' in match_stats:
        time.sleep(3)
        return match_stats
    else:
        print("Missing required team data")
        return None

def scrape_season(season_url=None):
    output_dir = 'scraped_data'
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f'match_stats_{timestamp}.csv')
    
    print(f"Will save data to: {output_file}")
    
    # get match urls
    match_urls = get_match_urls(season_url)
    matches = []
    
    # logging in case scraper faisl
    for match_url in match_urls:
        try:
            match_stats = scrape_match_stats(match_url)
            matches.append(match_stats)
        except Exception as e:
            print(f"Error scraping {match_url}: {e}")
    
    df = pd.DataFrame(matches)
    
    df.to_csv(output_file, index=False)
    print(f"\nSaved {len(df)} matches to {output_file}")
    
    return df

def generate_season_urls():
    base_url = "https://fbref.com/en/comps/9/{}-{}/schedule/{}-{}-Premier-League-Scores-and-Fixtures"
    seasons = []
    
    for year in range(2017, 2025):  
        season_start = str(year)
        season_end = str(year + 1)[-2:] 
        url = base_url.format(
            season_start, season_end,
            season_start, season_end
        )
        seasons.append(url)
    
    return seasons

if __name__ == "__main__":
    season_urls = generate_season_urls()
    
    for season_url in season_urls:
        print(f"\nScraping season: {season_url}")
        try:
            season_df = scrape_season(season_url)
            print(f"Season DataFrame shape: {season_df.shape}")
        except Exception as e:
            print(f"Error scraping season {season_url}: {e}")
        
        time.sleep(10)