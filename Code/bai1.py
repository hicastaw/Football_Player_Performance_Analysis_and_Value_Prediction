from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import pandas as pd
import time
import undetected_chromedriver as uc

# Function to convert nation name to abbreviation (only keep uppercase letters)
def convert_nation(nation_str):
    res = ''
    for x in nation_str:
        if x.isupper():
            res += x
    return res

# Initialize driver
options = uc.ChromeOptions()
service = Service(ChromeDriverManager().install())
driver = uc.Chrome(service=service, options=options)

# List of websites and corresponding table IDs
website = {
    'https://fbref.com/en/comps/9/stats/Premier-League-Stats': 'stats_standard',
    'https://fbref.com/en/comps/9/keepers/Premier-League-Stats': 'stats_keeper',
    'https://fbref.com/en/comps/9/shooting/Premier-League-Stats': 'stats_shooting',
    'https://fbref.com/en/comps/9/passing/Premier-League-Stats': 'stats_passing',
    'https://fbref.com/en/comps/9/gca/Premier-League-Stats': 'stats_gca',
    'https://fbref.com/en/comps/9/defense/Premier-League-Stats': 'stats_defense',
    'https://fbref.com/en/comps/9/possession/Premier-League-Stats': 'stats_possession',
    'https://fbref.com/en/comps/9/misc/Premier-League-Stats': 'stats_misc'
}

tables = {}

# Loop through all websites
for link, id in website.items():
    driver.get(link)
    time.sleep(3)
    print(f"Processing: {link}")
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    table = soup.find('table', id=id)
    if table:
        rows = table.find_all('tr')
        data = []
        for idx, row in enumerate(rows):
            cells = row.find_all(['th', 'td'])
            row_data = [cell.get_text(strip=True) for cell in cells]
            # Skip unwanted rows
            if not row_data or idx == 0 or (row_data[0] == 'Rk' and idx != 1):
                continue
            data.append(row_data[1:])

        # Get header from the first valid row
        tables[id] = pd.DataFrame(data[1:], columns=data[0])

# Process 'minutes played' data
df_stat_standard = tables['stats_standard']
df_stat_standard['Min'] = pd.to_numeric(df_stat_standard['Min'].str.replace(',', ''), errors='coerce')

# Filter players who played more than 90 minutes
df_stat_standard = df_stat_standard[df_stat_standard['Min'] > 90]
df_stat_standard = df_stat_standard.sort_values(by='Player')  # Sort players by name

bang = []

# Drop unnecessary columns for each table
idx_drop_standard = [5, 9, 12, 13, 14, 15, 21, 27, 28, 29, 32, 33, 34, 35]
df_stat_standard = df_stat_standard.drop(df_stat_standard.columns[idx_drop_standard], axis=1)
bang.append(df_stat_standard)

idx_drop_keeper = [11, 14, 19, 24]
cols = [0, 1, 2, 3] + idx_drop_keeper
tables['stats_keeper'] = tables['stats_keeper'].iloc[:, cols]
bang.append(tables['stats_keeper'])

idx_drop_shooting = [10, 12, 13, 15]
cols = [0, 1, 2, 3] + idx_drop_shooting
tables['stats_shooting'] = tables['stats_shooting'].iloc[:, cols]
bang.append(tables['stats_shooting'])

idx_drop_passing = [7, 9, 10, 14, 17, 20, 25, 26, 27, 28, 29]
cols = [0, 1, 2, 3] + idx_drop_passing
tables['stats_passing'] = tables['stats_passing'].iloc[:, cols]
bang.append(tables['stats_passing'])

idx_drop_gca = [7, 8, 15, 16]
cols = [0, 1, 2, 3] + idx_drop_gca
tables['stats_gca'] = tables['stats_gca'].iloc[:, cols]
bang.append(tables['stats_gca'])

idx_drop_defense = [7, 8, 13, 15, 16, 17, 18, 19]
cols = [0, 1, 2, 3] + idx_drop_defense
tables['stats_defense'] = tables['stats_defense'].iloc[:, cols]
bang.append(tables['stats_defense'])

idx_drop_possession = [7, 8, 9, 10, 11, 12, 14, 16, 18, 19, 21, 22, 23, 24, 25, 26, 27, 28]
cols = [0, 1, 2, 3] + idx_drop_possession
tables['stats_possession'] = tables['stats_possession'].iloc[:, cols]
bang.append(tables['stats_possession'])

idx_drop_misc = [10, 11, 12, 13, 19, 20, 21, 22]
cols = [0, 1, 2, 3] + idx_drop_misc
tables['stats_misc'] = tables['stats_misc'].iloc[:, cols]
bang.append(tables['stats_misc'])

# Merge all tables based on Player, Nation, Squad, and Pos
for table in bang[1:]:
    df_stat_standard = pd.merge(df_stat_standard, table, on=["Player", "Nation", "Squad", "Pos"], how='left')

# Remove duplicate players (keep first occurrence)
df_stat_standard = df_stat_standard.drop_duplicates(subset=['Player', 'Squad'])

# Replace missing values with "N/a"
df_stat_standard = df_stat_standard.replace("", "N/a").fillna("N/a")

# Convert nation names to abbreviations
df_stat_standard['Nation'] = df_stat_standard['Nation'].apply(convert_nation)

# Define final header
header = [
    "Player", "Nation", "Pos", "Team", "Age", "MP", "Starts", "Min", "Gls", "Ast", "CrdY", "CrdR", "xG", "xAG",
    "progression_prgc", "progression_prgp", "progression_prgr", "per90_gls", "per90_ast", "per90_xg", "per90_xag",
    "goalkeeping_performance_ga90", "goalkeeping_performance_savepct", "goalkeeping_performance_cspct", "goalkeeping_penalties_savepct",
    "shooting_standard_sotpct", "shooting_standard_sot_per90", "shooting_standard_g_sh", "shooting_standard_dist",
    "passing_total_cmp", "passing_total_cmppct", "passing_total_totdist", "passing_short_cmppct", "passing_medium_cmppct",
    "passing_long_cmppct", "passing_expected_kp", "passing_expected_1_3", "passing_expected_ppa", "passing_expected_crspa",
    "passing_expected_prgp", "creation_sca_sca", "creation_sca_sca90", "creation_gca_gca", "creation_gca_gca90",
    "defense_tackles_tkl", "defense_tackles_tklw", "defense_challenges_att", "defense_challenges_lost",
    "defense_blocks_blocks", "defense_blocks_sh", "defense_blocks_pass", "defense_blocks_int",
    "possession_touches_touches", "possession_touches_def_pen", "possession_touches_def_3rd",
    "possession_touches_mid_3rd", "possession_touches_att_3rd", "possession_touches_att_pen", "possession_takeons_att",
    "possession_takeons_succpct", "possession_takeons_tkldpct", "possession_carries_carries",
    "possession_carries_prgdist", "possession_carries_prgc", "possession_carries_1_3",
    "possession_carries_cpa", "possession_carries_mis", "possession_carries_dis", "possession_receiving_rec",
    "possession_receiving_prgr", "misc_performance_fls", "misc_performance_fld", "misc_performance_off",
    "misc_performance_crs", "misc_performance_recov", "misc_aerial_won", "misc_aerial_lost", "misc_aerial_wonpct"
]

# Swap "Team" and "Pos" columns
df_stat_standard.columns = header
cols = list(df_stat_standard.columns)
i, j = cols.index('Team'), cols.index('Pos')
cols[i], cols[j] = cols[j], cols[i]
df_stat_standard = df_stat_standard[cols]

# Save the final DataFrame to CSV
file_path = r'Code\\results.csv'  # use raw string to avoid path errors
df_stat_standard.to_csv(file_path, index=False, encoding='utf-8-sig')

# Close the driver
driver.quit()
