import pandas as pd
import requests
from rapidfuzz import fuzz, process
from unidecode import unidecode
import unicodedata
import re
from tmp import tra_ve_value

# 1) Text normalization and preprocessing functions

# Function to normalize text (remove accents and lowercase)
def normalize_text(s: str) -> str:
    # Remove accents from characters
    s = unidecode(s)
    # Convert to lowercase and keep only letters and numbers
    return s.strip()

# Token-sort function to ignore word order while matching
def preprocess_for_matching(s: str) -> str:
    tokens = normalize_text(s).split()
    tokens.sort()  # Sort tokens to make the matching order-insensitive
    return " ".join(tokens)

# 2) Function to match player names with a fuzz threshold
def match_player_name(name: str, df: pd.DataFrame, threshold: int = 75) -> str:
    name_proc = preprocess_for_matching(name)  # Preprocess the input name
    # Prepare lists of raw and processed names from the dataframe
    choices_raw = df['Player'].tolist()
    choices_proc = [preprocess_for_matching(x) for x in choices_raw]

    # Fuzzy matching using RapidFuzz
    match = process.extractOne(
        query=name_proc,
        choices=choices_proc,
        scorer=fuzz.ratio
    )     
    # If a match is found above the threshold, return the matched player's name
    if match and match[1] >= threshold:
        idx = match[2]
        return choices_raw[idx]
    return None  # Return None if no match is found

# 2.5) Function to process player and team name (for URL encoding purposes)
def xy_ly_ten_1(name, team):
    res1 = list(name.split())  # Split name into tokens
    res1 += list(team.split())  # Split team name and append to list
    return "%20".join(res1)  # Return a URL-safe string

def xy_ly_ten_2(name, team):
    res1 = list(name.split())  # Split name into tokens
    res1 += list(team.split())  # Split team name and append to list
    return " ".join(res1)  # Return a simple space-separated string

# 3) API request to fetch player data
link = 'https://www.footballtransfers.com/us/values/actions/most-valuable-football-players/overview'
headers = {
    'authority':'www.footballtransfers.com',
    'accept':'*/*',
    'content-type':'application/x-www-form-urlencoded; charset=UTF-8',
    'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36'
}
data = {
    'orderBy': 'estimated_value',
    'orderByDescending': 1,
    'page': 1,
    'pages': 0,
    'pageItems': 25,
    'positionGroupId': 'all',
    'mainPositionId': 'all',
    'playerRoleId': 'all',
    'age': 'all',
    'countryId': 'all',
    'tournamentId': 31
}

# Loop through pages of API to collect player data
data_player = []
for i in range(1, 23):
    print(f"Processing page {i}...")  # Page loading message
    data['page'] = i
    response = requests.post(url=link, data=data, headers=headers)
    data_json = response.json()
    player_data = data_json['records']
    df = pd.DataFrame(player_data)
    df = df[['player_name', 'team_name', 'estimated_value']]  # Only keep necessary columns
    data_player.append(df)  # Add current page's data to the list

# Concatenate all data into a single DataFrame
df_results = pd.concat(data_player, ignore_index=True)
new_header = ['Player', 'Team', 'Value']
df_results.columns = new_header  # Rename columns to match expected names

# 4) Read the results CSV file containing 'Player' and 'Min' columns
file_path = r'E:\ket_qua_bai_tap_lon\Code\results.csv'
df_tmp = pd.read_csv(file_path)

# 5) Match player names from the CSV with the data from the API using fuzzy matching
df_tmp['matched_name'] = df_tmp['Player'].apply(
    lambda x: match_player_name(x, df_results, threshold=85)  # Apply fuzzy matching with threshold of 85
)

# 6) Merge the data using the matched player names
df_merged = pd.merge(
    df_tmp[['Player', 'Team', 'Min', 'matched_name']],  # Use relevant columns from df_tmp
    df_results[['Player', 'Value']],  # Use relevant columns from API data
    left_on='matched_name',  # Merge using matched player names
    right_on='Player',  # Merge on 'Player' column
    how='left',  # Left join to keep all rows from df_tmp
    suffixes=('', '_tmp')  # Add suffix if column names overlap
)

# 7) Filter the merged data by 'Min' value > 900 (or any chosen threshold)
result_filtered = df_merged[df_merged['Min'] > 900].reset_index(drop=True)
head_result = ['Player', 'Team', 'Min', 'Value']  # Define desired column order
result_filtered = result_filtered[head_result]  # Reorder columns
result_filtered = result_filtered.sort_values(by='Player', ascending=True).reset_index(drop=True)

# 8) Handle missing 'Value' data by looking up the value using external function
result_tmp = result_filtered[result_filtered['Value'].isna()]
for idx, row in result_tmp.iterrows():
    name = unidecode(row['Player'])  # Normalize the player's name
    team = row['Team']
    name_header = xy_ly_ten_1(name, team)  # Prepare for URL encoding
    name_payload = xy_ly_ten_2(name, team)  # Prepare the query string
    value = tra_ve_value(name, team)  # Look up the value using the external function
    if value is not None:
        result_filtered.at[idx, 'Value'] = value  # Update the missing value
    else:
        print(f"Could not find value for player: {name} - {team}")  # Print message if value not found

# 9) Export the final result to a CSV file
file_path = r'E:\ket_qua_bai_tap_lon\Code\results4.csv'
result_filtered.to_csv(file_path, index=False, encoding='utf-8-sig')
print("Completed! The file 'results4.csv' has been saved.")  # Completion message
