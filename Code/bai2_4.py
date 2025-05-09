import pandas as pd
from pathlib import Path
# Load the CSV file into a DataFrame
current_dir = Path(__file__).parent
file_path = current_dir / 'results2.csv'
df = pd.read_csv(file_path)

# Create a set of teams from the second column (ignoring the first row which is header)
teams = set(df.iloc[1:, 1])

# Initialize a dictionary to keep track of how many highest stats each team has
danh_sach_highest = {team: 0 for team in teams}

# Get the headers from the DataFrame starting from the fourth column and take every third column
headers = list(df.columns)[3::3]

# Iterate over each header (statistic category) in the list
for head in headers:
    # Convert the column data to numeric, coercing errors to NaN
    numberic = df[head].apply(pd.to_numeric, errors='coerce')
    
    # Skip the column if all values are NaN
    if numberic.isna().all():
        continue
    
    # Get the index of the highest value in the column
    idx_max = numberic.idxmax()
    
    # Get the team that has the highest value for this statistic
    team = df.iloc[idx_max, 1]
    
    # Update the dictionary by incrementing the count for the team with the highest stat
    danh_sach_highest[team] += 1
    
    # Print the team with the highest value for this stat
    print(f"Highest {head} is {team}")

# Determine the team that has the most highest stats
team_win = max(danh_sach_highest, key=danh_sach_highest.get)

# Print the team with the most highest stats
print(f"Highest team is {team_win}")
