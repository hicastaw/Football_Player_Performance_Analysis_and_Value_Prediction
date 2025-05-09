import pandas as pd
import numpy as np

# Function to convert age from 'year-days' format to decimal years
def convert_age(age_str):
    if isinstance(age_str, str) and '-' in age_str:
        try:
            # Split by '-' to separate years and days
            year, days = map(int, age_str.split('-'))
            return round((year + days / 365), 2)  # Convert to decimal years
        except:
            return None  # Return None if the conversion fails
    return None  # Return None if the format is not 'year-days'

# Read the CSV file and apply age conversion
file_path = r'Code\\results.csv'
df = pd.read_csv(file_path)
df['Age'] = df['Age'].apply(convert_age)  # Apply the convert_age function to the 'Age' column
headers = list(df.columns)  # List all column names

# Prepare the header for the result table
a = ['']
for head in headers[4:]:
    a.append(f'Median of {head}')
    a.append(f'Mean of {head}')
    a.append(f'Std of {head}')

# Get a list of unique team names and add 'all' for the overall summary
teams = sorted(list(set(list(df['Team']))))
teams = ['all'] + teams  # Add an 'all' category for the summary of all teams

# Initialize an empty list to hold the result rows
result_rows = []

# Calculate statistics for each team
for team in teams:
    if team == 'all':
        res = ['all']  # 'all' represents the overall data, not a specific team
        for head in headers[4:]:
            group_copy = df.copy()  # Create a copy of the dataframe
            group_copy[head] = pd.to_numeric(group_copy[head], errors='coerce')  # Convert to numeric
            # Calculate median, mean, and standard deviation for the column
            res.append(round(group_copy[head].median(), 2))
            res.append(round(group_copy[head].mean(), 2))
            res.append(round(group_copy[head].std(), 2))
        result_rows.append(res)  # Append the overall statistics to the result rows
    else:
        res = [team]  # Add the team name as the first column
        group = df[df['Team'] == team]  # Filter the dataframe by team
        for head in headers[4:]:
            group_copy = group.copy()  # Create a copy of the filtered dataframe
            group_copy[head] = pd.to_numeric(group_copy[head], errors='coerce')  # Convert to numeric
            # Calculate median, mean, and standard deviation for the column
            res.append(round(group_copy[head].median(), 2))
            res.append(round(group_copy[head].mean(), 2))
            res.append(round(group_copy[head].std(), 2))
        result_rows.append(res)  # Append the team's statistics to the result rows

# Create a DataFrame from the result rows
summary_df = pd.DataFrame(result_rows, columns=a)

# (Optional) Save the result to a CSV file
file_path = r'Code\\results2.csv'
summary_df.to_csv(file_path, index=True, encoding='utf-8-sig')  # Save to CSV with UTF-8 encoding
