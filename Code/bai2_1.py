import pandas as pd 
import numpy as np

# Function to convert age string format 'year-days' into a decimal number
def convert_age(age_str):
    if isinstance(age_str, str) and '-' in age_str:
        try:
            year, days = map(int, age_str.split('-'))
            return round((year + days / 365), 2)
        except:
            return None
    return None  # Return None if not a string or wrong format

# Read CSV file
file_path = r'E:\ket_qua_bai_tap_lon\Code\results.csv'  # use raw string to avoid path errors
df = pd.read_csv(file_path)

# Apply age conversion
df['Age'] = df['Age'].apply(convert_age)

# Get list of column headers
headers = list(df.columns)

a = []

# Iterate over statistical columns (starting from column index 4)
for header in headers[4:]:
    df_copy = df.copy()
    # Convert current column to numeric, ignore errors
    df_copy[header] = pd.to_numeric(df_copy[header], errors='coerce')
    # Drop rows with NaN in the current column
    df_copy = df_copy.dropna(subset=[header])
    # Get top 3 players with highest values in the current statistic
    df_tmp = df_copy.nlargest(3, header)
    a.append(f'-------------Top 3 players with highest {header}: -------------')
    a.append(df_tmp.iloc[:, [0, 1, 2, 3, df_tmp.columns.get_loc(header)]])
    # Get bottom 3 players with lowest values in the current statistic
    df_tmp_2 = df_copy.nsmallest(3, header)
    a.append(f'-------------Bottom 3 players with lowest {header}: -------------')
    a.append(df_tmp_2.iloc[:, [0, 1, 2, 3, df_tmp_2.columns.get_loc(header)]])

# Export results to text file
file_path = r'Code\\top_3.txt'
with open(file_path, 'w', encoding='utf-8') as f:
    for item in a:
        if isinstance(item, str):
            f.write(item + '\n')
        else:
            f.write(item.to_string(index=False) + '\n\n')  # Print DataFrame nicely
