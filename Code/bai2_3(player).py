import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
# Function to convert age from 'year-days' format to decimal years
def convert_age(age_str):
    if isinstance(age_str, str) and '-' in age_str:
        try:
            # Split the 'year-days' format into year and days, then convert to decimal years
            year, days = map(int, age_str.split('-'))
            return round((year + days / 365), 2)  # Convert to decimal years
        except:
            return None  # Return None if there's an error in conversion
    return None  # Return None if the format is not 'year-days'

# Load the data from the specified file path
file_path = r'Code\\results.csv'
df = pd.read_csv(file_path)

# Convert the 'Age' column to decimal years using the function
df['Age'] = df['Age'].apply(convert_age)

# Define the list of statistical columns we want to create histograms for
headers = ["Gls", "shooting_standard_sotpct", "shooting_standard_sot_per90", 
           "defense_tackles_tklw", "defense_challenges_att", "defense_challenges_lost"
           ]

# Loop through each column in the 'headers' list to create a histogram
for head in headers:
    # Make a copy of the dataframe to avoid modifying the original one
    df_copy = df.copy()
    
    # Convert the column to numeric, forcing errors to NaN
    df_copy[head] = pd.to_numeric(df_copy[head], errors='coerce')
    
    # Drop rows where the current column has NaN values
    df_copy = df_copy.dropna(subset=[head])
    
    # Create the histogram plot
    plt.figure(figsize=(10, 6))  # Set the figure size
    plt.hist(df_copy[head], bins=30, edgecolor='red')  # Plot the histogram with 30 bins and red edges
    
    # Set the title and labels
    plt.title(f'Histogram of {head}')
    plt.xlabel(head)
    plt.ylabel('Frequency')
    
    # Add grid lines for better readability
    plt.grid(axis='y', alpha=0.75)
    
    # Display the plot
    plt.show()
