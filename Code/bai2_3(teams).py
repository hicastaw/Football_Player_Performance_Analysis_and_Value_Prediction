import pandas as pd
import matplotlib.pyplot as plt

def convert_age(age_str):
    if isinstance(age_str, str) and '-' in age_str:
        try:
            year, days = map(int, age_str.split('-'))
            return round((year + days / 365),2)
        except:
            return None
    return None
file_path = r'E:\ket_qua_bai_tap_lon\Code\results.csv'
df=pd.read_csv(file_path)
teams= sorted(list(set(list(df['Team']))))
headers = list(df.columns)[4:]

for team in teams:
    df_copy=df.copy()
    df_copy['Age'] = df_copy['Age'].apply(convert_age)
    df_copy = df_copy[df_copy['Team'] == team]
    for head in headers:
        df_copy[head] = pd.to_numeric(df_copy[head], errors='coerce')
        df_copy = df_copy.dropna(subset=[head])
        plt.figure(figsize=(10, 6))
        plt.hist(df_copy[head], bins=30, edgecolor='red')
        plt.title(f'Histogram of {head} for {team}')
        plt.xlabel(head)
        plt.ylabel('Frequency')
        plt.grid(axis='y', alpha=0.75,linestyle='--',color='gray')
        plt.show()
        
