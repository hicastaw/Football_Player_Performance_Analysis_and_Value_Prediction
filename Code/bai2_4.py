import pandas as pd

file_path = r'E:\ket_qua_bai_tap_lon\Code\results2.csv'
df= pd.read_csv(file_path)

teams= set(df.iloc[1:,1])

danh_sach_highest = {team : 0 for team in teams}

headers=list(df.columns)[3::3]

for head in headers:
    numberic=df[head].apply(pd.to_numeric, errors='coerce')
    if numberic.isna().all():
        continue    
    idx_max=numberic.idxmax()
    team=df.iloc[idx_max, 1]
    danh_sach_highest[team] +=1
    print(f"Highest {head} is {team}")
team_win=max(danh_sach_highest, key=danh_sach_highest.get)
print(f"Highest team is {team_win}")
