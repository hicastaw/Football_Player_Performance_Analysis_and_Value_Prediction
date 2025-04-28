import pandas as pd
import numpy as np

def convert_age(age_str):
    if isinstance(age_str, str) and '-' in age_str:
        try:
            year, days = map(int, age_str.split('-'))
            return round((year + days / 365),2)
        except:
            return None
    return None  # nếu không phải chuỗi hoặc sai định dạng

df= pd.read_csv("results.csv")
df['Age'] = df['Age'].apply(convert_age)
headers=list(df.columns)
a=[]
for header in headers[4:]:
    df_copy=df.copy()
    df_copy[header] = pd.to_numeric(df_copy[header], errors='coerce')
    df_copy = df_copy.dropna(subset=[header])
    df_tmp=df_copy.nlargest(3,header)
    a.append(f'-------------Top 3 player highest at {header}: -------------')
    a.append(df_tmp.iloc[:, [0,1, 2, 3, df_tmp.columns.get_loc(header)]])
    df_tmp_2=df_copy.nsmallest(3,header)
    a.append(f'-------------Bottom 3 player lowest at {header}: -------------')
    a.append(df_tmp_2.iloc[:, [0,1, 2, 3, df_tmp_2.columns.get_loc(header)]])
file_path = r'E:\ket_qua_bai_tap_lon\Code\top_3.txt'
with open(file_path, 'w', encoding='utf-8') as f:
    for item in a:
        if isinstance(item, str):
            f.write(item + '\n')
        else:
            f.write(item.to_string(index=False) + '\n\n')  # In DataFrame đẹp hơn

