import pandas as pd
import numpy as np

def convert_age(age_str):
    if isinstance(age_str, str) and '-' in age_str:
        try:
            year, days = map(int, age_str.split('-'))
            return round((year + days / 365),2)
        except:
            return None
    return None

# Đọc file và xử lý tuổi
df = pd.read_csv("results.csv")
df['Age'] = df['Age'].apply(convert_age)
headers = list(df.columns)

# Danh sách tiêu đề hàng đầu
a = ['']
for head in headers[4:]:
    a.append(f'Median of {head}')
    a.append(f'Mean of {head}')
    a.append(f'Std of {head}')

# Lấy danh sách đội
teams = sorted(list(set(list(df['Team']))))
teams = ['all'] + teams  # thêm dòng tổng hợp tất cả các đội
# Dòng dữ liệu kết quả
result_rows = []
# Tính toán theo từng đội
for team in teams:
    if team=='all':
        res=['all']
        for head in headers[4:]:
            group_copy=df.copy()
            group_copy[head] = pd.to_numeric(group_copy[head], errors='coerce')
            res.append(round(group_copy[head].median(),2))
            res.append(round(group_copy[head].mean(),2))
            res.append(round(group_copy[head].std(),2))
        result_rows.append(res)
    else:
        res = [team]  # tên đội là cột đầu tiên
        group = df[df['Team'] == team]  # lọc theo đội
        for head in headers[4:]:
            group_copy=group.copy()
            group_copy[head] = pd.to_numeric(group_copy[head], errors='coerce')  # chuyển về số nếu cần
            res.append(round(group_copy[head].median(),2))
            res.append(round(group_copy[head].mean(),2))
            res.append(round(group_copy[head].std(),2))
        
        result_rows.append(res)

# Đưa về DataFrame kết quả
summary_df = pd.DataFrame(result_rows, columns=a)


# (tuỳ chọn) Lưu ra file Excel hoặc CSV
file_path = r'E:\ket_qua_bai_tap_lon\Code\results2.csv'
summary_df.to_csv(file_path, index=True,encoding='utf-8-sig')