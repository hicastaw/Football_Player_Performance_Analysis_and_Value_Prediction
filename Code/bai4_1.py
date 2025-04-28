import pandas as pd
import requests
from rapidfuzz import fuzz, process
from unidecode import unidecode
import unicodedata
import re
from tmp import tra_ve_value
# 1) Các hàm chuẩn hóa và preprocess
def normalize_text(s: str) -> str:
    # Bỏ accent
    s = unidecode(s)
    # Chuyển lowercase và chỉ giữ chữ, số
    return s.strip()

# Token‑sort để bất chấp thứ tự
def preprocess_for_matching(s: str) -> str:
    tokens = normalize_text(s).split()
    tokens.sort()
    return " ".join(tokens)

# 2) Hàm tìm match tên cầu thủ với ngưỡng fuzz
def match_player_name(name: str, df: pd.DataFrame, threshold: int = 75) -> str:
    name_proc = preprocess_for_matching(name)
    # Chuẩn bị list và list đã preprocess
    choices_raw = df['Player'].tolist()
    choices_proc = [preprocess_for_matching(x) for x in choices_raw]

    # Fuzzy match
    match = process.extractOne(
        query=name_proc,
        choices=choices_proc,
        scorer=fuzz.ratio
    )     
    if match and match[1] >= threshold:
        idx = match[2]
        return choices_raw[idx]
    return None
def xy_ly_ten_1(name,team):
    res1=list(name.split())
    res1+=list(team.split())
    return "%20".join(res1)
def xy_ly_ten_2(name,team):
    res1=list(name.split())
    res1+=list(team.split())
    return " ".join(res1)


# 3) Đọc và ghép dữ liệu từ API
link = 'https://www.footballtransfers.com/us/values/actions/most-valuable-football-players/overview'
headers={
   'authority':'www.footballtransfers.com',
   'accept':'*/*',
   'content-type':'application/x-www-form-urlencoded; charset=UTF-8',
   'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36'
}
data={
    'orderBy': 'estimated_value',
    'orderByDescending': 1,
    'page': 1,
    'pages': 0,
    'pageItems': 25,
    'positionGroupId': all,
    'mainPositionId': 'all',
    'playerRoleId': all,
    'age': 'all',
    'countryId': 'all',
    'tournamentId': 31
}
data_player=[]
for i in range(1,23):
    print(f"Đang tải trang {i}...")
    data['page']=i
    respones=requests.post(url=link,data=data,headers=headers)
    du_lieu=respones.json()
    d_t=du_lieu['records']
    df=pd.DataFrame(d_t)
    df=df[['player_name','team_name','estimated_value']]
    data_player.append(df)

df_results = pd.concat(data_player, ignore_index=True)
new_header=['Player','Team','Value']
df_results.columns=new_header
# 4) Đọc file chứa cột Min
file_path = r'E:\ket_qua_bai_tap_lon\Code\results.csv'
df_tmp = pd.read_csv(file_path)  # cột 'Player' và 'Min'

df_tmp['matched_name'] = df_tmp['Player'].apply(
    lambda x: match_player_name(x, df_results, threshold=85)
)

# Kiểm tra cột 'matched_name' đã được tạo hay chưa
# Tiến hành merge sau khi chắc chắn cột 'matched_name' đã có
df_merged = pd.merge(
    df_tmp[['Player','Team','Min','matched_name']],  # Chỉ lấy các cột cần thiết
    df_results[['Player', 'Value']],  # Chỉ lấy các cột cần thiết
    left_on='matched_name',  # Dùng 'matched_name' từ df_results
    right_on='Player',  # Dùng 'Player' từ df_tmp
    how='left',  # Thực hiện left join
    suffixes=('', '_tmp')  # Thêm suffix nếu có trùng tên cột
)

# 6) Lọc theo Min > 900 (hoặc ngưỡng tuỳ chọn)
result_filtered = df_merged[df_merged['Min'] > 900].reset_index(drop=True)
head_result=['Player','Team','Min','Value']
result_filtered=result_filtered[head_result]
result_filtered = result_filtered.sort_values(by='Player', ascending=True).reset_index(drop=True)
#  them value bi thieu
result_tmp = result_filtered[result_filtered['Value'].isna()]
for idx,row in result_tmp.iterrows():
    name=unidecode(row['Player'])
    team=row['Team']
    name_header=xy_ly_ten_1(name,team)
    name_payload=xy_ly_ten_2(name,team)
    value=tra_ve_value(name,team)
    if value is not None:
        result_filtered.at[idx, 'Value'] = value
    else:
        print(f"Không tìm thấy giá trị cho cầu thủ: {name} - {team}")
    

# 7) Xuất kết quả
file_path = r'E:\ket_qua_bai_tap_lon\Code\results4.csv'
result_filtered.to_csv(file_path, index=False, encoding='utf-8-sig')
print("Hoàn thành! File 'results6.csv' đã được lưu.")

