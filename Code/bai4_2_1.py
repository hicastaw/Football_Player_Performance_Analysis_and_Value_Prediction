import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # Import seaborn để có giao diện đồ thị đẹp hơn (tùy chọn)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR # Có thể thêm SVR vào tuning
from sklearn.neural_network import MLPRegressor # Có thể thêm MLPRegressor vào tuning
import joblib
import warnings

# Bỏ qua cảnh báo (tùy chọn) - Cẩn thận khi sử dụng trong môi trường sản phẩm
warnings.filterwarnings('ignore')

# Hàm chuyển đổi giá trị tiền tệ
def convert_value(value_str):
    """
    Chuyển đổi chuỗi giá trị tiền tệ (ví dụ: '€10M', '€2.5B') sang số thực.
    Sửa lỗi nhân sai giá trị triệu và tỷ.
    """
    if isinstance(value_str, str):
        value_str = value_str.replace('€', '').replace(',', '').strip()
        if 'M' in value_str.upper(): # Sử dụng .upper() để xử lý cả 'm' và 'M'
            try:
                return float(value_str.upper().replace('M', '')) * 1e6 # 1 triệu là 1e6
            except ValueError:
                return None
        elif 'B' in value_str.upper(): # Sử dụng .upper() để xử lý cả 'b' và 'B'
            try:
                return float(value_str.upper().replace('B', '')) * 1e9 # 1 tỷ là 1e9
            except ValueError:
                return None
    return None # Trả về None nếu không phải chuỗi hoặc định dạng không khớp

# Hàm chuyển đổi tuổi (định dạng năm-ngày)
def convert_age(age_str):
    """
    Chuyển đổi chuỗi tuổi định dạng 'năm-ngày' sang số thực (ví dụ: '25-180' -> 25.49).
    """
    if isinstance(age_str, str) and '-' in age_str:
        try:
            year, days = map(int, age_str.split('-'))
            # Giả sử một năm có 365 ngày
            return round(year + days / 365, 2)
        except ValueError:
            return None
    return None # Trả về None nếu không phải chuỗi hoặc định dạng sai

# 1. Load và tiền xử lý dữ liệu
def load_and_preprocess(file_path):
    """
    Tải dữ liệu, lọc các cầu thủ thi đấu trên 900 phút, xử lý tuổi và giá trị.
    Có thể merge với file 'results4.csv' để lấy cột giá trị.
    """
    print(f"Đang tải dữ liệu từ: {file_path}")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {file_path}")
        return None

    print(f"Tổng số mẫu ban đầu: {len(df)}")

    # Chỉ giữ lại các cầu thủ đã thi đấu tối thiểu 900 phút
    df = df[df['Min'] > 900].copy()
    print(f"Số mẫu sau khi lọc Min > 900: {len(df)}")

    # Thay thế 'N/a' bằng giá trị thiếu của pandas (nên dùng pd.NA hoặc np.nan)
    # Việc thay bằng 0 ở đây có thể làm sai lệch dữ liệu nếu 0 không có nghĩa là giá trị thiếu
    # Nên dùng df.replace('N/a', pd.NA, inplace=True)
    df.replace('N/a', 0, inplace=True) # Cẩn thận với dòng này, có thể nên dùng pd.NA


    # Áp dụng hàm chuyển đổi tuổi
    df['Age'] = df['Age'].apply(convert_age)

    # Cố gắng merge với giá trị từ file khác nếu cần
    value_file_path = file_path = r'E:\ket_qua_bai_tap_lon\Code\results4.csv'
    try:
        df_tmp = pd.read_csv(value_file_path)
        # Thực hiện merge, sử dụng giá trị từ df_tmp nếu có
        df = pd.merge(df, df_tmp[['Player', 'Value']], on='Player', how='left', suffixes=('', '_from_tmp'))
        # Ưu tiên giá trị từ file tmp nếu có và không phải NA
        if 'Value_from_tmp' in df.columns:
             df['Value'] = df['Value_from_tmp'].combine_first(df['Value'])
             df.drop(columns=['Value_from_tmp'], inplace=True)
        print(f"Đã merge dữ liệu với {value_file_path}")
    except FileNotFoundError:
        print(f"Không tìm thấy file {value_file_path}, bỏ qua bước merge giá trị.")
    except Exception as e:
        print(f"Lỗi khi merge dữ liệu: {e}")
        # Tùy chọn: Nếu merge thất bại nặng, có thể trả về None hoặc xử lý lỗi khác

    # --- Bắt đầu sửa logic xử lý cột 'Value' ---
    # Đảm bảo cột Value có kiểu object để xử lý chuỗi và NA
    # Thay thế các giá trị rỗng, 'nan', 'N/a' bằng None
    df['Value'] = df['Value'].astype(str).replace(['nan', 'N/a', ''], None)

    # Áp dụng chuyển đổi giá trị
    df['Value'] = df['Value'].apply(convert_value)

    # Loại bỏ các hàng mà việc chuyển đổi 'Value' thất bại (kết quả là None hoặc vẫn là NA)
    df = df[df['Value'].notnull()].copy()
    # --- Kết thúc sửa logic xử lý cột 'Value' ---

    # --- Bỏ dòng dropna quá mạnh tay ---
    # df.dropna(inplace=True) # Bỏ dòng này vì nó loại bỏ quá nhiều hàng

    print(f"Số mẫu sau khi xử lý và loại bỏ giá trị thiếu ở cột 'Value': {len(df)}")

    # Kiểm tra lại kiểu dữ liệu của cột Value

    return df

# 2. Chuẩn hóa dữ liệu và chọn features
# Hàm này giờ nhận df làm tham số đầu vào
def prepare_data(df):
    """
    Chọn features, xử lý giá trị thiếu trong features và tách X, y.
    """
    if df is None or df.empty:
        print("DataFrame rỗng hoặc không hợp lệ, không thể chuẩn bị dữ liệu.")
        return None, None, None

    # Danh sách các features để sử dụng
    # Bao gồm cả các features liên quan đến thủ môn theo yêu cầu trước đó
    features = [
        "Age", "Gls", "Ast", "xG", "xAG", "per90_gls", "per90_ast", "per90_xg",
        "per90_xag", "shooting_standard_sotpct", "shooting_standard_sot_per90",
        "shooting_standard_g_sh", "shooting_standard_dist", "possession_takeons_att",
        "possession_takeons_succpct", "creation_sca_sca", "creation_sca_sca90",
        "creation_gca_gca", "creation_gca_gca90", "defense_tackles_tkl", "defense_tackles_tklw", "defense_challenges_att",
        "defense_challenges_lost", "defense_blocks_blocks", "defense_blocks_sh",
        "defense_blocks_pass", "defense_blocks_int", "possession_touches_def_pen",
        "possession_touches_def_3rd", "misc_performance_recov", "misc_aerial_won",
        "misc_aerial_wonpct",
        # Các features thủ môn - chỉ sử dụng nếu bạn chắc chắn chúng có ý nghĩa
        # hoặc nếu bạn đang dự đoán giá trị cho thủ môn (và dữ liệu của bạn bao gồm họ)
        "goalkeeping_performance_ga90", "goalkeeping_performance_savepct",
        "goalkeeping_performance_cspct", "goalkeeping_penalties_savepct"
    ]

    # Chỉ giữ lại các features có trong dataframe
    available_features = [f for f in features if f in df.columns]
    print(f"\nSố lượng features sẽ sử dụng: {len(available_features)}")
    print(f"Các features sẽ sử dụng: {available_features}")

    # Kiểm tra xem các features có tồn tại và không rỗng sau khi chọn
    if not available_features:
         print("Không có features nào hợp lệ được tìm thấy sau khi lọc.")
         return None, None, None

    # Xử lý giá trị bị thiếu trong các features đã chọn VÀ cột Value
    # Loại bỏ các hàng có giá trị thiếu trong CÁC FEATURES ĐƯỢC CHỌN + Value
    initial_row_count = len(df)
    # Đảm bảo các cột features tồn tại trước khi chọn subset
    features_to_check_na = [f for f in available_features if f in df.columns] + ['Value']
    df_processed = df.dropna(subset=features_to_check_na).copy()
    print(f"Số mẫu sau khi loại bỏ các hàng có giá trị thiếu trong features đã chọn và Value: {len(df_processed)} (Đã loại bỏ {initial_row_count - len(df_processed)} hàng)")

    if df_processed.empty:
        print("DataFrame rỗng sau khi loại bỏ giá trị thiếu trong features, không thể tiếp tục.")
        return None, None

    # Tách X và y (Sửa lỗi cú pháp ở đây)
    X = df_processed[available_features]
    y = df_processed['Value']

    # Chuyển đổi các cột feature đã chọn sang kiểu số, ép lỗi thành NaN (đảm bảo chắc chắn)
    # Dù đã dropna ở trên, bước này để đề phòng và đảm bảo kiểu dữ liệu
    # Sử dụng vòng lặp để áp dụng pd.to_numeric cho từng cột
    for col in X.columns:
         X[col] = pd.to_numeric(X[col], errors='coerce')

    print(f"Kích thước tập X sau chuẩn bị cuối cùng: {X.shape}")
    print(f"Kích thước tập y sau chuẩn bị cuối cùng: {y.shape}")


    return X, y # Trả về X và y

# Định nghĩa các mô hình để đánh giá
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(random_state=42),
    'XGBoost': XGBRegressor(random_state=42),
    # Bạn có thể thêm các mô hình khác nếu muốn (ví dụ: SVR, MLPRegressor - cần tuning thêm)
    # 'SVR': SVR(),
    # 'MLPRegressor': MLPRegressor(random_state=42, max_iter=500),
}

# Định nghĩa lưới siêu tham số cho các mô hình cần tuning
# Hãy giữ lưới nhỏ khi chạy thử để tiết kiệm thời gian
param_grids = {

 'Random Forest': {
        'n_estimators': [100, 200], # Số lượng cây
        'max_depth': [10, 20, None], # Độ sâu tối đa của cây
        'min_samples_split': [2, 5], # Số lượng mẫu tối thiểu để tách một node
        'min_samples_leaf': [1, 2] # Số lượng mẫu tối thiểu tại node lá
 },

 'XGBoost': {
        'n_estimators': [100, 200], # Số lượng boost trees
        'learning_rate': [0.05, 0.1], # Tốc độ học
        'max_depth': [3, 5], # Độ sâu tối đa của cây
        'subsample': [0.8, 1.0], # Tỷ lệ mẫu được lấy để huấn luyện mỗi cây
        'colsample_bytree': [0.8, 1.0], # Tỷ lệ features được lấy để huấn luyện mỗi cây
        'gamma': [0, 0.1] # Mức giảm mất mát tối thiểu để thực hiện phân tách
 }

}

# Hàm đánh giá các mô hình (bao gồm cả tuning)
def evaluate_models(X, y, models):
    """
    Chia dữ liệu, chuẩn hóa, thực hiện Grid Search cho các mô hình cần tuning
    và đánh giá tất cả các mô hình trên tập kiểm tra.
    """
    if X is None or y is None or X.empty:
        print("Dữ liệu X hoặc y rỗng, không thể đánh giá mô hình.")
        return pd.DataFrame(), None

    # Chia dữ liệu thành tập huấn luyện và kiểm tra
    # Việc chia này nên được thực hiện TRƯỚC khi chuẩn hóa
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"\nKích thước tập huấn luyện (X_train, y_train): {X_train.shape}, {y_train.shape}")
    print(f"Kích thước tập kiểm tra (X_test, y_test): {X_test.shape}, {y_test.shape}")

    # Chuẩn hóa dữ liệu - Fit trên tập train, Transform trên cả train và test
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    result = []
    print("\nBắt đầu đánh giá các mô hình (bao gồm tuning nếu có):")
    for name, model in models.items():
        print(f"\n--- Đang xử lý mô hình: {name} ---")
        best_model = model
        best_params = "N/A (Không tuning)" # Mặc định không tuning

        # --- Đánh giá mô hình tốt nhất (sau tuning hoặc mặc định) trên tập kiểm tra ---
        print(f"  Đang đánh giá mô hình {name} trên tập kiểm tra...")
        try:
            best_model.fit(X_train_scaled, y_train)
            y_pred = best_model.predict(X_test_scaled)

            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            result.append({
                'Model': name,
                'MSE': mse,
                'MAE': mae,
                'R2': r2,
                'Best_Params': best_params # Lưu các tham số tốt nhất hoặc mặc định/lỗi
            })
            print(f"  Đánh giá hoàn thành cho {name}.")
            print(f"  Kết quả trên tập kiểm tra: MSE={mse:.2f}, MAE={mae:.2f}, R2={r2:.4f}")
        except Exception as e:
             print(f"  Lỗi khi đánh giá {name} trên tập kiểm tra: {e}")
             result.append({
                 'Model': name, 'MSE': None, 'MAE': None, 'R2': None,
                 'Best_Params': best_params, 'Error': f'Evaluation on test set failed: {e}'
             })

    return pd.DataFrame(result), scaler
def tunning(X,y,model,param_grids):
   
    # Việc chia này nên được thực hiện TRƯỚC khi chuẩn hóa
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.22, random_state=42)

    # Chuẩn hóa dữ liệu - Fit trên tập train, Transform trên cả train và test
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\nBắt đầu tuning mô hình nếu có):")
    
    print(f"\n--- Đang xử lý mô hình: {model} ---")
    best_model = model
    best_params = "N/A (Không tuning)" # Mặc định không tuning
    # --- Đánh giá mô hình tốt nhất (sau tuning hoặc mặc định) trên tập kiểm tra ---
    if model in param_grids:
        # Thực hiện tuning nếu có lưới tham số cho mô hình này
        print(f"  Bắt đầu tuning cho {model} bằng Grid Search...")
        grid_search = GridSearchCV(
            estimator=models[model],
            param_grid=param_grids[model],
            cv=5, # Số lượng fold cross validation
            scoring='r2', # Tiêu chí đánh giá
            n_jobs=-1, # Sử dụng tất cả các CPU core
            verbose=2 # Hiển thị tiến trình tuning
        )
        grid_search.fit(X_train_scaled, y_train)
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        print(f"  Tuning hoàn thành cho {model}.")
        print(f"  Best parameters: {best_params}")
        print(f"  Best R2 score trên tập huấn luyện (cross-validation): {grid_search.best_score_:.4f}")
    else:
        # Không có lưới tuning, huấn luyện mô hình với tham số mặc định
        print(f"  Huấn luyện mô hình {model} với tham số mặc định kết quả cho được là như cũ...")
    try:
        y_pred = best_model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"  Đánh giá hoàn thành cho {model}.")
        print(f"  Kết quả trên tập kiểm tra: MSE={mse:.2f}, MAE={mae:.2f}, R2={r2:.4f}")
    except Exception as e:
            print(f"  Lỗi khi đánh giá {model} trên tập kiểm tra: {e}")
      
# 3. Main execution
if __name__ == "__main__":
    # Đường dẫn đến file dữ liệu
    data_file_path = r'E:\ket_qua_bai_tap_lon\Code\results.csv'

    # 1. Load và tiền xử lý dữ liệu
    df = load_and_preprocess(data_file_path)

    if df is not None and not df.empty:
        # In thông tin về DataFrame sau tiền xử lý
        print("\nKiểm tra các giá trị thiếu sau tiền xử lý:")
        print(df.isnull().sum()[df.isnull().sum() > 0])

        # 2. Chuẩn bị dữ liệu (tách X, y và xử lý thiếu giá trị trong features)
        # Truyền df đã load vào prepare_data
        X, y = prepare_data(df)

        if X is not None and y is not None and not X.empty:
             # 3. Đánh giá các mô hình (bao gồm tuning)
             # Truyền thêm param_grids vào hàm evaluate_models
             evaluation_results, scaler = evaluate_models(X, y, models)

             if not evaluation_results.empty:
                print("\n--- Kết quả đánh giá cuối cùng của các mô hình ---")
                # Hiển thị cột Best_Params đầy đủ hơn
                # Tạm thời tắt giới hạn hiển thị cột để in kết quả tuning
                pd.set_option('display.max_colwidth', None)
                print(evaluation_results.round(4))
                # Đặt lại giới hạn hiển thị cột về giá trị mặc định hoặc phù hợp
                pd.set_option('display.max_colwidth', 50)


                # --- Vẽ biểu đồ kết quả đánh giá ---
                print("\nĐang vẽ biểu đồ kết quả đánh giá...")

                # Tạo 3 subplot xếp ngang nhau (1 hàng, 3 cột)
                # Điều chỉnh figsize để phù hợp với bố cục ngang
                fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6)) # 1 hàng, 3 cột


                # Plot MSE (trên axes[0])
                evaluation_results.plot(x='Model', y='MSE', kind='bar', ax=axes[0], legend=False)
                axes[0].set_title('So sánh MSE giữa các Mô hình')
                axes[0].set_ylabel('MSE')
                # Đặt rotation = 0 để nhãn nằm ngang
                axes[0].tick_params(axis='x', rotation=0)
                axes[0].grid(axis='y', linestyle='--', alpha=0.7) # Thêm lưới ngang

                # Plot MAE (trên axes[1])
                evaluation_results.plot(x='Model', y='MAE', kind='bar', ax=axes[1], legend=False, color='orange')
                axes[1].set_title('So sánh MAE giữa các Mô hình')
                axes[1].set_ylabel('MAE')
                # Đặt rotation = 0 để nhãn nằm ngang
                axes[1].tick_params(axis='x', rotation=0)
                axes[1].grid(axis='y', linestyle='--', alpha=0.7)

                # Plot R2 (trên axes[2])
                evaluation_results.plot(x='Model', y='R2', kind='bar', ax=axes[2], legend=False, color='green')
                axes[2].set_title('So sánh R2 giữa các Mô hình')
                axes[2].set_ylabel('R2')
                 # Đặt rotation = 0 để nhãn nằm ngang
                axes[2].tick_params(axis='x', rotation=0)
                axes[2].grid(axis='y', linestyle='--', alpha=0.7)
                # Đặt giới hạn trục y cho R2 từ giá trị nhỏ nhất (hoặc 0) đến 1
                min_r2 = evaluation_results['R2'].min()
                axes[2].set_ylim(min(min_r2 * 0.9, 0) if pd.notnull(min_r2) and min_r2 < 0 else 0, 1.05) # Tăng giới hạn trên lên 1.05


                plt.tight_layout() # Tự động điều chỉnh khoảng cách giữa các subplot
                plt.show() # Hiển thị biểu đồ
                evaluation_results.sort_values(by='R2', ascending=False, inplace=True)
                # In model tốt nhất
                print(f'\nTừ biểu đồ và số liệu đánh giá, ta có thể thấy model tốt nhất là: {evaluation_results.iloc[0]["Model"]}')
                print(f'\n--------------------------------')
                name_model=evaluation_results.iloc[0]["Model"]
                tunning(X,y,name_model,param_grids)
             else:
                print("\nKhông thể đánh giá mô hình.")

        else:
            print("\nKhông thể chuẩn bị dữ liệu (X hoặc y rỗng sau khi chọn features và xử lý thiếu).")
    else:
        print("\nKhông thể tải hoặc tiền xử lý dữ liệu thành công. Vui lòng kiểm tra file đầu vào và logic xử lý.")
                
    print("\n--- Quá trình thực thi hoàn thành ---")