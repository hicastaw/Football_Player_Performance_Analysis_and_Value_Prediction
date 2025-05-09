import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import seaborn as sns
from rac import top_20_features
# Function to convert currency values
def convert_value(value_str):
    if isinstance(value_str, str):
        value_str = value_str.replace('€', '').replace(',', '').strip()
        if 'M' in value_str.upper(): # Use .upper() to handle both 'm' and 'M'
            try:
                return float(value_str.upper().replace('M', '')) * 1e6 # 1 million is 1e6
            except ValueError:
                return None
    return None # Return None if not a string or format mismatch

# Function to convert age (format year-day)
def convert_age(age_str):
    if isinstance(age_str, str) and '-' in age_str:
        try:
            year, days = map(int, age_str.split('-'))
            # Assume one year has 365 days
            return round(year + days / 365, 2)
        except ValueError:
            return None
    return None # Return None if not a string or wrong format

# 1. Load and preprocess data
def load_and_preprocess(file_path):
    print(f"Loading data from: {file_path}")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File not found {file_path}")
        return None

    print(f"Initial sample size: {len(df)}")

    # Keep only players who played at least 900 minutes
    df = df[df['Min'] > 900].copy()
    print(f"Sample size after filtering Min > 900: {len(df)}")

    df.replace('N/a', 0, inplace=True) 
    # Apply age conversion function
    df['Age'] = df['Age'].apply(convert_age)

    # Try to merge with value file if needed
    current_dir = Path(__file__).parent
    file_path = current_dir / 'transfer_value.csv'
    value_file_path = file_path
    try:
        df_tmp = pd.read_csv(value_file_path)
        # Perform merge, use value from df_tmp if available
        df = pd.merge(df, df_tmp[['Player', 'Value']], on='Player', how='left')
        print(f"Merged data with {value_file_path}")
    except Exception as e:
        print(f"Error during merging data: {e}")

    # Apply value conversion
    df['Value'] = df['Value'].apply(convert_value)

    print(f"Sample size after processing: {len(df)}")
    return df

# 2. Standardize data and select features
# This function now takes df as input parameter
def prepare_data(df):
    """
    Select features, handle missing values in features and split X, y.
    """
    if df is None or df.empty:
        print("Empty or invalid DataFrame, cannot prepare data.")
        return None, None, None

    # List of features to use
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
        "goalkeeping_performance_ga90", "goalkeeping_performance_savepct",
        "goalkeeping_performance_cspct", "goalkeeping_penalties_savepct"
    ]

    # Keep only available features in dataframe
    available_features = [f for f in features if f in df.columns]
    print(f"\nNumber of features to use: {len(available_features)}")
    print(f"Features to use: {available_features}")

    # Check if features exist and are not empty after selection
    if not available_features:
        print("No valid features found after filtering.")
        return None, None, None

    # Split X and y
    X_tmp = df[available_features]
    y_tmp = df['Value']
    header_results=top_20_features(X_tmp,y_tmp)
    X=X_tmp[header_results]
    y=y_tmp 

    # Apply pd.to_numeric to each column using a loop
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    print(f"X shape after final preparation: {X.shape}")
    print(f"y shape after final preparation: {y.shape}")

    return X, y # Return X and y

# Define models to evaluate
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(random_state=42),
    'XGBoost': XGBRegressor(random_state=42)
}

# Define parameter grid for models to tune
# Keep grid small for trial runs to save time
param_grids = {
    'Random Forest': {
        'n_estimators': [100],                # giữ số cây vừa phải
        'max_depth': [5, 10],                 # giới hạn độ sâu cây (tránh cây quá phức tạp)
        'min_samples_split': [5, 10],         # tăng số lượng mẫu tối thiểu để chia node
        'min_samples_leaf': [2, 4, 6],        # tăng số mẫu ở lá để giảm số node
        'max_features': ['sqrt', 0.5]         # chọn ngẫu nhiên số lượng feature tại mỗi node
    },

    'XGBoost': {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'gamma': [0, 0.1]
    }
}

# Function to evaluate models
def evaluate_models(X, y, models):
    """
    Evaluate models on test set.
    """
    if X is None or y is None or X.empty:
        print("X or y is empty, cannot evaluate models.")
        return pd.DataFrame(), None

    # Split data before scaling
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"\nTrain size (X_train, y_train): {X_train.shape}, {y_train.shape}")
    print(f"Test size (X_test, y_test): {X_test.shape}, {y_test.shape}")

    # Standardize data - Fit on train, transform on both train and test
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    result = []
    print("\nStarting model evaluation (including tuning if available):")
    for name, model in models.items():
        print(f"\n--- Processing model: {name} ---")
        best_model = model
        params = "N/A (No tuning)" # Default no tuning

        # Evaluate the best model on test set
        print(f"  Evaluating {name} on test set...")
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
                'Params': params
            })
            print(f"  Evaluation completed for {name}.")
            print(f"  Test results: MSE={mse:.2f}, MAE={mae:.2f}, R2={r2:.4f}")
        except Exception as e:
            print(f"  Error evaluating {name} on test set: {e}")
            result.append({
                'Model': name, 'MSE': None, 'MAE': None, 'R2': None,
                'Best_Params': params, 'Error': f'Evaluation on test set failed: {e}'
            })

    return pd.DataFrame(result), scaler

def tunning(X, y, model, param_grids):
    # Split data before scaling
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\nStarting model tuning (if available):")
    print(f"\n--- Processing model: {model} ---")
    best_model = model
    best_params = "N/A (No tuning)"  # Default no tuning

    if model in param_grids:
        print(f"  Starting tuning for {model} using Grid Search...")
        grid_search = GridSearchCV(
            estimator=models[model],
            param_grid=param_grids[model],
            cv=5,
            scoring='r2',
            n_jobs=-1,
            verbose=2
        )
        grid_search.fit(X_train_scaled, y_train)
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        print(f"  Tuning completed for {model}.")
        print(f"  Best parameters: {best_params}")
        print(f"  Best R2 score on training set (cross-validation): {grid_search.best_score_:.4f}")
    else:
        print(f"  Training {model} with default parameters, result remains unchanged...")
        best_model.fit(X_train_scaled, y_train)

    try:
        # Predictions
        y_train_pred = best_model.predict(X_train_scaled)
        y_test_pred = best_model.predict(X_test_scaled)

        # Evaluate train
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)

        # Evaluate test
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        print(f"  Evaluation completed for {model}.")
        print(f"  Train results: MSE={train_mse:.2f}, MAE={train_mae:.2f}, R2={train_r2:.4f}")
        print(f"  Test results : MSE={test_mse:.2f}, MAE={test_mae:.2f}, R2={test_r2:.4f}")

    except Exception as e:
        print(f"  Error evaluating {model} on test set: {e}")

# 3. Main execution
if __name__ == "__main__":
    # Path to data file
    current_dir = Path(__file__).parent
    file_path = current_dir / 'results.csv'
    data_file_path = file_path

    # 1. Load and preprocess data
    df = load_and_preprocess(data_file_path)

    if df is not None and not df.empty:
        print("\nChecking missing values after preprocessing:")
        print(df.isnull().sum()[df.isnull().sum() > 0])

        # 2. Prepare data (split X, y and handle missing values in features)
        X, y = prepare_data(df)

        if X is not None and y is not None and not X.empty:
            # 3. Evaluate models (including tuning)
            evaluation_results, scaler = evaluate_models(X, y, models)

            if not evaluation_results.empty:
                print("\n--- Final evaluation results of models ---")
                evaluation_results.sort_values(by='R2', ascending=False, inplace=True)
                print(evaluation_results.round(4))

                print("\nDrawing evaluation result plots...")

                # Create 3 subplots horizontally
                fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

                # Plot MSE(on axes[0])
                evaluation_results.plot(x='Model', y='MSE', kind='bar', ax=axes[0], legend=False)
                axes[0].set_title('Comparison of MSE between Models')
                axes[0].set_ylabel('MSE')
                # Set rotation = 0 so labels are horizontal
                axes[0].tick_params(axis='x', rotation=0)
                axes[0].grid(axis='y', linestyle='--', alpha=0.7) # Add horizontal grid

                # Plot MAE (on axes[1])
                evaluation_results.plot(x='Model', y='MAE', kind='bar', ax=axes[1], legend=False, color='orange')
                axes[1].set_title('Comparison of MAE between Models')
                axes[1].set_ylabel('MAE')
                # Set rotation = 0 so labels are horizontal
                axes[1].tick_params(axis='x', rotation=0)
                axes[1].grid(axis='y', linestyle='--', alpha=0.7)

                # Plot R2 (on axes[2])
                evaluation_results.plot(x='Model', y='R2', kind='bar', ax=axes[2], legend=False, color='green')
                axes[2].set_title('Comparison of R2 between Models')
                axes[2].set_ylabel('R2')
                # Set rotation = 0 so labels are horizontal
                axes[2].tick_params(axis='x', rotation=0)
                axes[2].grid(axis='y', linestyle='--', alpha=0.7)
                # Set y-axis limits for R2 from the minimum value (or 0) to 1
                min_r2 = evaluation_results['R2'].min()
                axes[2].set_ylim(min(min_r2 * 0.9, 0) if pd.notnull(min_r2) and min_r2 < 0 else 0, 1.05) # Increase upper limit to 1.05

                plt.tight_layout() # Automatically adjust spacing between subplots
                plt.show() # Display the plot

                # Print the best model
                print(f'\nFrom the chart and evaluation metrics, we can see that the best model is: {evaluation_results.iloc[0]["Model"]}')
                print(f'\n--------------------------------')
                name_model = evaluation_results.iloc[0]["Model"]
                tunning(X, y, name_model, param_grids)
            else:
                print("\nUnable to evaluate models.")
        else:
             print("\nUnable to prepare data (X or y is empty after selecting features and handling missing values).")
    else:
        print("\nUnable to successfully load or preprocess data. Please check the input file and processing logic.")
                
    print("\n--- Execution completed ---")