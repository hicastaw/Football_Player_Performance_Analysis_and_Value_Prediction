import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

import seaborn as sns



def top_20_features(X, y):
    def convert_abs(x):
        return abs(x) if isinstance(x, (int, float)) else 0
    # Split data before scaling
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model= RandomForestRegressor(random_state=42)
    try:
        model.fit(X_train_scaled, y_train)
    
        importances = model.feature_importances_
        feature_names = X.columns  # Ensure that X is a DataFrame, if it's not, convert it first.
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
        importance_df['Importance']= importance_df['Importance'].apply(convert_abs)
        importance_df.sort_values(by='Importance', ascending=False, inplace=True)

        # Print top 20 important features
        print("\nTop 20 feature quan trọng nhất theo Random Forest:")
        print(importance_df.head(20))

        # Plot bar chart for visualization
        plt.figure(figsize=(12, 8))
        sns.barplot(data=importance_df.head(20), x='Importance', y='Feature', palette='viridis')
        plt.title('Top 20 Feature Quan Trọng Nhất (Random Forest)')
        plt.xlabel('Mức độ quan trọng')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.savefig('top_20_features_rf.png', dpi=300)
        plt.show()
        # Return the top 20 features as a list
        return importance_df['Feature'].head(20).tolist()

    except Exception as e:
        print(f"  Error evaluating {model} on test set: {e}")
        return None