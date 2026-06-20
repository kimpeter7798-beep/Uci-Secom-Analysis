
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Correct file paths provided by the user
features_path = "C:\\Users\\kimpe\\OneDrive\\바탕 화면\\3-2\\융합시스템최적설계(김성수 교수님)\\se-com\\반도체 웨이퍼 1567 데이터 20개 특성.csv"
labels_path = "C:\\Users\\kimpe\\OneDrive\\바탕 화면\\3-2\\융합시스템최적설계(김성수 교수님)\\se-com\\반도체 웨이퍼 1567 데이터 20개 특성 labels.csv"

try:
    features_df = pd.read_csv(features_path)
    labels_df = pd.read_csv(labels_path)
except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    exit()

# Merge data
df = pd.concat([features_df, labels_df], axis=1)

# Drop constant columns
df.drop(columns=df.columns[df.std() == 0], inplace=True)

# Drop columns with > 5% missing values
missing_over_5_percent = df.columns[df.isnull().mean() > 0.05]
df.drop(columns=missing_over_5_percent, inplace=True)

# Impute remaining missing values with the mean
for col in df.columns[df.isnull().any()]:
    df[col].fillna(df[col].mean(), inplace=True)

# Scale features
scaler = MinMaxScaler()
# Ensure 'Pass_Fail' column exists before trying to drop it
if 'Pass_Fail' in df.columns:
    features_to_scale = df.columns.drop('Pass_Fail')
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
else:
    print("Warning: 'Pass_Fail' column not found. Scaling all columns.")
    df.iloc[:, :] = scaler.fit_transform(df.iloc[:, :])


# Save preprocessed data
df.to_csv('final_data.csv', index=False)

print("Data preprocessing complete and saved to final_data.csv")
