import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# read csv file into dataframe
df = pd.read_csv('all_matches.csv')

# filter out matches where the start date is before 2010 or after 2016
df = df[(df['start_date'].str[:4].astype(int) >= 2010) & (df['start_date'].str[:4].astype(int) <= 2018)]


df = df.dropna(subset=['player_victory'])

df['player_victory'] = df['player_victory'].apply(lambda x: 1 if x == 't' else 0)


# dropping columns that we think aren't useful or mostly null
df.drop(["currency", "start_date", "end_date", "prize_money", "retirement", "location", "nation", "player_name", "opponent_name", "seed", "duration"], axis=1, inplace=True)
# Remove rows with more than a threshold number of missing values
threshold = df.shape[1] // 2 # adjust this value based on your needs
df.dropna(thresh=df.shape[1] - threshold, inplace=True)

# Impute missing values
for column in df.columns:
    if df[column].dtype == 'object':  # Categorical column
        mode_val = df[column].mode()[0]
        df[column].fillna(mode_val, inplace=True)
    else:  # Numeric column
        # Use mean or median based on your preference
        mean_val = df[column].mean()
        # median_val = df[column].median()
        df[column].fillna(mean_val, inplace=True)
        # df[column].fillna(median_val, inplace=True


# 5. Feature Scaling
# numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
# scaler = StandardScaler()
# df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# 6. Encoding Categorical Variables
# categorical_columns = df.select_dtypes(include=['object']).columns
# df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# save the filtered dataframe to a new csv file
df.to_csv('matches_between_2010_2018.csv', index=False)

# print(df)