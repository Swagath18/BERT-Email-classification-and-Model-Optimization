import pandas as pd
from sklearn.model_selection import train_test_split

# Load full dataset
df = pd.read_csv("./data/train.csv")

# Split 80% train, 20% test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Save
train_df.to_csv("./data/train.csv", index=False)
test_df.to_csv("./data/test.csv", index=False)
