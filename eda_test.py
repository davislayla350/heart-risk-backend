# imports for data handling and visualization
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# load dataset into a dataframe
df = pd.read_csv("data/heart.csv")

# show dataset structure info
print("Dataset Info:")
print(df.info())

# show basic statistics for numeric columns
print("\nBasic Statistics:")
print(df.describe())

# check for any missing values in each column
print("\nAny missing values?")
print(df.isnull().sum())

# select only numeric columns for correlation analysis
numeric_df = df.select_dtypes(include=['int64', 'float64'])

# plot heatmap of feature correlations
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Between Features")
plt.tight_layout()
plt.show()
