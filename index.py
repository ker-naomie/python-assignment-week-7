import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Task 1: Load and Explore the Dataset
try:
    # Load dataset (using the Iris dataset as an example)
    from sklearn.datasets import load_iris
    iris = load_iris()
    iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    iris_df['species'] = iris.target
    print("Dataset loaded successfully.")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# Display the first few rows
print("\nDataset Preview:")
print(iris_df.head())

# Check data types and missing values
print("\nDataset Information:")
print(iris_df.info())

# Check for missing values
print("\nMissing Values:")
print(iris_df.isnull().sum())

# No missing values in this dataset. If present, we could handle them here.

# Task 2: Basic Data Analysis
# Compute basic statistics 
print("\nDescriptive Statistics:")
print(iris_df.describe())

# Group by species and compute the mean of each numerical column
species_mean = iris_df.groupby('species').mean()
print("\nMean values per species:")
print(species_mean)

# Task 3: Data Visualization
sns.set(style="whitegrid")

# Line chart showing trends over features for a sample (using the first instance of each species as an example)
plt.figure(figsize=(10, 6))
for species_id in iris_df['species'].unique():
    sample = iris_df[iris_df['species'] == species_id].iloc[0, :-1]
    plt.plot(iris.feature_names, sample, label=f"Species {species_id}")
plt.title("Trends Over Features (Sample Data)")
plt.xlabel("Features")
plt.ylabel("Values")
plt.legend(title="Species")
plt.tight_layout()
plt.savefig("line_chart_trends.png")
plt.show()

# Bar chart showing average feature values per species
plt.figure(figsize=(10, 6))
species_mean.plot(kind='bar', colormap='viridis', ax=plt.gca())
plt.title("Average Feature Values per Species")
plt.xlabel("Species")
plt.ylabel("Mean Value")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("bar_chart_average.png")
plt.show()

# Histogram of a numerical column (sepal length)
plt.figure(figsize=(8, 6))
sns.histplot(iris_df[iris.feature_names[0]], kde=True, bins=15, color='blue')
plt.title("Distribution of Sepal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("histogram_sepal_length.png")
plt.show()

# Scatter plot to visualize relationship between sepal length and sepal width
plt.figure(figsize=(8, 6))
sns.scatterplot(x=iris.feature_names[0], y=iris.feature_names[1], hue='species', data=iris_df, palette='coolwarm', s=100)
plt.title("Sepal Length vs Sepal Width")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.legend(title="Species")
plt.tight_layout()
plt.savefig("scatter_sepal_length_width.png")
plt.show()

# Findings and Observations
print("\nFindings and Observations:")
print("1. The average feature values vary significantly across species.")
print("2. Sepal length and sepal width show an interesting relationship across species.")
print("3. The distribution of sepal length appears to be bimodal.")
