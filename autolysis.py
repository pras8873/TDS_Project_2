import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
import requests
import json

# Function to read the CSV file
def load_csv(filename):
    try:
        data = pd.read_csv(filename)
        print(f"Loaded {filename} successfully with utf-8 encoding.")
        return data
    except UnicodeDecodeError:
        print(f"Failed to load {filename} with utf-8 encoding. Detecting file encoding...")
        try:
            from charset_normalizer import detect
            with open(filename, 'rb') as f:
                raw_data = f.read()
                detected = detect(raw_data)
                encoding = detected['encoding']
                print(f"Detected encoding: {encoding}")
                data = pd.read_csv(filename, encoding=encoding)
                print(f"Loaded {filename} successfully with {encoding} encoding.")
                return data
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            sys.exit(1)

# Function to perform basic data inspection
def inspect_data(data):
    report = {
        'columns': data.columns.tolist(),
        'dtypes': data.dtypes.astype(str).to_dict(),
        'missing_values': data.isnull().sum().to_dict(),
        'summary_statistics': data.describe(include='all').to_dict()
    }
    return report
# Function to detect outliers
outliar_columns = []
def detect_outliers(data):
    numerical_data = data.select_dtypes(include=['number'])
    outlier_report = {}
    for col in numerical_data.columns:
        q1 = numerical_data[col].quantile(0.25)
        q3 = numerical_data[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = numerical_data[(numerical_data[col] < lower_bound) | (numerical_data[col] > upper_bound)]
        if not outliers.empty:
            outliar_columns.append(col)
        outlier_report[col] = len(outliers)
    print(f"Columns with outliers: {outliar_columns}")
    return outlier_report


# Function to generate visualizations
def create_visualizations(data, output_prefix):
    numerical_data = data.select_dtypes(include=['number'])

    if not numerical_data.empty:
        # Correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(numerical_data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
        plt.title("Correlation Heatmap")
        plt.savefig(f"{output_prefix}/correlation_heatmap.png")
        plt.close()
        print(f"Saved {output_prefix}/correlation_heatmap.png")

        # Boxplot for columns with outliers only
        for col in outliar_columns:
            plt.figure(figsize=(8, 6))
            sns.boxplot(x=numerical_data[col])
            plt.title(f"Boxplot of {col} (Outliers)")
            plt.savefig(f"{output_prefix}/boxplot_{col}.png")
            plt.close()
            print(f"Saved {output_prefix}/boxplot_{col}.png")

        # Distribution plots with log scaling for high values
        for col in numerical_data.columns:
            unique_values = numerical_data[col].nunique()
            variance = numerical_data[col].var()
            
            if unique_values > 10 and variance > 0.1:  # Skip columns with low cardinality or low variance
                plt.figure(figsize=(8, 6))
                
                # Apply log scaling if max value is significantly high
                if numerical_data[col].max() > 1e6:
                    scaled_data = np.log1p(numerical_data[col])  # log1p avoids log(0) issues
                    sns.histplot(scaled_data, kde=True, bins=30)
                    plt.title(f"Log-scaled Distribution of {col}")
                else:
                    sns.histplot(numerical_data[col], kde=True, bins=30)
                    plt.title(f"Distribution of {col}")
                
                plt.savefig(f"{output_prefix}/distribution_{col}.png")
                plt.close()
                print(f"Saved {output_prefix}/distribution_{col}.png")
            else:
                print(f"Skipping distribution plot for {col}: low cardinality ({unique_values}) or low variance ({variance:.3f}).")

    # Missing value heatmap
    if data.isnull().any().any():
        plt.figure(figsize=(10, 6))
        sns.heatmap(data.isnull(), cbar=False, cmap="viridis")
        plt.title("Missing Values Heatmap")
        plt.savefig(f"{output_prefix}/missing_values.png")
        plt.close()
        print(f"Saved {output_prefix}/missing_values.png")

    # Dendrogram for hierarchical clustering
    if np.isfinite(numerical_data).all().all():
        linkage_matrix = linkage(numerical_data, method='ward')
        plt.figure(figsize=(12, 8))
        dendrogram(linkage_matrix, labels=data.index.tolist())
        plt.title("Hierarchical Clustering Dendrogram")
        plt.savefig(f"{output_prefix}/dendrogram.png")
        plt.close()
        print(f"Saved {output_prefix}/dendrogram.png")
    else:
        print("Numerical data contains non-finite values. Skipping dendrogram creation.")

# Perform clustering and PCA analysis
def cluster_and_pca_analysis(data, output_prefix):
    numerical_data = data.select_dtypes(include=['number']).dropna()
    if numerical_data.shape[0] > 1:
        # Clustering
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(numerical_data)
        
        # Ensure the original data aligns with the clusters
        data = data.loc[numerical_data.index]  # Filter data to match numerical_data's index
        data['Cluster'] = clusters
        
        sns.scatterplot(x=numerical_data.iloc[:, 0], y=numerical_data.iloc[:, 1], hue=clusters, palette='viridis')
        plt.title("K-Means Clustering")
        plt.savefig(f"{output_prefix}/kmeans_clustering.png")
        plt.close()
        print(f"Saved {output_prefix}/kmeans_clustering.png")

        # PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(numerical_data)
        data['PCA1'] = pca_result[:, 0]
        data['PCA2'] = pca_result[:, 1]
        sns.scatterplot(x=data['PCA1'], y=data['PCA2'], hue=clusters, palette='viridis')
        plt.title("PCA Analysis")
        plt.savefig(f"{output_prefix}/pca_analysis.png")
        plt.close()
        print(f"Saved {output_prefix}/pca_analysis.png")


# Generate narrative using LLM
def generate_narrative(data_report, output_prefix, outlier_report):
    token = os.environ.get("AIPROXY_TOKEN")
    if not token:
        print("AIPROXY_TOKEN environment variable is not set. Exiting.")
        sys.exit(1)

    base_url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    prompt = f"""
    Analyze the following dataset summary and generate a story:
    Columns and Types: {data_report['dtypes']}
    Missing Values: {data_report['missing_values']}
    Summary Statistics: {data_report['summary_statistics']}
    Outlier Report: {outlier_report}
    Provide a story describing the dataset, key insights, and actionable conclusions.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a data analyst."},
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = requests.post(base_url, headers=headers, data=json.dumps(payload))
        if response.status_code != 200:
            print(f"Error generating narrative: {response.status_code} - {response.json()}")
            return

        narrative = response.json()["choices"][0]["message"]["content"]
        with open("README.md", "w") as f:
            f.write(narrative)
        print(f"Saved narrative as README.md")
    except Exception as e:
        print(f"Error generating narrative: {e}")

# Main function
def main():
    if len(sys.argv) != 2:
        print("Usage: python autolysis.py <dataset.csv>")
        sys.exit(1)

    filename = sys.argv[1]
    output_prefix = os.path.splitext(filename)[0]
    data = load_csv(filename)
    report = inspect_data(data)
    outlier_report = detect_outliers(data)
    create_visualizations(data, output_prefix)
    cluster_and_pca_analysis(data, output_prefix)
    generate_narrative(report, output_prefix, outlier_report)

if __name__ == "__main__":
    main()
