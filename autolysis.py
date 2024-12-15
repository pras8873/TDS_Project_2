import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from openai import OpenAI

# Function to read the CSV file
import pandas as pd
import sys

def load_csv(filename):
    try:
        # Attempt to read with the default 'utf-8' encoding
        data = pd.read_csv(filename)
        print(f"Loaded {filename} successfully with utf-8 encoding.")
        return data
    except UnicodeDecodeError:
        print(f"Failed to load {filename} with utf-8 encoding. Detecting file encoding...")

        # Detect encoding using charset-normalizer
        try:
            from charset_normalizer import detect
            with open(filename, 'rb') as f:
                raw_data = f.read()
                detected = detect(raw_data)
                encoding = detected['encoding']
                print(f"Detected encoding: {encoding}")

                # Load the file with the detected encoding
                data = pd.read_csv(filename, encoding=encoding)
                print(f"Loaded {filename} successfully with {encoding} encoding.")
                return data
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            sys.exit(1)


# Function to perform basic data inspection
def inspect_data(data):
    report = {}
    report['columns'] = data.columns.tolist()
    report['dtypes'] = data.dtypes.astype(str).to_dict()
    report['missing_values'] = data.isnull().sum().to_dict()
    report['summary_statistics'] = data.describe(include='all').to_dict()
    return report

# Function to generate visualizations
def create_visualizations(data, output_prefix):
    # Visualization 1: Correlation heatmap for numerical data
    numerical_data = data.select_dtypes(include=['number'])
    if not numerical_data.empty:
        plt.figure(figsize=(10, 8))
        sns.heatmap(numerical_data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
        plt.title("Correlation Heatmap")
        plt.savefig(f"{output_prefix}/correlation_heatmap.png")
        plt.close()
        print(f"Saved {output_prefix}/correlation_heatmap.png")

    # Visualization 2: Missing value heatmap
    if data.isnull().any().any():
        plt.figure(figsize=(10, 6))
        sns.heatmap(data.isnull(), cbar=False, cmap="viridis")
        plt.title("Missing Values Heatmap")
        plt.savefig(f"{output_prefix}/missing_values.png")
        plt.close()
        print(f"Saved {output_prefix}/missing_values.png")

    # Visualization 3: Example boxplot for the first numerical column (if any)
    if not numerical_data.empty:
        first_col = numerical_data.columns[0]
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=data[first_col])
        plt.title(f"Boxplot of {first_col}")
        plt.savefig(f"{output_prefix}/boxplot.png")
        plt.close()
        print(f"Saved {output_prefix}/boxplot.png")

# Function to generate narrative using LLM
import requests
import os
import json

def generate_narrative(data_report, output_prefix):
    # Fetch the token from the environment variable
    token = os.environ.get("AIPROXY_TOKEN")
    if not token:
        print("AIPROXY_TOKEN environment variable is not set. Exiting.")
        sys.exit(1)
    
    # AI Proxy URL
    base_url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

    # Prepare the LLM prompt
    prompt = f"""
    Analyze the following dataset summary and generate a story:
    Columns and Types: {data_report['dtypes']}
    Missing Values: {data_report['missing_values']}
    Summary Statistics: {data_report['summary_statistics']}
    Provide a story describing the dataset, key insights, and actionable conclusions.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a data analysis assistant."},
            {"role": "user", "content": prompt}
        ]
    }
    
    try:
        # Make the POST request to AI Proxy
        response = requests.post(base_url, headers=headers, data=json.dumps(payload))

        # Check if the response is successful
        if response.status_code != 200:
            print(f"Error generating narrative: {response.status_code} - {response.json()}")
            return

        # Extract the narrative from the response
        narrative = response.json()["choices"][0]["message"]["content"]

        # Save the narrative to a README.md file
        with open("README.md", "w") as f:
            f.write(narrative)
        print(f"Saved narrative as README.md")

        # Display additional cost headers
        cost = response.headers.get("cost")
        monthly_cost = response.headers.get("monthlyCost")
        print(f"Request cost: ${cost}, Monthly cost so far: ${monthly_cost}")

    except Exception as e:
        print(f"Error generating narrative: {e}")

# Main function
def main():
    if len(sys.argv) != 2:
        print("Usage: uv run autolysis.py <dataset.csv>")
        sys.exit(1)
    
    filename = sys.argv[1]
    output_prefix = os.path.splitext(filename)[0]
    data = load_csv(filename)
    report = inspect_data(data)
    create_visualizations(data, output_prefix)
    generate_narrative(report, output_prefix)

if __name__ == "__main__":
    main()
