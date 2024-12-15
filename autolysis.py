# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "chardet",
#     "matplotlib",
#     "numpy",
#     "openai",
#     "pandas",
#     "python-dotenv",
#     "seaborn",
#     "requests",
#     "scikit-learn",
#     "statsmodels"
# ]
# ///
import os
import pandas as pd
import numpy as np
import sys as sys
import seaborn as sns
import matplotlib.pyplot as plt
import openai
import requests
import chardet
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from dotenv import load_dotenv
import scipy.stats as stats
import statsmodels.formula.api as sm
import random
import json
import base64
import re


load_dotenv()

"""# Read csv file and put the data into pandas DataFrame"""

# Read csv file and put the data into pandas DataFrame
def read_csv(file_path):
  """
  Function:
    Reads a CSV file.
  Args:
    file_path: Path to the CSV file.
  Returns:
    A dataframe having data from csv file.
  """
  print(file_path)
  try:
    # Get the encoding of the file
    with open(file_path, 'rb') as f:
      result = chardet.detect(f.read())
      encoding = result['encoding']
      print(encoding)
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path, encoding=encoding, encoding_errors='ignore')
    print(df.head())
    return df
  except FileNotFoundError:
    print(f"Error: File not found at path: {file_path}")
    return None
  except pd.errors.ParserError:
    print(f"Error: Could not parse the CSV file. Check its format.")
    return None
  except Exception as e:
    print(f"An unexpected error occurred: {e}")
    return None

"""# Check the number of rows, columns and missing values"""

# Check the number of rows, columns and missing values
def analyze_df(df):
  """
   Gets dataframe, analyzes its rows, columns, and missing values.
  Args:
    df: dataframe to analyze.
  Returns:
    A dictionary containing the analysis results.
  """

  try:
    # get the shape of dataframe
    num_rows, num_cols = df.shape

    # Calculate the number of missing values in each column
    missing_values = df.isnull().sum()

    # Calculate the percentage of missing values in each column
    missing_percentages = (missing_values / num_rows) * 100

    #calculate unique values in each categorical column and add in return statement
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    unique_values_per_column = {col: df[col].nunique() for col in categorical_columns}

    # Store the analysis results in a dictionary
    results = {
        "rows": num_rows,
        "columns": num_cols,
        "percentage_missing_values": missing_percentages.to_dict(),  # Convert to dictionary
        "unique_values_per_column": unique_values_per_column  # Convert to dictionary
        }
    print(results)
    return results
  except Exception as e:
    print(f"An unexpected error occurred: {e}")
    return None

"""# Get the descriptive statistics for the numerical features in the dataframe"""

# Get the descriptive statistics for the numerical features in the dataframe
def get_descriptive_stats(df):
  """
  Function:
    To get the descriptive statistics for given dataframe
  Args:
    df: dataframe to be analyzed.
  Returns:
    dataframe with descriptive statistics
  """
  desc_stats = df.describe()
  return desc_stats

"""# Do univariate Analysis"""

# Do univariate Analysis
def univariate_analysis(df):
    """
    Performs univariate analysis for a given variable.

    Args:
        df: The input DataFrame.
        variable: The name of the variable to analyze.
        plot_type: The type of plot to create ("hist", "box", "kde", "qq").
    """
    numeric_features = df.select_dtypes(include=np.number).columns.tolist()
    data_reset = df.reset_index()
    fig, axes = plt.subplots(1, len(numeric_features), figsize=(15, 6))
    subsample_size = 5000
    for variable in numeric_features:
      # Descriptive statistics
      print(f"Descriptive Statistics for {variable}:")
      print(df[variable].describe())
      # Check for normality (Shapiro-Wilk test with subsampling)
      if len(df[variable]) > subsample_size:
        subsample = random.sample(df[variable].tolist(), subsample_size)  # Take random subsample
        _, p_value = stats.shapiro(subsample)
        print(f"Shapiro-Wilk Test for Normality (subsampled, p-value): {p_value:.3f}")
      else:
        _, p_value = stats.shapiro(df[variable])
        print(f"Shapiro-Wilk Test for Normality (p-value): {p_value:.3f}")
    for i, column in enumerate(numeric_features):
      melted_data = pd.melt(data_reset, id_vars=['index'], value_vars=[column], var_name='Column', value_name='Value')
      sns.histplot(x ='Value', data=melted_data, ax=axes[i])
      axes[i].set_title(f'histogram for {column}')
      axes[i].set_ylabel('Value')

    plt.tight_layout()

"""# Get the numeric and categorical columns for analysing the data"""

#  Identifies numeric and categorical columns in a Pandas DataFrame
def get_column_types(df):
  """
  Identifies numeric and categorical columns in a Pandas DataFrame.
  Args:
    df: The input Pandas DataFrame.
  Returns:
    A tuple containing two lists: numeric column names and categorical column names.
  """
  numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
  categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
  return numeric_cols, categorical_cols

"""# Generate correlation heatmap"""

# Generate correlation heatmap
def generate_correlation_heatmap(df, title="Correlation Heatmap"):
  """
  Generates a correlation heatmap for numerical variables in a Pandas DataFrame.
  Args:
    df: The input Pandas DataFrame.
    title: The title of the heatmap (default: "Correlation Heatmap").
  Returns:
    A correlation matrix.
  """

  # Select only numerical features for correlation analysis
  numerical_df = df.select_dtypes(include=['number'])

  # Calculate the correlation matrix
  correlation_matrix = numerical_df.corr()
  # Create the heatmap using seaborn
  plt.figure(figsize=(10, 8))  # Adjust figure size as needed
  sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
  plt.title(title)
  plt.savefig('heatmap.png', dpi=64)
  plt.show()
  return correlation_matrix

"""# Get the LLM response for given prompt"""

# Get the LLM response for given prompt
def get_LLMResponse(prompt):
  """
  Function:
    Generates a response from LLM for the given prompt.
  Args:
    prompt: the prompt to be sent to LLM
  Returns:
    response from LLM
  """
  AIPROXY_TOKEN = os.getenv('AIPROXY_TOKEN')
  #AIPROXY_TOKEN = userdata.get('AIPROXY_TOKEN')
  API_URL = 'https://aiproxy.sanand.workers.dev/openai/v1/chat/completions'

  headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AIPROXY_TOKEN}",
    }
  data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
    }

  response = requests.post(API_URL, headers=headers, json=data)
  print("LLM Response code: ", response.status_code)
  if response.status_code == 200:
    return response
  else:
    exit(1)

# Get the LLM response for given Image with prompt
def get_LLMResponseForImage(Image, prompt):
  """
  Function:
    Generates a response from LLM for the given Image with prompt.
  Args:
    Image: Image to be analyzed by LLM
    prompt: the prompt to be sent to LLM
  Returns:
    response from LLM
  """
  AIPROXY_TOKEN = os.getenv('AIPROXY_TOKEN')
  #AIPROXY_TOKEN = userdata.get('AIPROXY_TOKEN')
  API_URL = 'https://aiproxy.sanand.workers.dev/openai/v1/chat/completions'

  # Read the image file as binary data
  with open(Image, 'rb') as image_file:
      image_data = image_file.read()

  # Encode the image data to base64
  base64_image = base64.b64encode(image_data).decode('utf-8')

  headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AIPROXY_TOKEN}",
    }

  data = {
      "model": "gpt-4o-mini",
      "messages": [
          {
              "role": "user",
              "content": [
                  {
                      "type": "text",
                      "text": prompt
                  },
                  {
                      "type": "image_url",
                      "image_url": {
                          "detail": "low",
                          # Instead of passing the image URL, we create a base64 encoded data URL
                          "url": f"data:image/jpeg;base64,{base64_image}"
                      }
                  }
              ]
          }
      ]
  }
  response = requests.post(API_URL, headers=headers, json=data)
  print("LLM Response code: ", response.status_code)
  if response.status_code == 200:
    return response
  else:
    exit(1)

# prompts the LLM to suggest feature engineering strategies

def get_LLMResponseForFeatures(df, column_info, summary_stats):
    """
    Prompts the LLM to suggest feature engineering strategies.

    Args:
        df: The input DataFrame.
        column_info: A dictionary containing column names and their data types.
        summary_stats: Summary statistics of the DataFrame.

    Returns:
        A dictionary containing suggested features for dropping, one-hot encoding, and ordinal encoding.
    """

    prompt = f"""
    Do Data analysis for data: {df.head(10)} having columns: {column_info} and summary statistics: {summary_stats}.
    Identify features that can be dropped(ex. features that are not useful for predictive modeling or clustering).
    Identify features for one-hot encoding.
    Identify features for ordinal encoding.
    Provide only a JSON format response with the following structure:
    {{ "drop_features": ["feature1", "feature2", ...], "onehot_features": ["feature3", "feature4", ...], "ordinal_features": ["feature5", "feature6", ...]}}
    """

    # Send the prompt to the LLM (using your preferred method, e.g., get_LLMResponse)
    response = get_LLMResponse(prompt)

    if response.status_code == 200:
        return response
    else:
        print(f"Request failed with status code: {response.status_code}")
        return None

"""# Get Percentage of outliers in each column"""

# Get Percentage of outliers in each column

def get_outlier_percentage(df, column):
  """
  Function:
    Gets the percentage of outliers for given column.
  Args:
    df: Dataframe
    column: Column to analyse outliers
  Returns:
    Percentage of outliers in given column.
  """
  Q1 = df[column].quantile(0.25)
  Q3 = df[column].quantile(0.75)
  IQR = Q3 - Q1
  lower_bound = Q1 - 1.5 * IQR
  upper_bound = Q3 + 1.5 * IQR
  outliers = ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()
  total_entries = df[column].count()
  return (outliers / total_entries) * 100

"""# Identify Clusters in the given dataset"""

# To generate elbow chart to identify optimal K for clustering
def generate_elbow_chart(X_scaled, k_range=(1, 11), filename="elbow_chart.png"):
    """
    Generates an elbow chart for KMeans clustering and saves it as an image.

    Args:
        X_scaled: Scaled data for clustering.
        k_range: Range of k values to consider.
        filename: Name of the file to save the image (default: "elbow_chart.png").
    """
    wcss = []  # Within-cluster sum of squares

    for k in range(k_range[0], k_range[1]):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)

    # Plot the elbow method
    plt.figure(figsize=(8, 8))  # Adjust figure size for desired aspect ratio
    plt.plot(range(k_range[0], k_range[1]), wcss, marker='o')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('WCSS')
    plt.savefig(filename, dpi=64, bbox_inches='tight', pad_inches=0.5)  # Adjust dpi and padding as needed
    plt.show()

# Identify Clusters in the given dataset
def identify_clusters(X_scaled, k=3):
    """
    Identifies clusters in a DataFrame using KMeans clustering.
    Args:
        X_scaled: Scaled data for clustering.
        k: Number of clusters (default: 3).
    Returns:
        A DataFrame with cluster labels.
    """
    kmeans = KMeans(n_clusters=k, random_state=42)
    X_scaled['cluster'] = kmeans.fit_predict(X_scaled)
    return X_scaled

# Applies PCA to scaled data and retains components explaining a specified variance threshold.
def apply_pca_with_variance(X_scaled, variance_threshold=0.90):
    """
    Applies PCA to scaled data and retains components explaining a specified variance threshold.

    Args:
        X_scaled: Scaled data for PCA.
        variance_threshold: Desired variance explained (default: 0.80).

    Returns:
        A tuple containing:
            - pca: The fitted PCA object.
            - X_pca: The transformed data with reduced dimensions.
    """
    pca = PCA()  # Initialize PCA without specifying n_components
    pca.fit(X_scaled)  # Fit PCA to the scaled data

    # Calculate cumulative explained variance
    cumulative_variance = pca.explained_variance_ratio_.cumsum()

    # Find the number of components explaining the desired variance
    n_components = len(cumulative_variance[cumulative_variance <= variance_threshold]) + 1

    # if n_components == len(cumulative_variance) + 1: # if n_components is greater than or equal to maximum components required to explain the desired variance, then we will set n_componenets to a number to prevent error
    #     n_components = len(cumulative_variance)
    # print(f"Number of components explaining {variance_threshold*100:.2f}% variance: {n_components}")

    # Apply PCA with the determined number of components
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    return pca, X_pca

"""# Preprocess data"""

# Imputes missing values, scales numerical features, and encodes categorical features.
def preprocess_data(df_data, numeric_features, categorical_features):
    """
    Imputes missing values, scales numerical features, and encodes categorical features.

    Args:
        df_data: The input DataFrame.
        numeric_features: A list of numerical features to process.
        categorical_features: A list of categorical features to process.

    Returns:
        A DataFrame with preprocessed features.
    """

    # 1. Impute and scale numerical features
    num_imputer = SimpleImputer(strategy='mean')
    df_data[numeric_features] = num_imputer.fit_transform(df_data[numeric_features])
    scaler = StandardScaler()
    df_data[numeric_features] = scaler.fit_transform(df_data[numeric_features])
    print(df_data.shape)

    # 2. Impute and encode categorical features
    cat_imputer = SimpleImputer(strategy='most_frequent')  # Impute with most frequent value
    df_data[categorical_features] = cat_imputer.fit_transform(df_data[categorical_features])
    # if No. of unique values in a column is more than 20, then drop the column
    for col in df_data[categorical_features]:
      if df_data[col].nunique() > 20:
        print(f"Dropping column '{col}' due to more than 20 unique values.")
        df_data = df_data.drop(columns=[col])
    categorical_features = [col for col in categorical_features if col in df_data.columns]
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')  # One-hot encoding
    encoded_data = encoder.fit_transform(df_data[categorical_features])
    encoded_df_data = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_features))
    df_data = df_data.drop(categorical_features, axis=1)  # Drop original categorical columns
    df_data = pd.concat([df_data, encoded_df_data], axis=1)  # Concatenate encoded features
    print(df_data.shape)
    return df_data

# Imputes missing values, scales numerical features, and encodes categorical features.
def imputeAndScale_NumericFeatures(df_data, numeric_features):
    """
    Imputes missing values, scales numerical features, and encodes categorical features.

    Args:
        df_data: The input DataFrame.
        numeric_features: A list of numerical features to process.
        categorical_features: A list of categorical features to process.

    Returns:
        A DataFrame with preprocessed features.
    """
    numeric_features = [col for col in numeric_features if col in df_data.columns]
    num_imputer = SimpleImputer(strategy='mean')
    df_data[numeric_features] = num_imputer.fit_transform(df_data[numeric_features])
    scaler = StandardScaler()
    df_data[numeric_features] = scaler.fit_transform(df_data[numeric_features])
    print(df_data.shape)
    return df_data

# Drops given features from dataframe
def dropFeaturesFromDataFrame(df, features):
    """
    Drops given features from dataframe

    Args:
        df: The input DataFrame.
        response_json: JSON containing LLM's feature dropping suggestions.

    Returns:
        DataFrame with dropped features and imputed numerical features.
    """
    df = df.drop(columns=drop_features, errors='ignore')
    return df

# Encodes features and imputes missing values for categorical and ordinal features.
def encodeWithOneHotEncoder(df, onehot_features):
    """
    Encodes features and imputes missing values for categorical and ordinal features.

    Args:
        df: The input DataFrame.
        response_json: JSON containing LLM's encoding suggestions.

    Returns:
        DataFrame with encoded and imputed features.
    """
    # Impute missing values before one-hot encoding
    cat_imputer = SimpleImputer(strategy='most_frequent')  # You can change the strategy
    df[onehot_features] = cat_imputer.fit_transform(df[onehot_features])
    for col in df[onehot_features]:
      if df[col].nunique() > 50:
        print(f"Dropping column '{col}' due to more than 30 unique values.")
        df = df.drop(columns=[col])
    onehot_features = [col for col in onehot_features if col in df.columns]
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_data = encoder.fit_transform(df[onehot_features])
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(onehot_features))
    df = df.drop(columns=onehot_features, errors='ignore')
    df = pd.concat([df, encoded_df], axis=1)
    return df

# Encodes features and imputes missing values for categorical and ordinal features.
def encodeWithOridinalEncoder(df, ordinal_features):
    """
    Encodes features and imputes missing values for categorical and ordinal features.

    Args:
        df: The input DataFrame.
        response_json: JSON containing LLM's encoding suggestions.

    Returns:
        DataFrame with encoded and imputed features.
    """
    # Impute missing values before one-hot encoding
    cat_imputer = SimpleImputer(strategy='most_frequent')  # You can change the strategy
    df[ordinal_features] = cat_imputer.fit_transform(df[ordinal_features])
    # Impute missing values before ordinal encoding
    ord_imputer = SimpleImputer(strategy='most_frequent')  # You can change the strategy
    df[ordinal_features] = ord_imputer.fit_transform(df[ordinal_features])

    encoder = OrdinalEncoder()
    df[ordinal_features] = encoder.fit_transform(df[ordinal_features])

    return df

"""# Chi2 test"""

# Performs Chi-squared test for independence between pairs of categorical features.
def chi2_test_for_features(df, categorical_features):
    """
    Performs Chi-squared test for independence between pairs of categorical features.

    Args:
        df: The input DataFrame.
        categorical_features: A list of categorical feature columns.

    Returns:
        A dictionary containing Chi-squared statistics and p-values for each pair of features.
    """
    results = {}
    for feature1 in categorical_features:
        for feature2 in categorical_features:
            if feature1 != feature2:  # Avoid comparing a feature to itself
                # Create contingency table
                contingency_table = pd.crosstab(df[feature1], df[feature2])

                # Perform Chi-squared test
                chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)

                # Store results
                results[(feature1, feature2)] = {
                    "chi2_statistic": chi2_stat,
                    "p_value": p_value,
                    "degrees_of_freedom": dof,
                   # "expected_frequencies": expected
                }
    return results

"""# Generate cluster map in 2D"""

# Apply PCA to reduce dimensionality to 2 and generate cluster map
def generate_cluster_map(X_scaled):
    """
    Applies PCA to reduce dimensionality to 2 and generates a cluster map.

    Args:
        X_scaled: Scaled data for clustering.

    Returns:
        A DataFrame with cluster labels and PCA coordinates.
    """
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)


    # Visualize clusters using scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=X_scaled['cluster'], cmap='viridis')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Clusters Visualization')
    plt.savefig('clusters_2d.png', dpi=64)
    plt.show()
    return df

"""# Copy the files to correct directory"""

# Copy the files to correct directory
def copy_file(source_path, destination_path):
    """
    Copies a file from the source path to the destination path.

    Args:
        source_path: Path to the source file.
        destination_path: Path to the destination file.
    """
    with open(source_path, 'rb') as source_file, open(destination_path, 'wb') as destination_file:
      destination_file.write(source_file.read())

"""# Generate the markdown file from the response got from LLM"""

# Generate the markdown file from the response got from LLM
def generate_report(df, numeric_columns, file_path, response_json):
    """
    Generates a report folder with README.md and cluster image.

    Args:
        df: The input DataFrame.
        numeric_columns: List of numerical columns for clustering.
        file_path: Path to the CSV file.
        response_json: JSON response from OpenAI.
    """

    # 1. Create folder named after CSV file (without extension)
    file_name = os.path.splitext(os.path.basename(file_path))[0]  # Extract filename without extension
    folder_path = file_name  # Create folder with the same name
    os.makedirs(folder_path, exist_ok=True)  # Create folder if it doesn't exist

    # 2. Copy cluster image to the folder
    image_source = 'clusters_2d.png'
    image_destination = os.path.join(folder_path, image_source)
    copy_file(image_source, image_destination)  # Copy image to folder
    image_source = 'heatmap.png'
    image_destination = os.path.join(folder_path, image_source)
    copy_file(image_source, image_destination)  # Copy image to folder


    # 3. Generate README.md from OpenAI response JSON
    try:
        markdown_content = response_json['choices'][0]['message']['content']
        readme_path = os.path.join(folder_path, 'README.md')
        # Add image to README.md
        markdown_content = markdown_content.replace("```markdown","")
        markdown_content = markdown_content.replace("```", "")
        with open("heatmap.png", "rb") as image_file:
          encoded_string = base64.b64encode(image_file.read()).decode()
          markdown_content = markdown_content.replace("heatmap.png", f"data:image/png;base64,{encoded_string}")
        with open("clusters_2d.png", "rb") as image_file:
          encoded_string = base64.b64encode(image_file.read()).decode()
          markdown_content = markdown_content.replace("clusters_2d.png", f"data:image/png;base64,{encoded_string}")

        with open(readme_path, 'w') as f:
            f.write(markdown_content)
    except (KeyError, IndexError) as e:
        print(f"Error extracting README content from JSON: {e}")
        print("Using default README content.")
        # ... (add your default README content here) ...

    print(f"Report generated in folder: {folder_path}")

"""# Apply VarianceThreshold for feature selection"""

# Apply VarianceThreshold for feature selection
def apply_variance_threshold(df, threshold=0.05):
  """
  Applies VarianceThreshold for feature selection.

  Args:
      df: The input DataFrame.
      threshold: Variance threshold (default: 0.05).

  Returns:
      A DataFrame with selected features.
  """
  selector = VarianceThreshold(threshold=0.01)  # Adjust threshold as needed
  X_scaled_selected = selector.fit_transform(df)

  # Get selected feature names
  selected_features = [df.columns[i] for i in range(len(df.columns)) if selector.get_support()[i]]

  # Create a new DataFrame with selected features
  X_scaled_selected_df = pd.DataFrame(X_scaled_selected, columns=selected_features)
  print('df_shape: ', df.shape,'X_scaled_selected_df.shape: ', X_scaled_selected_df.shape)
  return X_scaled_selected_df

"""# Main Function"""

############ MAIN FUNCTION############

if __name__ == "__main__":
  if len(sys.argv) != 2:
    print("Usage: python script_name.py <csv_file_path>")
    sys.exit(1)

  file_path = sys.argv[1]
  df = read_csv(file_path)
  if df is None:
      exit(1)
  descriptive_stats = get_descriptive_stats(df)
  #print(descriptive_stats)
  df_info = analyze_df(df)
  numeric_columns, categorical_columns = get_column_types(df)
  corr_mat = generate_correlation_heatmap(df)
  outlier_percentages = {}
  for column in numeric_columns:
    outlier_percentage = get_outlier_percentage(df, column)
    outlier_percentages[column] = outlier_percentage

  print("Outlier Percentages:")
  for column, percentage in outlier_percentages.items():
    print(f"  {column}: {percentage:.2f}%")
  column_info = df.dtypes.to_dict()  # Column names and types
  summary_stats = df.describe().to_string()  # Summary statistics
  response = get_LLMResponseForFeatures(df, column_info, summary_stats)
  # Extract the content from the response
  content = response.json()['choices'][0]['message']['content']
  print(content)
  content = content.replace("```json", "").replace("```", "")
  try:
      content = json.loads(content)
  except json.JSONDecodeError as e:
      print(f"Error decoding JSON: {e}")
      # Handle the error, e.g., exit or use default values
      content = {}  # or some default value
  df_new = pd.DataFrame()
  drop_features = content.get("drop_features", [])
  if drop_features:
    df_new_min = dropFeaturesFromDataFrame(df, drop_features)
    df_new = df_new_min.copy()
  else:
    df_new = df
  onehot_features = content.get("onehot_features", [])
  if onehot_features:
    df_new = encodeWithOneHotEncoder(df_new, onehot_features)
  else:
    df_new = df_new
  ordinal_features = content.get("ordinal_features", [])
  if ordinal_features:
    df_new = encodeWithOridinalEncoder(df_new, ordinal_features)
  else:
    df_new = df_new
  # get categorical features from df
  cat_features = [col for col in df_new.columns if df_new[col].dtype == 'object']
  if cat_features:
    df_new = encodeWithOneHotEncoder(df_new, cat_features)
  else:
    df_new = df_new
  df_new_scaled = imputeAndScale_NumericFeatures(df_new, numeric_columns)
  # perform chi2 test if df_new_min has more than 1 categorical features

  categorical_columns = [col for col in df_new_min.columns if df_new_min[col].dtype == 'object']
  if len(categorical_columns) > 1:
    chi2_feature_results = chi2_test_for_features(df_new_min, categorical_columns)
    print("Chi-squared Test Results for Features:")
    print(chi2_feature_results)
  else:
    chi2_feature_results = 'Chi2 test could not be performed as no. of categorical variable is only one'
  # Apply variance threshold on the scaled dataframe
  X_scaled_selected_df = apply_variance_threshold(df_new_scaled)
  generate_elbow_chart(X_scaled_selected_df)
  prompt = "I have an elbow chart for KMeans clustering. Determine the optimal number of clusters (k) from chart.Return only optimal k as a dictionary variable"
  response = get_LLMResponseForImage('elbow_chart.png', prompt)
  # Extract the content from the response
  content = response.json()['choices'][0]['message']['content']

  # Use a regular expression to find the integer value of k
  match = re.search(r'\d+', content)
  # Let the optimal k value be 3 by default
  optimal_k=3
  if match:
    optimal_k = int(match.group())
    print(f"Optimal k: {optimal_k}")
  clustered_df = identify_clusters(X_scaled_selected_df,optimal_k )
  generate_cluster_map(clustered_df)
  prompt = f"""
  Generate a README.md that has analysis of data. It should have heading ```Data Analysis Report for file name: {file_path}.
  1st header should be 'Introduction' which should include objectives like: basic information about the data, and what could be modeled from the data.
  2nd header should be 'Dataset Overview'. This should include Filename,number of samples, number of features. this information can be got from filename:{file_path} and data info:{df_info}.
  3rd header should be 'Data Description'. This should include each column with data type, number of uniqe values, percentage of missing values and description of each feature as a table. This can be got from data info, column_info:{column_info} and summary_stats:{summary_stats}.
  4th header should be 'Data Quality'. This should have give the columns having significant missing values and significant percent of outlier. percentage of outliers can be got from outlier_percentage:{outlier_percentages}. Give the inference from chi2 test from chi2_feature_results:{chi2_feature_results}.
  5th header should be 'Data Visualization'. Here show the image cluster: clusters_2d.png and correlation heatmap: heatmap.png
  7th header should be 'Data Processing Steps'. Here explain that Data processing steps like Imputations, onehotencoding, ordinal encoding, scaling and PCA has been used.
  8th header should be 'Key Insights and Next Steps'. Here write your key insights and next steps and suggestions.
  """

  # Send the prompt to the LLM (using your preferred method, e.g., get_LLMResponse)
  response = get_LLMResponse(prompt)
  generate_report(df, numeric_columns, file_path, response.json())