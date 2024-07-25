# URLs-Wandb
#This project aims to classify URLs as phishing or legitimate using machine learning techniques. The approach involves preprocessing URL data, feature extraction using TF-IDF, and training a Logistic Regression model.

Requirements
Make sure to install the necessary libraries before running the code:

bash
Copy code
pip install wandb scikit-learn pandas
Project Setup
1. Install Libraries
python
Copy code
!pip install wandb scikit-learn
2. Import Libraries
python
Copy code
import wandb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
3. Weights & Biases Setup
Login to Weights & Biases and initialize a new run:

python
Copy code
wandb.login()
wandb.init(project="phishing-url-classification", entity="your_wandb_entity")
Replace "your_wandb_entity" with your actual Weights & Biases entity.

4. Load Dataset
Load the dataset using pandas:

python
Copy code
url_data = pd.read_csv('/content/urlset.csv', encoding='ISO-8859-1', error_bad_lines=False, warn_bad_lines=True)
5. Preprocess Data
Label Encoding: Convert labels to numeric values.
python
Copy code
le = LabelEncoder()
url_data['label'] = le.fit_transform(url_data['label'])
Feature Extraction: Extract features from URLs using TF-IDF.
python
Copy code
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(url_data['url'])  # Assuming the column with URLs is named 'url'
y = url_data['label']
6. Split Dataset
Split the data into training and testing sets:

python
Copy code
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
7. Train Model
Initialize and train the Logistic Regression model:

python
Copy code
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)
8. Evaluate Model
Make predictions and evaluate accuracy:

python
Copy code
y_pred_lr = lr_model.predict(X_test)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print(f"Logistic Regression Accuracy: {accuracy_lr}")
9. Log Results
Log the results to Weights & Biases:

python
Copy code
wandb.log({"Logistic Regression Accuracy": accuracy_lr})
10. Finish Run
End the Weights & Biases run:

python
Copy code
wandb.finish()
Dataset
The dataset used in this project is expected to be in CSV format with at least two columns:

url: Contains the URL strings.
label: Contains the labels indicating whether the URL is phishing or legitimate.
License
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments
Weights & Biases for experiment tracking and visualization.
Scikit-learn for machine learning algorithms.
