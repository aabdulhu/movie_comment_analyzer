# Seed value
# Apparently you may use different seed values at each stage
seed_value= 1

# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set the `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set the `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)

import pandas as pd
import numpy as np
import optuna
#import xgboost as xgb

from numpy import absolute, std, mean
#from lightgbm import LGBMRegressor
from math import sqrt


from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, BaggingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet, RidgeCV, HuberRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error,r2_score, mean_absolute_error,make_scorer

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
sentimentVader = SentimentIntensityAnalyzer()

# nltk.download('all')

def adjr2(y_test, y_pred):

  #display adjusted R-squared
  r2 = r2_score(y_test,y_pred)
  x = (1-r2)*(len(y_test)-1)
  y = ((len(y_test))-X.shape[1]-1)
  Adjusted = 1-(x/y)

  return Adjusted


# Read the dataset
df = pd.read_csv("FinalDataset.csv")
df.info()

# Dimensions of the dataset
df.shape

# What data types do we have in the dataset?
df.dtypes

# Does the dataset contain any NAs?
df.isna().any()

# Let's check some summary statistics (only applicable to numeric features)
df.describe()


# # Plot histogram
# plt.hist(df['Female'], bins=10, edgecolor='black')
# plt.title('Frequency Distribution of Female')
# plt.xlabel('Female')
# plt.ylabel('Frequency')
# plt.grid(True)
# plt.show()


# # Plot histogram
# plt.hist(df['RatioFemale'], bins=10, edgecolor='black')
# plt.title('Frequency Distribution of RatioFemale')
# plt.xlabel('RatioFemale')
# plt.ylabel('Frequency')
# plt.grid(True)
# plt.show()

# # Get descriptive statistics for RatioFemale
# ratio_female_stats = df['RatioFemale'].describe()

# print("Descriptive Statistics for RatioFemale:")
# print(ratio_female_stats)

# """Based on this data, use ratio female becausde it has fewer 0s."""

# import matplotlib.pyplot as plt

# # Plot histogram
# plt.hist(df['StarsScore'], bins=10, edgecolor='black')
# plt.title('Frequency Distribution of StarsScore')
# plt.xlabel('StarsScore')
# plt.ylabel('Frequency')
# plt.grid(True)
# plt.show()

# # Assuming df is your DataFrame
# top_studios_frequency = df['TopStudios'].value_counts()

# # Compute the total number of observations
# total_observations = len(df)

# # Calculate count and percentage for each category
# top_studios_count_percentage = pd.DataFrame({
#     'Count': top_studios_frequency,
#     'Percentage': (top_studios_frequency / total_observations) * 100
# })

# # Display the count and percentage table
# print(top_studios_count_percentage)

# from scipy.stats import pearsonr

# # Calculate Pearson correlation coefficient and p-value
# corr_coef, p_value = pearsonr(df['RatioBlack'], df['RatioWhite'])

# print("Correlation coefficient between RatioBlack and RatioWhite:", corr_coef)
# print("P-value for the correlation coefficient:", p_value)

"""Based on this, drop whites from data."""

# Checking if we have all features ready / we need
df.head(5)

# List of features you want to use
selected_numer_features = ['TrailerPublishYear','TrailerPublishDays','TrailerDuration','ProductionBudget','RatioFaceNo','FaceNo','RatioFemale', 'AvgFaceSize', 'RatioFaceCoverage', 'AverageAge','RatioSad','RatioHappy', 'RatioFear',  'RatioAngry','RatioSurprise',
'RatioDisgust', 'RatioAsian', 'RatioIndian', 'RatioBlack', 'RatioMiddle','RatioHispanic', 'TotalComments'
]
selected_cat_features = ['Action','Comedy',
'Documentary','Drama', 'PG-13','R','Not Rated','TopStars','TopStudios', 'ProductionCountry'
]

selected_features = ['TrailerPublishYear','TrailerPublishDays','TrailerDuration','ProductionBudget', 'Action','Comedy',
'Documentary','Drama', 'PG-13','R','Not Rated','TopStars','TopStudios', 'ProductionCountry',
'RatioFaceNo','FaceNo','RatioFemale', 'AvgFaceSize', 'RatioFaceCoverage', 'AverageAge','RatioSad','RatioHappy', 'RatioFear',  'RatioAngry','RatioSurprise',
'RatioDisgust', 'RatioAsian', 'RatioIndian', 'RatioBlack', 'RatioMiddle','RatioHispanic', 'TotalComments'
]

cd2 = pd.read_csv("Comments\FilteredComments2.csv")
cd3 = pd.read_csv("Comments\FilteredComments3.csv")
cd4 = pd.read_csv("Comments\FilteredComments4.csv")
cd5 = pd.read_csv("Comments\FilteredComments5.csv")
cd6 = pd.read_csv("Comments\FilteredComments6.csv")
cd7 = pd.read_csv("Comments\FilteredComments7.csv")
cd8 = pd.read_csv("Comments\FilteredComments8.csv")
cd9 = pd.read_csv("Comments\FilteredComments9.csv")
cd10= pd.read_csv("Comments\FilteredComments10.csv")
cd11= pd.read_csv("Comments\FilteredComments11.csv")
cd12= pd.read_csv("Comments\FilteredComments12.csv")
cd13= pd.read_csv("Comments\FilteredComments13.csv")
cd14= pd.read_csv("Comments\FilteredComments14.csv")
cd15= pd.read_csv("Comments\FilteredComments15.csv")

cd_append = pd.concat([cd2, cd3, cd4, cd5, cd6, cd7, cd8, cd9, cd10, cd11, cd12, cd13, cd14, cd15], ignore_index=True)
cd_append = cd_append.drop_duplicates()

# movie_comments = cd_append[cd_append.MovieLink == "https://www.imdb.com/title/tt2735292/?ref_=adv_li_tt"]

# create preprocess_text function
def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text.lower())
    # Remove stop words
    filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]
    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    # Join the tokens back into a string
    processed_text = ' '.join(lemmatized_tokens)
    return processed_text

# initialize NLTK sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# create get_sentiment function
def get_sentiment(text):
    scores = analyzer.polarity_scores(text)
    sentiment = 1 if scores['pos'] > 0 else 0
    return sentiment




# s_score = sentiment.polarity_scores("The book was a perfect balance between wrtiting style and plot.")

# print(s_score)
# print(s_score['compound'])
# sys.exit()

df['Sentiment_mean'] = 0
df['PN_ratio'] = 0

for x in range(len(df['Movie_Link'])):

    cd_append_comments = cd_append[cd_append.MovieLink == df['Movie_Link'][x]]
    cd_append_comments = cd_append_comments.reset_index()

    #NLTK stuff
    # cd_append_comments['Comments_pro'] = cd_append_comments['Comments'].apply(preprocess_text)
    # cd_append_comments['Sentiment'] = cd_append_comments['Comments_pro'].apply(get_sentiment)

    sentiment = 0
    pos_sentiment = 0
    neg_sentiment = 0
    sentiment_ratio = 0

    #Blob stuff
    for k in cd_append_comments['Comments']:
            blob = TextBlob(k)
            sentiment = blob.sentiment.polarity
            if (sentiment>=0):
                pos_sentiment +=1
            else:
                neg_sentiment +=1

    #Vader stuff
    # for k in cd_append_comments['Comments']:
    #         s_score = sentimentVader.polarity_scores(k)
    #         sentiment = s_score['compound']
    #         if (sentiment>0.05):
    #             pos_sentiment +=1
    #         elif (sentiment<0.05):
    #             neg_sentiment +=1
    

    if neg_sentiment == 0:
        neg_sentiment = 1
        print("NEGATIVE_COMMENT ZERO")

    df.loc[x, "PN_ratio"] = np.float64(pos_sentiment) 

    # PN_ratio = np.float64(pos_sentiment) 

    # if(df.loc[x, "TotalComments"] <= 500):
    #     df.loc[x, "PN_ratio"] = np.float64(PN_ratio*1)
    # elif(df.loc[x, "TotalComments"] > 500 and df.loc[x, "TotalComments"] <= 1000 ):
    #     df.loc[x, "PN_ratio"] = np.float64(PN_ratio*2)
    # elif(df.loc[x, "TotalComments"] > 1000 and df.loc[x, "TotalComments"] <= 2000 ):
    #     df.loc[x, "PN_ratio"] = np.float64(PN_ratio*3)
    # else:
    #     df.loc[x, "PN_ratio"] = np.float64(PN_ratio*4)

    print(x, df.loc[x, "TotalComments"],pos_sentiment,neg_sentiment, df.loc[x, "PN_ratio"])
    

df.to_csv("Sentiments.csv")

df['TotalComments'] = df['PN_ratio']


# Select only the desired features
df = df[selected_features]
df_numer=df[selected_numer_features]
# Print the dataset to verify the changes
print(df.head())

#Discover outliers with mathematical function Z-Score-
# sns.boxplot(x=df['TotalComments'])

# Function to identify and exclude dummy variables
def get_numerical_non_dummy_cols(df):
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    non_dummy_cols = [col for col in numerical_cols if df[col].nunique() > 2 or not set(df[col].unique()).issubset({0, 1})]
    return non_dummy_cols

# Function to remove outliers using Z-score method
def remove_outliers_zscore(df, threshold=3):
    # Select only numerical columns that are not dummy variables
    numerical_cols = get_numerical_non_dummy_cols(df)

    # Calculate Z-scores
    z_scores = np.abs(stats.zscore(df[numerical_cols]))

    # Keep only rows where all Z-scores are below the threshold
    df_cleaned = df[(z_scores < threshold).all(axis=1)]

    return df_cleaned

# Remove outliers using Z-score method
df_cleaned = remove_outliers_zscore(df)

# Print the cleaned DataFrame
print(df_cleaned.head())

print(df_cleaned)

df_cleaned.shape

df_cleaned.info()

"""# Linear regression - transformed skewed numerical features"""

# Assuming df_cleaned is your DataFrame after removing outliers
df = df_cleaned

# Define the feature sets (as provided in the question)
all_features = [
    'TrailerPublishYear','TrailerPublishDays', 'TrailerDuration', 'ProductionBudget', 'Action', 'Comedy', 'Documentary', 'Drama', 'PG-13', 'R', 'Not Rated',
    'TopStars', 'TopStudios', 'ProductionCountry', 'RatioFaceNo', 'FaceNo',
    'AvgFaceSize', 'RatioFaceCoverage', 'RatioFemale', 'AverageAge', 'RatioSad',
    'RatioHappy', 'RatioFear', 'RatioAngry', 'RatioSurprise', 'RatioDisgust',
    'RatioAsian', 'RatioIndian', 'RatioBlack', 'RatioMiddle', 'RatioHispanic'
]

# Define features for the control models
cont_features = [
    'TrailerPublishYear','TrailerPublishDays', 'TrailerDuration', 'ProductionBudget', 'Action', 'Comedy', 'Documentary', 'Drama', 'PG-13', 'R', 'Not Rated',
    'TopStars', 'TopStudios', 'ProductionCountry'
]

cont_fn_features = cont_features + ['RatioFaceNo', 'FaceNo']
cont_fs_features = cont_features + ['AvgFaceSize', 'RatioFaceCoverage']
cont_gender_features = cont_features + ['RatioFemale']
cont_age_features = cont_features + ['AverageAge']
cont_emo_features = cont_features + ['RatioSad', 'RatioHappy', 'RatioFear', 'RatioAngry', 'RatioSurprise', 'RatioDisgust']
cont_race_features = cont_features + ['RatioAsian', 'RatioIndian', 'RatioBlack', 'RatioMiddle', 'RatioHispanic']

# Function to apply transformations to skewed features
def transform_skewed_features(df, skewed_features):
    df_transformed = df.copy()  # Make a copy to avoid modifying the original DataFrame
    for feature in skewed_features:
        if df_transformed[feature].skew() > 1:  # Check if the feature is skewed
            # Apply Box-Cox transformation
            df_transformed[feature], _ = stats.boxcox(df_transformed[feature] + 1)  # Add 1 to handle zero values

            # Uncomment one of the following lines to use an alternative transformation

            # Apply logarithmic transformation
            # df_transformed[feature] = np.log1p(df_transformed[feature])  # log1p to handle zero values

            # Apply square root transformation
            # df_transformed[feature] = np.sqrt(df_transformed[feature])

    return df_transformed

# List of numerical features that are potentially skewed
numerical_features = ['TrailerPublishDays', 'TrailerDuration', 'ProductionBudget', 'RatioFaceNo', 'FaceNo',
                      'AvgFaceSize', 'RatioFaceCoverage', 'RatioFemale', 'AverageAge', 'RatioSad',
                      'RatioHappy', 'RatioFear', 'RatioAngry', 'RatioSurprise', 'RatioDisgust',
                      'RatioAsian', 'RatioIndian', 'RatioBlack', 'RatioMiddle', 'RatioHispanic']

# Identify skewed features based on a skewness threshold
skewed_features = [feature for feature in numerical_features if df[feature].skew() > 1]

# Apply transformations to skewed features
df_transformed = transform_skewed_features(df, skewed_features)

# Transform the target variable if it's skewed
if df['TotalComments'].skew() > 1:
    # df_transformed['TotalComments'] = np.sqrt(df['TotalComments'])
    # df_transformed['TotalComments'] = np.log1p(df['TotalComments'])
    df_transformed['TotalComments'], _ = stats.boxcox(df['TotalComments'] + 1)

# dcorr = df_transformed.corr()
# dcorr.to_excel('corr-matrix.xlsx', index=False)

# Define the target variable
y = df_transformed['TotalComments'].values


# Function to train and evaluate a model
def train_and_evaluate_model(features, model_name):
    # Prepare the data
    X = df_transformed[features]

    # Train the model
    model = LinearRegression()
    model.fit(X, y)

    # r_sq = model.score(X, y)
    # print("For features: ",features)
    # print("R_squared: ",r_sq)

    # Predict and evaluate on the same data
    y_pred = model.predict(X)

    # Calculate residuals
    residuals = y - y_pred

    # Calculate metrics
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    n = len(y)
    p = X.shape[1]
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

    # Use statsmodels to get the p-values
    X_sm = sm.add_constant(X)  # Add a constant to the model (intercept)
    model_sm = sm.OLS(y, X_sm).fit()

    # Extract coefficients and p-values
    coefs = model_sm.params.values
    pvals = model_sm.pvalues

    return {
        'Model': model_name,
        'Features': features,
        'R-squared': r2,
        'Adj. R-squared': adj_r2,
        'MSE': mse,
        'MAE': mae,
        'Coefficients': coefs,
        'P-values': pvals,
        'y_pred': y_pred,
        'residuals': residuals
    }

# List of models and their corresponding features
models = [
    {'name': 'all', 'features': all_features},
    {'name': 'cont', 'features': cont_features}
    # {'name': 'cont_fn', 'features': cont_fn_features},
    # {'name': 'cont_fs', 'features': cont_fs_features},
    # {'name': 'cont_gender', 'features': cont_gender_features},
    # {'name': 'cont_age', 'features': cont_age_features},
    # {'name': 'cont_emo', 'features': cont_emo_features},
    # {'name': 'cont_race', 'features': cont_race_features}
]


# Train and evaluate each model
results = []
for model in models:
    model_results = train_and_evaluate_model(model['features'], model['name'])
    results.append(model_results)

# Retrieve predictions and residuals from one of the models
model_to_plot = results[0]  # Example: using the first model for residual plots
y_pred = model_to_plot['y_pred']
residuals = model_to_plot['residuals']

# Prepare the results for Excel export
excel_data = []

for result in results:
    for feature, coef, pval in zip(['Intercept'] + result['Features'], result['Coefficients'], result['P-values']):
        excel_data.append({
            'Model': result['Model'],
            'Feature': feature,
            'Coefficient': coef,
            'P-value': pval,
            'R-squared': result['R-squared'],
            'Adj. R-squared': result['Adj. R-squared'],
            'MSE': result['MSE'],
            'MAE': result['MAE']
        })

# Convert to DataFrame
excel_df = pd.DataFrame(excel_data)

# Export the results to an Excel file
excel_df.to_excel('regression_results.xlsx', index=False)

# Print the DataFrame for verification
print(excel_df)

# Generate the LaTeX table
latex_table = excel_df.to_latex(index=False)

# Save the LaTeX table to a file
with open('regression_results.tex', 'w') as f:
    f.write(latex_table)

# print(latex_table)

df.shape

# # Plot Residuals vs. Fitted Values
# plt.figure(figsize=(10, 6))
# plt.scatter(y_pred, residuals)
# plt.axhline(y=0, color='r', linestyle='--')
# plt.xlabel('Fitted values')
# plt.ylabel('Residuals')
# plt.title('Residuals vs. Fitted Values')
# plt.show()

# # Histogram of residuals
# plt.figure(figsize=(10, 6))
# sns.histplot(residuals, kde=True)
# plt.xlabel('Residuals')
# plt.title('Histogram of Residuals')
# plt.show()

# # Q-Q plot
# sm.qqplot(residuals, line='45')
# plt.title('Q-Q Plot of Residuals')
# plt.show()

# # Residuals vs. Predicted values
# plt.figure(figsize=(10, 6))
# plt.scatter(y_pred, residuals)
# plt.axhline(y=0, color='r', linestyle='--')
# plt.xlabel('Predicted values')
# plt.ylabel('Residuals')
# plt.title('Residuals vs. Predicted Values')
# plt.show()

# # Residuals vs. Actual values
# plt.figure(figsize=(10, 6))
# plt.scatter(y, residuals)
# plt.axhline(y=0, color='r', linestyle='--')
# plt.xlabel('Actual values')
# plt.ylabel('Residuals')
# plt.title('Residuals vs. Actual Values')
# plt.show()

# """# KDE plots transformed"""

# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from scipy import stats

# # Generate KDE plots for original and transformed features
# for feature in df_transformed:
#     plt.figure(figsize=(12, 6))

#     # KDE plot for the original feature
#     plt.subplot(1, 2, 1)
#     sns.kdeplot(df[feature], fill=True, color="blue")
#     plt.title(f'Original {feature}')

#     # KDE plot for the transformed feature
#     plt.subplot(1, 2, 2)
#     sns.kdeplot(df_transformed[feature], fill=True, color="green")
#     plt.title(f'Transformed {feature}')

#     plt.show()