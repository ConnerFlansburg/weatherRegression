"""
https://colab.research.google.com/drive/1rRQF0bvKSTOR4ZzLEVhRi7Z3GFzzmMQ2#scrollTo=0KMc_bkEwbbn
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pathlib as pth
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings('ignore')

total_dataset = pd.read_csv(str(pth.Path.cwd() / 'data' / 'kdfw_processed_data.csv'), index_col=0).sort_values(by='date')
total_dataset = total_dataset.replace('********', np.nan).replace(np.inf,np.nan).dropna(how='any', axis=1)

# * 3 Data Preprocessing * #


# * Partitioning Data * #
# Function to split data based on given dates
def split_data_year(input_data, labels, start_date_str, end_date_str):
    data = input_data.copy()
    date_list = pd.to_datetime(data['date'])
    date_mask = (date_list > start_date_str) & (date_list <= end_date_str)
    out_data = data.loc[date_mask, :].drop(['date'], axis=1)
    out_labels = labels.loc[date_mask, :]
    return out_data, out_labels


# Remove MOS data based on previous work
mosCols = [key for key in total_dataset.columns if 'MOS' in key]
total_dataset = total_dataset.drop(mosCols, axis=1)
# Get total data
total_label_data = total_dataset.filter(like='OBS')
dropCols = list(total_label_data.columns)
total_feature_data = total_dataset.copy(deep=True).drop(dropCols,axis=1)

# Training data between 2011 and 2017
train_features, train_labels = split_data_year(total_feature_data, total_label_data,
                                               '2011-01-01', '2017-12-31')

# Testing data between 2018 and 2019
# Keep a month between train/test data for greater likelihood
# of dataset independence
test_features, test_labels = split_data_year(total_feature_data, total_label_data,
                                             '2018-02-01', '2019-12-31')

# * Transforming Data * #
no_outlier_train_features = train_features.copy()
no_outlier_test_features = test_features.copy()

train_standard_dev = train_features.std()
train_mean = train_features.mean()

for column in train_features.columns:
    outlier_threshold_value = 3.0 * train_standard_dev[column]

    # pandas.where() documentation:
    # Where cond is True, keep the original value. Where False, replace with corresponding value from other
    no_outlier_train_features[column].where(
        np.abs(train_features[column] - train_mean[column]) < outlier_threshold_value,
        train_mean[column],
        inplace=True)

    no_outlier_test_features[column].where(
        np.abs(test_features[column] - train_mean[column]) < outlier_threshold_value,
        train_mean[column],
        inplace=True)


no_outlier_min_max_model = MinMaxScaler().fit(no_outlier_train_features)
scaled_no_outlier_train_features = no_outlier_min_max_model.transform(no_outlier_train_features)
scaled_no_outlier_test_features = no_outlier_min_max_model.transform(no_outlier_test_features)


# * Dimensionality Reduction * #
pca_model = PCA(n_components=0.65, svd_solver='full').fit(scaled_no_outlier_train_features)
pca_train_features = pca_model.transform(scaled_no_outlier_train_features)
pca_test_features = pca_model.transform(scaled_no_outlier_test_features)

print(f'shape: {pca_train_features.shape}')


# * 4 Model Training * #

high_temp_linreg_model = LinearRegression()
# High Temperature Model
high_temp_linreg = high_temp_linreg_model.fit(pca_train_features,train_labels['OBS_tmpf_max'].values)

low_temp_linreg_model = LinearRegression()
# Low Temperature Model
low_temp_linreg = low_temp_linreg_model.fit(pca_train_features,train_labels['OBS_tmpf_min'].values)

wind_linreg_model = LinearRegression()
# Max Wind Speed Model
wind_linreg = wind_linreg_model.fit(pca_train_features,train_labels['OBS_sknt_max'].values)


# * 5 Evaluation * #
high_temp_y_pred_linreg = high_temp_linreg.predict(pca_test_features)
high_temp_mse_linreg = mean_squared_error(high_temp_y_pred_linreg,test_labels['OBS_tmpf_max'].values)

print(f'Mean square error of Linear Regression: {high_temp_mse_linreg}')






