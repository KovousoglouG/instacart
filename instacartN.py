# 1. Preparation

## 1.1. Import packages


from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile
import os

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
#import matplotlib.pyplot as plt

import gc
gc.enable() 


api = KaggleApi({"username":"ntinoskoumentakis","key":"19a9fe24e405b6eb01900a3fbd4b5419"})
api.authenticate()
files = api.competition_download_files("Instacart-Market-Basket-Analysis")


with zipfile.ZipFile('Instacart-Market-Basket-Analysis.zip', 'r') as zip_ref:
    zip_ref.extractall('./input')


working_directory = os.getcwd()+'/input'
os.chdir(working_directory)
for file in os.listdir(working_directory):   # get the list of files
    if zipfile.is_zipfile(file): # if it is a zipfile, extract it
        with zipfile.ZipFile(file) as item: # treat the file as a zip
           item.extractall()  # extract it in the working directory


## 1.2. Load data


orders = pd.read_csv('../input/orders.csv' )
order_products_train = pd.read_csv('../input/order_products__train.csv')
order_products_prior = pd.read_csv('../input/order_products__prior.csv')
products = pd.read_csv('../input/products.csv')
#aisles = pd.read_csv('../input/aisles.csv')
#departments = pd.read_csv('../input/departments.csv')


#orders = orders.loc[orders.user_id.isin(orders.user_id.drop_duplicates().sample(frac=0.05, random_state=25))]


orders.head()


order_products_train.head()


order_products_prior.head()


products.head()


#aisles.head()


#departments.head()


## 1.3. Reshape data


orders['eval_set'] = orders['eval_set'].astype('category')
products['product_name'] = products['product_name'].astype('category')
#aisles['aisle'] = aisles['aisle'].astype('category')
#departments['department'] = departments['department'].astype('category')


# 2. Create predictor variables


op = orders.merge(order_products_prior, on='order_id', how='inner')
op.head()


del order_products_prior
gc.collect()


## 2.1. User predictors

### 2.1.1. Number of orders per user


user = op.groupby('user_id')['order_number'].max().to_frame('user_orders')
user = user.reset_index()
user.head()


### 2.1.2. Frequency of reordered products per user


user_reorder_frequency = op.groupby('user_id')['reordered'].mean().to_frame('user_reorder_frequency')
user_reorder_frequency = user_reorder_frequency.reset_index()
user_reorder_frequency.head()


user = user.merge(user_reorder_frequency, on='user_id', how='left')
del user_reorder_frequency
gc.collect()
user.head()


### 2.1.3. Days since last order of a user


user_days_since_last_order = orders[(orders.eval_set == 'train') | (orders.eval_set == 'test')][['user_id', 'days_since_prior_order']]
user_days_since_last_order.columns = ['user_id', 'user_days_since_last_order']
user_days_since_last_order.head()


user = user.merge(user_days_since_last_order, on='user_id', how='left')
del user_days_since_last_order
gc.collect()
user.head()


## 2.2. Product predictors

### 2.2.1. Number of orders per product


product = op.groupby('product_id')['order_id'].count().to_frame('product_orders')
product = product.reset_index()
product.head()


### 2.2.2. Average product position in the cart


product_average_position = op.groupby('product_id')['add_to_cart_order'].mean().to_frame('product_average_position')
product_average_position = product_average_position.reset_index()
product_average_position.head()


product = product.merge(product_average_position, on='product_id', how='left')
del product_average_position
gc.collect()
product.head()


### 2.2.3 Probability for a product to be reordered


product_reorder_probability = op.groupby('product_id')['reordered'].mean().to_frame('product_reorder_probability')
product_reorder_probability = product_reorder_probability.reset_index()
product_reorder_probability.head()


product = product.merge(product_reorder_probability, on='product_id', how='left')
del product_reorder_probability
gc.collect()
product.head()


### 2.2.4. Number of orders per aisle and department


product_aisle_department = op[['product_id']].merge(products[['product_id', 'aisle_id', 'department_id']], on='product_id', how='left')
product_aisle_department.head()


product_aisle_count = product_aisle_department.groupby('aisle_id')['product_id'].count().to_frame('product_aisle_count')
product_aisle_count = product_aisle_count.reset_index()
product_aisle_count.head()


product_department_count = product_aisle_department.groupby('department_id')['product_id'].count().to_frame('product_department_count')
product_department_count = product_department_count.reset_index()
product_department_count.head()


products = products.merge(product_aisle_count, on = 'aisle_id', how = 'left')
products = products.merge(product_department_count, on = 'department_id', how = 'left')
products.head()


product = product.merge(products[['product_id', 'product_aisle_count', 'product_department_count']], on='product_id', how='left')
del product_aisle_department, product_aisle_count, product_department_count
gc.collect()
product.head()


### 2.2.5 Product one shot ratio


item = op.groupby(['product_id', 'user_id'])['order_id'].count().to_frame('total')
item = item.reset_index(1)
item.head()

item_one = item[item.total==1]
item_one = item_one.groupby('product_id')['total'].count().to_frame('product_one_shot_customers')
item_one.head()


item_size = item.groupby('product_id')['user_id'].count().to_frame('product_unique_customers')
item_size.head()


results = pd.merge(item_one, item_size, on='product_id', how='right')
results.head()


results['product_one_shot_ratio'] = results['product_one_shot_customers']/results['product_unique_customers']
results.head()


product = product.merge(results, on='product_id', how='left')
del item, item_size, item_one
gc.collect()
product.head()


### 2.2.6 Aisles' and departments' mean one-shot ratio


results = results.merge(products[['product_id', 'aisle_id', 'department_id']], on = ['product_id'], how='left')
results.head()


product_aisle_mean_one_shot_ratio = results.groupby('aisle_id')['product_one_shot_ratio'].mean().to_frame('product_aisle_mean_one_shot_ratio')
product_aisle_mean_one_shot_ratio.head()


product_department_mean_one_shot_ratio = results.groupby('department_id')['product_one_shot_ratio'].mean().to_frame('product_department_mean_one_shot_ratio')
product_department_mean_one_shot_ratio.head()


products = products.merge(product_aisle_mean_one_shot_ratio, on = 'aisle_id', how = 'left')
products = products.merge(product_department_mean_one_shot_ratio, on = 'department_id', how = 'left')
products.head()


product = product.merge(products[['product_id', 'product_aisle_mean_one_shot_ratio', 'product_department_mean_one_shot_ratio']], on='product_id', how='left')
del results, product_aisle_mean_one_shot_ratio, product_department_mean_one_shot_ratio
gc.collect()
product.head()


## 2.3. User - product predictors

### 2.3.1. Number of orders per user and product


user_product = op.groupby(['user_id', 'product_id'])['order_id'].count().to_frame('user_product_orders')
user_product = user_product.reset_index()
user_product.head()


### 2.3.2. Number of product orders in the last 5 user orders


op['order_number_back'] = op.groupby('user_id')['order_number'].transform(max) - op.order_number + 1 
op.head()


user_product_last_5_orders = op[op.order_number_back <= 5].groupby(['user_id','product_id'])['order_id'].count().to_frame('user_product_last_5_orders')
user_product_last_5_orders = user_product_last_5_orders.reset_index()
user_product_last_5_orders.head()


### 2.3.3. Ratio of product orders in the last 5 user orders


user_product_last_5_orders['user_product_last_5_orders_ratio'] = user_product_last_5_orders.user_product_last_5_orders / 5
user_product_last_5_orders.head()


user_product = user_product.merge(user_product_last_5_orders, on = ['user_id', 'product_id'], how = 'left')
del user_product_last_5_orders
gc.collect()
user_product.head()


### 2.3.4. Max days that a user has gone without buying a product in the last 5 orders


user_product_last_5_orders_max_days = op[op.order_number_back <= 6].groupby(['user_id', 'product_id'])['days_since_prior_order'].max().to_frame('user_product_last_5_orders_max_days')
user_product_last_5_orders_max_days = user_product_last_5_orders_max_days.reset_index()
user_product_last_5_orders_max_days.head()


user_product = user_product.merge(user_product_last_5_orders_max_days, on = ['user_id', 'product_id'], how = 'left')
del user_product_last_5_orders_max_days
gc.collect()
user_product.head()


### 2.3.5. Median days that a user has gone without buying a product in the last 5 orders


user_product_last_5_orders_median_days = op[op.order_number_back <= 6].groupby(['user_id', 'product_id'])['days_since_prior_order'].median().to_frame('user_product_last_5_orders_median_days')
user_product_last_5_orders_median_days = user_product_last_5_orders_median_days.reset_index()
user_product_last_5_orders_median_days.head()


user_product = user_product.merge(user_product_last_5_orders_median_days, on = ['user_id', 'product_id'], how = 'left')
del user_product_last_5_orders_median_days
gc.collect()
user_product.head()


### 2.3.6. Median hour of day that a user orders a product in the last 5 orders


user_product_last_5_orders_median_hour = op[op.order_number_back <= 6].groupby(['user_id', 'product_id'])['order_hour_of_day'].median().to_frame('user_product_last_5_orders_median_hour')
user_product_last_5_orders_median_hour = user_product_last_5_orders_median_hour.reset_index()
user_product_last_5_orders_median_hour.head()


user_product = user_product.merge(user_product_last_5_orders_median_hour, on = ['user_id', 'product_id'], how = 'left')
del user_product_last_5_orders_median_hour
gc.collect()
user_product.head()


### 2.3.7. Frequency of a user ordering a product after he first purchased it


user_product_orders = op.groupby(['user_id', 'product_id'])['order_id'].count().to_frame('user_product_orders')
user_product_orders = user_product_orders.reset_index()
user_product_orders.head()


user_orders = op.groupby('user_id')['order_number'].max().to_frame('user_orders')
user_orders = user_orders.reset_index()
user_orders.head()


user_product_first_order = op.groupby(['user_id', 'product_id'])['order_number'].min().to_frame('user_product_first_order')
user_product_first_order  = user_product_first_order.reset_index()
user_product_first_order.head()


span = pd.merge(user_orders, user_product_first_order, on='user_id', how='right')
span.head()


span['span'] = span.user_orders - span.user_product_first_order + 1
span.head()


user_product_frequency_after_1st = pd.merge(user_product_orders, span, on=['user_id', 'product_id'], how='left')
user_product_frequency_after_1st.head()


del user_orders, user_product_first_order, span
gc.collect()


user_product_frequency_after_1st['user_product_frequency_after_1st'] = user_product_frequency_after_1st.user_product_orders / user_product_frequency_after_1st.span
user_product_frequency_after_1st.head()


user_product_frequency_after_1st = user_product_frequency_after_1st[['user_id', 'product_id', 'user_product_frequency_after_1st']]
user_product_frequency_after_1st.head()


user_product = user_product.merge(user_product_frequency_after_1st, on=['user_id', 'product_id'], how='left')
del user_product_frequency_after_1st
gc.collect()
user_product.head()


### 2.3.8. Average product position per user


user_product_average_position = op.groupby(['user_id', 'product_id'])['add_to_cart_order'].mean().to_frame('user_product_average_position')
user_product_average_position = user_product_average_position.reset_index()
user_product_average_position.head()


user_product = user_product.merge(user_product_average_position, on=['user_id', 'product_id'], how='left')
del user_product_average_position
gc.collect()
user_product.head()


### 2.3.9. Orders since a user last ordered a product


user_product_orders_since_last_order = op.groupby(['user_id', 'product_id'])['order_number_back'].min().to_frame('user_product_orders_since_last_order')
user_product_orders_since_last_order = user_product_orders_since_last_order.reset_index()
user_product_orders_since_last_order.head()


user_product = user_product.merge(user_product_orders_since_last_order, on=['user_id', 'product_id'], how='left')
del user_product_orders_since_last_order
gc.collect()
user_product.head()


### 2.3.10. Orders since a user last ordered a product divided by the average orders between reorders of that product from the user


user_product['user_product_orders_since_last_order_div_mean_orders_between_purchases'] = user_product.user_product_orders_since_last_order * user_product.user_product_frequency_after_1st
user_product.head()


### 2.3.11. Days since a user last ordered a product


cumulative_days_since_prior_order = op.groupby(['user_id', 'order_number_back'])[['days_since_prior_order']].min()
cumulative_days_since_prior_order = cumulative_days_since_prior_order.fillna(0)
cumulative_days_since_prior_order = cumulative_days_since_prior_order.reset_index()
cumulative_days_since_prior_order.head()


cumulative_days_since_prior_order['cumulative_days_since_prior_order'] = cumulative_days_since_prior_order.groupby('user_id')['days_since_prior_order'].transform(pd.Series.cumsum)
cumulative_days_since_prior_order.head(11)


cumulative_days_since_prior_order = cumulative_days_since_prior_order.merge(user[['user_id', 'user_days_since_last_order']], on = 'user_id')
cumulative_days_since_prior_order.head()


cumulative_days_since_prior_order['cumulative_days_since_prior_order'] = cumulative_days_since_prior_order.cumulative_days_since_prior_order + cumulative_days_since_prior_order.user_days_since_last_order
cumulative_days_since_prior_order.head()


op = op.merge(cumulative_days_since_prior_order[['user_id', 'order_number_back', 'cumulative_days_since_prior_order']], on = ['user_id', 'order_number_back'], how = 'left')
op.head()


user_product_days_since_last_order = op.groupby(['user_id', 'product_id'])['cumulative_days_since_prior_order'].min().to_frame('user_product_days_since_last_order')
user_product_days_since_last_order = user_product_days_since_last_order.reset_index()
user_product_days_since_last_order.head()


user_product = user_product.merge(user_product_days_since_last_order, on=['user_id', 'product_id'], how='left')
user_product.head()


### 2.3.12. Average days between reorders of a product from a user


user_product_mean_days_between_orders = op.groupby(['user_id', 'product_id'])['order_number_back'].max().to_frame('orders_since_first_order')
user_product_mean_days_between_orders = user_product_mean_days_between_orders.reset_index()
user_product_mean_days_between_orders.head()


user_product_mean_days_between_orders = user_product_mean_days_between_orders.merge(cumulative_days_since_prior_order[['user_id', 'order_number_back', 'cumulative_days_since_prior_order']], left_on = ['user_id', 'orders_since_first_order'], right_on = ['user_id', 'order_number_back'])
user_product_mean_days_between_orders.rename(columns={'cumulative_days_since_prior_order':'days_since_first_order'}, inplace=True)
user_product_mean_days_between_orders.head()


user_product_mean_days_between_orders = user_product_mean_days_between_orders.merge(user_product[['user_id', 'product_id', 'user_product_orders']], on = ['user_id', 'product_id'], how = 'left')
user_product_mean_days_between_orders.head()


user_product_mean_days_between_orders['user_product_mean_days_between_orders'] = user_product_mean_days_between_orders.days_since_first_order / user_product_mean_days_between_orders.user_product_orders
user_product_mean_days_between_orders.head()


user_product = user_product.merge(user_product_mean_days_between_orders[['user_id', 'product_id', 'user_product_mean_days_between_orders']], on=['user_id', 'product_id'], how='left')
user_product.head()


### 2.3.13. Days since a user last ordered a product divided by the average days between reorders of that product from the user


user_product['user_product_days_since_last_order_div_mean_days_between_purchases'] = user_product.user_product_days_since_last_order / user_product.user_product_mean_days_between_orders
del cumulative_days_since_prior_order, user_product_days_since_last_order, user_product_mean_days_between_orders
gc.collect()
user_product.head()


### 2.3.14 Proportion of orders of a user that include a product


user_product_proportion_of_orders_with_product = user_product[['user_id', 'product_id', 'user_product_orders']]
user_product_proportion_of_orders_with_product.head()


user_product_proportion_of_orders_with_product = user_product_proportion_of_orders_with_product.merge(user[['user_id', 'user_orders']], on = 'user_id', how = 'left')
user_product_proportion_of_orders_with_product.head()


user_product_proportion_of_orders_with_product['user_product_proportion_of_orders_with_product'] = user_product_proportion_of_orders_with_product.user_product_orders / user_product_proportion_of_orders_with_product.user_orders
user_product_proportion_of_orders_with_product.head()


user_product = user_product.merge(user_product_proportion_of_orders_with_product[['user_id', 'product_id', 'user_product_proportion_of_orders_with_product']], on=['user_id', 'product_id'], how='left')
del user_product_proportion_of_orders_with_product
gc.collect()
user_product.head()


## 2.4. Merge features

### 2.4.1. Merge with user


data = user_product.merge(user, on='user_id', how='left')
data.head()


### 2.4.2. Merge with product


data = data.merge(product, on='product_id', how='left')
data = data.fillna(0)
data.head()


### 2.4.3. Delete unused DataFrames


del op, user, product, user_product
gc.collect()


# 3. Create train and test DataFrames

## 3.1. Include information about the last order of each user


orders_future = orders[((orders.eval_set=='train') | (orders.eval_set=='test'))]
orders_future = orders_future[ ['user_id', 'eval_set', 'order_id'] ]
orders_future.head()


data = data.merge(orders_future, on='user_id', how='left')
data.head()


del orders_future
gc.collect()


## 3.2. Prepare the train DataFrame


data_train = data[data.eval_set=='train']
data_train.head()


data_train = data_train.merge(order_products_train[['product_id','order_id', 'reordered']], on=['product_id','order_id'], how='left' )
data_train.head()


del order_products_train
gc.collect()


data_train['reordered'] = data_train['reordered'].fillna(0)
data_train = data_train.set_index(['user_id', 'product_id'])
data_train = data_train.drop(['eval_set', 'order_id'], axis=1)
data_train.head()


## 3.3. Prepare the test DataFrame


data_test = data[data.eval_set=='test']
data_test = data_test.set_index(['user_id', 'product_id'])
data_test = data_test.drop(['eval_set','order_id'], axis=1)
data_test.head()


del data
gc.collect()


# 4. Create predictive model


#{'max_depth' : [5, 6], 'subsample' : [0.8, 0.9]}
#{'subsample': 0.8, 'max_depth': 6}
#0.9103662081586509

#{'max_depth' : [6], 'subsample' : [0.6, 0.7, 0.8], 'colsample_bytree' : [0.6, 0.7, 0.8]}
#{'subsample': 0.7, 'colsample_bytree': 0.7, 'max_depth': 6}
#0.9103629041916839

#{'max_depth' : [6], 'subsample' : [0.7], 'colsample_bytree' : [0.7], 'lambda' : [0.8, 0.9, 1.0], 'gamma' : [0.0, 0.1, 0.2]}
#{'subsample': 0.7, 'colsample_bytree': 0.7, 'max_depth': 6, 'gamma': 0.2, 'lambda': 0.9}
#0.9103699841208988

#{'max_depth' : [6, 7, 8], 'subsample' : [0.7], 'colsample_bytree' : [0.7], 'lambda' : [0.9], 'gamma' : [0.2, 0.3, 0.4]}
#{'subsample': 0.7, 'colsample_bytree': 0.7, 'max_depth': 8, 'gamma': 0.3, 'lambda': 0.9}
#0.9105283385376713

#{'max_depth' : [8, 9, 10], 'subsample' : [0.6, 0.7, 0.8], 'colsample_bytree' : [0.7], 'lambda' : [0.9], 'gamma' : [0.3]}
#{'subsample': 0.7, 'colsample_bytree': 0.7, 'max_depth': 8, 'gamma': 0.3, 'lambda': 0.9}
#0.9105500503205969

#{'max_depth' : [8], 'subsample' : [0.7], 'colsample_bytree' : [0.6, 0.7, 0.8], 'lambda' : [0.8, 0.9, 1.0], 'gamma' : [0.3]}
#{'subsample': 0.7, 'colsample_bytree': 0.8, 'max_depth': 8, 'gamma': 0.3, 'lambda': 0.9})
#0.9105413184078985

#{'max_depth' : [8], 'subsample' : [0.7], 'colsample_bytree' : [0.8, 0.9], 'lambda' : [0.9], 'gamma' : [0.2, 0.3, 0.4]}
#{'subsample': 0.7, 'colsample_bytree': 0.8, 'max_depth': 8, 'gamma': 0.2, 'lambda': 0.9}
#0.9105284565364915


## 4.1. Tune model with Grid Search

# learning_rate = 0.1 n_estimators = 140
#{'max_depth' : [7, 6], 'min_child_weight' : [2, 3, 4]}
#('The best params are: ', {'max_depth': 7, 'min_child_weight': 3})
#('The achieved score with these params is: ', 0.9105473363477312)

# {'max_depth' : [7, 8, 9]}
# 'The best params are: ', {'max_depth': 9})
# ('The achieved score with these params is: ', 0.9105845059761093)

#{'gamma' : [0.1, 0.2, 0.3], 'lambda' : [0.8, 0.9, 1] }
#'The best params are: ', {'gamma': 0.3, 'lambda': 1})
#('The achieved score with these params is: ', 0.9105896979242002)
#('The best params are: ', {'gamma': 0.1, 'lambda': 0.9})
#('The achieved score with these params is: ', 0.9105918219029646)
# ('The best params are: ', {'gamma': 0.2, 'lambda': 0.8})
#('The achieved score with these params is: ', 0.9105989018321795)
#('The best params are: ', {'gamma': 0.2, 'lambda': 1})
#('The achieved score with these params is: ', 0.9105950078711113)

# {'gamma' : [0.5, 0.4, 0.3, 0.2, 0.1], 'lambda' : [1] 
# ('The best params are: ', {'gamma': 0.3, 'lambda': 1})
# ('The achieved score with these params is: ', 0.9105932378888076)

# {'subsample' : [0.6, 0.7, 0.8, 0.9], 'colsample_bytree' : [0.6, 0.7, 0.8, 0.9]}
#('The best params are: ', {'subsample': 0.8, 'colsample_bytree': 0.7})
#('The achieved score with these params is: ', 0.9106162476587559)

#  'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100] }
#('The best params are: ', {'reg_alpha': 1})
#('The achieved score with these params is: ', 0.9106130616906092)

# {'learning_rate' :[0.1, 0.01, 0.001, 0.0001], 'n_estimators' : [100, 200, 300, 500]}
#'The best params are: ', {'n_estimators': 500, 'learning_rate': 0.1})
#('The achieved score with these params is: ', 0.910288210938467)

# {'learning_rate' : [0.1, 0.01, 0.001, 0.0001], 'n_estimators' : [100, 500, 1000, 5000]}
#('The best params are: ', {'n_estimators': 5000, 'learning_rate': 0.1})
#('The achieved score with these params is: ', 0.9105948898722911)


param_grid =  { 'reg_alpha':[0.1 ,0.5, 1, 3, 16] }
    
xg = xgb.XGBClassifier( learning_rate= 0.1,
                        n_estimators= 160,
                        eval_metric= 'logloss',
                        subsample= 0.8,
                        min_child_weight= 3,
                        max_depth= 9,
                        gamma=0.3 ,
                        colsample_bytree=0.7,
                        nthread=4,
                        scale_pos_weight=1,
                        seed=27, 
                        tree_method = 'gpu_hist', 
                        num_boost_round = 10)

grid_search = GridSearchCV(estimator = xg, param_grid = param_grid, cv = 4, verbose = 2, n_jobs = 2)


xg.get_params()


grid_search.fit(data_train.drop('reordered', axis=1), data_train.reordered)

print('The best params are: ', grid_search.best_params_)
print('The achieved score with these params is: ', grid_search.best_score_)


## 4.2. Train model


#dm_train = xgb.DMatrix(data = data_train.drop('reordered', axis=1), label = data_train.reordered)
#dm_test = xgb.DMatrix(data = data_test)


#params = {'objective' : 'binary:logistic',
 #         'tree_method' : 'gpu_hist',
 #         'eval_metric' : 'logloss',
 #         'subsample': 0.8,
 #         'colsample_bytree': 0.7,
 #         'max_depth': 9,
 #         'min_child_weight' : 3,
 #        'gamma': 0.3,
 #        'lambda': 1}

#xg = xgb.train(dtrain = dm_train, params = params, num_boost_round = 10)


#xgb.plot_importance(xg)
#plt.show()


## 4.3. Make predictions


#test_pred = (xg.predict(dm_test) >= 0.21)
#test_pred[0:20]


# 5. Prepare submission file


#data_test['prediction'] = test_pred
#data_test.head()


#final = data_test.reset_index()
#final = final[['product_id', 'user_id', 'prediction']]
#gc.collect()
#final.head()


#orders_test = orders.loc[orders.eval_set=='test',("user_id", "order_id") ]
#orders_test.head()


#final = final.merge(orders_test, on='user_id', how='left')
#final.head()


#final = final.drop('user_id', axis=1)
#final['product_id'] = final.product_id.astype(int)

#del orders, test_pred
#del orders_test, data_test
#gc.collect()

#final.head()


#d = dict()
#for row in final.itertuples():
 #   if row.prediction== 1:
 #       try:
 #           d[row.order_id] += ' ' + str(row.product_id)
 #       except:
 #           d[row.order_id] = str(row.product_id)

#for order in final.order_id:
#    if order not in d:
#        d[order] = 'None'
#        
#gc.collect()


#sub = pd.DataFrame.from_dict(d, orient='index')

#sub.reset_index(inplace=True)
#sub.columns = ['order_id', 'products']

#sub.head()


#sub.shape[0]


#sub.to_csv('sub.csv', index=False)
