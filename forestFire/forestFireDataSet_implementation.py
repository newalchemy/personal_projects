# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 21:19:46 2020

@author: timhe
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import math
import statistics
import copy

from sklearn import preprocessing

from sklearn.svm import SVR

import xgboost as xgb;

from sklearn.model_selection import KFold
from scipy.integrate import simps

from pathlib import Path

# Paths based on user setup.  If this was going to be used more frequently, I would have put
# these, as well as any other parameters, into a separate config file and written a parser
# which reads those parameters and assigns them.
save_loc = 'C:/Users/timhe/OneDrive/Documents/StatsWorkFolder/CS5526/Plots/';
file_location = 'C:/Users/timhe/Temple Coursework/TempleData/ForestFire/forestfires.csv';


# Transformation functions for response variable.
def area_transform_fn(area_val):
    area_out = np.log(area_val + 1);
    return area_out;

def invert_area_transform_fn(area_val_proc):
    outval = np.e ** (area_val_proc) - 1;
    return outval;

def area_transform_fn_log2(area_val):
    area_out = np.log2(area_val + 1);
    return area_out;

def identity(area_val):
    return area_val;

def invert_area_transform_fn_log2(area_val_proc):
    outval = 2 ** (area_val_proc) - 1;
    return outval;

def area_transform_fn_log2_sqrt(area_val):
    area_out = np.sqrt(np.log2(area_val + 1));
    return area_out;

def invert_area_transform_fn_log2_sqrt(area_val_proc):
    outval = 2 ** (area_val_proc**2) - 1;
    return outval;


    
# Tool for plotting REC Curve, based on the implementation here:
# https://amirhessam88.github.io/regression-error-characteristics-curve/
def REC(y_true , y_pred):
    
    # initilizing the lists
    Accuracy = []
    
    # initializing the values for Epsilon
    Begin_Range = 0
    End_Range = 15
    Interval_Size = 1
    
    # List of epsilons
    Epsilon = np.arange(Begin_Range , End_Range , Interval_Size)
    
    # Main Loops
    for i in range(len(Epsilon)):
        count = 0.0
        for j in range(len(y_true)):
            if np.abs(y_true[j] - y_pred[j]) < Epsilon[i]:
                count = count + 1
        
        Accuracy.append(count/len(y_true))
    
    # Calculating Area Under Curve using Simpson's rule
    AUC = simps(Accuracy , Epsilon ) / End_Range;
        
    # returning epsilon , accuracy , area under curve    
    return Epsilon , Accuracy , AUC
    

# Area noise estimation for sigma, for the SVM based on the results in the paper which originally published this.
def area_noise_estimation(y_areas, df):
    y_hat = statistics.mean(y_areas);
    mss = statistics.mean((y_areas - y_hat)**2);
    n = len(y_areas);
    return mss*n/(n - df);

# Generate plots used for measuring the performance of the model.
def plot_performance(y_pred, y_actual, y_pred_orig, tech_name, tech_name_legend, save_loc):
    Epsilon, Accuracy, AUC = REC(y_actual, y_pred);
    inxes = [i for i in range(num_samples)];

    f1 = plt.figure();
    plt.title("Regression Error Characteristic (REC) for {}".format(tech_name))
    plt.plot(Epsilon, Accuracy, "--b",lw =3, label=tech_name_legend)
    plt.xlabel("Absolute Error")
    plt.ylabel("Accuracy (%)")
    plt.text(1.1, 0.07, "AUC = %0.4f" %AUC , fontsize=15);
    plt.legend();
    plt.grid(True);
    file_name = '{}/{}{}'.format(save_loc, tech_name, 'RECcurve');
    f1.savefig(fname=file_name);
    
    f2 = plt.figure();
    
    lower_actuals = [y for y in y_actual if y < 20];
    lower_preds = [y_pred[i] for i, y in enumerate(y_actual) if y < 20];
    lower_inxes = [i for i in range(len(lower_actuals))];

    middle_actuals = [y for y in y_actual if y >= 20 and y < 150];
    middle_preds = [y_pred[i] for i, y in enumerate(y_actual) if y >= 20 and y < 150];
    middle_inxes = [len(lower_inxes) + 1 + i for i in range(len(middle_actuals))];

    upper_actuals = [y for y in y_actual if y >= 150];
    upper_preds = [y_pred[i] for i, y in enumerate(y_actual) if y >= 150];
    upper_inxes = [len(lower_inxes) + len(middle_inxes) + 1 + i for i in range(len(upper_actuals))];

    
    plt.scatter(lower_inxes, lower_actuals, c='#1f77b4', label='Actual Fire Area Burned');
    plt.scatter(lower_inxes, lower_preds, c='#e377c2', label=tech_name_legend);
    plt.xlabel("Ordered Test Set")
    plt.ylabel("Burned Area (hA)")
    plt.title('Ordered Test Set Plot for {} - ha less than 20'.format(tech_name));
    plt.legend();
    plt.grid(True);
    file_name = '{}/{}{}{}'.format(save_loc, tech_name, 'OrderedTestSet', 'Lower');
    f2.savefig(fname=file_name);

    f3 = plt.figure();    
    plt.scatter(middle_inxes, middle_actuals, c='#1f77b4', label='Actual Fire Area Burned');
    plt.scatter(middle_inxes, middle_preds, c='#e377c2', label=tech_name_legend);
    plt.xlabel("Ordered Test Set")
    plt.ylabel("Burned Area (hA)")
    plt.title('Ordered Test Set Plot for {} - ha between 20 and 150'.format(tech_name));
    plt.legend();
    plt.grid(True);
    file_name = '{}/{}{}{}'.format(save_loc, tech_name, 'OrderedTestSet', 'Middle');
    f3.savefig(fname=file_name);

    f4 = plt.figure();    
    plt.scatter(upper_inxes, upper_actuals, c='#1f77b4', label='Actual Fire Area Burned');
    plt.scatter(upper_inxes, upper_preds, c='#e377c2', label=tech_name_legend);
    plt.xlabel("Ordered Test Set")
    plt.ylabel("Burned Area (hA)")
    plt.title('Ordered Test Set Plot for {} - ha more than 150'.format(tech_name));
    plt.legend();
    plt.grid(True);
    file_name = '{}/{}{}{}'.format(save_loc, tech_name, 'OrderedTestSet', 'Upper');
    f4.savefig(fname=file_name);
    
    act_max = max(y_pred_orig);
    
    biggest_bin = 0.0;
    bin_size = 0.25;
    while (biggest_bin <= act_max): biggest_bin += bin_size;
    biggest_inx = int(biggest_bin/bin_size);
    
    bin_bounds = [i*0.25 for i in range(0, biggest_inx)];
    
    f6 = plt.figure();
    plt.hist(y_pred_orig, facecolor='#1f77b4');
    plt.xlabel("Burned Area (Transformed)")
    plt.ylabel("Count")
    plt.title('Transformed output distribution for technique {}'.format(tech_name));
    plt.grid(True);
    file_name = '{}/{}_{}'.format(save_loc, tech_name, 'ResponseTransDistribution');
    f6.savefig(fname=file_name);
    
    f7 = plt.figure();
    plt.hist(y_pred, facecolor='#1f77b4', bins=bin_bounds);
    plt.xlabel("Burned Area (hA)")
    plt.ylabel("Count")
    plt.title('Untransformed output distribution for technique {}'.format(tech_name));
    plt.grid(True);
    file_name = '{}/{}_{}'.format(save_loc, tech_name, 'ResponseOrigDistribution');
    f7.savefig(fname=file_name);


    # Writeout key statistics:  MAD and RMSE
    
    diffs = [y_actual[i] - y_pred[i] for i in range(len(y_pred))];
    abs_diffs = np.abs(diffs);
    MAD = (1/len(abs_diffs)) * sum(abs_diffs);
    
    RMSE = np.sqrt(sum(np.square(diffs)) / len(diffs));
    
    return AUC, MAD, RMSE;

# Can be used to evaluate any SKLearn regressor (Note:  I originally was going to experiment with several more sklearn regressors)
def evalute_regressor(regressor, input_data_matrix, area_vals_transformed, area_inv, save_loc, tech_name, tech_name_legend):
    kf = KFold(n_splits=10, shuffle=True);
    num_samples = len(input_data_matrix[:,0]);
    inxes = [i for i in range(num_samples)];

    splits = kf.split(inxes);

    test_out = pd.DataFrame(columns=['y_pred', 'y_actual']);

    for train_inxes, test_inxes in splits:
        train_samples = input_data_matrix[train_inxes,:];
        test_samples = input_data_matrix[test_inxes,:];
        train_areas = area_vals_transformed[train_inxes];
        test_areas = area_vals_transformed[test_inxes];
    
        regressor.fit(train_samples, train_areas);
        y_pred = regressor.predict(test_samples);
        temp_df = pd.DataFrame();
        temp_df.insert(0, 'y_pred', y_pred);
        temp_df.insert(1, 'y_actual', np.array(test_areas));
        temp_df.insert(2, 'y_pred_orig', y_pred);
        test_out = pd.concat([test_out, temp_df]);

    # area_vals = df.area.transform(areaTransformFn);
    test_out.y_pred = test_out.y_pred.transform(area_inv);
    test_out.y_actual = test_out.y_actual.transform(area_inv);

    test_out = test_out.sort_values(by='y_actual', ascending=True);

    # def plot_performance(y_pred, y_actual, tech_name, tech_name_legend, save_loc):
    AUC, MAD, RMSE = plot_performance(test_out.y_pred.to_list(), test_out.y_actual.to_list(), test_out.y_pred_orig.to_list(),tech_name, tech_name_legend, save_loc);
    return AUC, MAD, RMSE;
    
        
centerer = preprocessing.StandardScaler(with_std=True);

df_orig = pd.read_csv(file_location);
df_fm = df_orig.copy();

# The authors of the paper believed that the temporal and spatial data for the fires was 
# useless for prediction, so we won't be using those variables here.
uniq_months = df_fm.month.unique();
df_fm = df_fm.drop(labels='day', axis=1);
df_fm = df_fm.drop(labels='X', axis=1);
df_fm = df_fm.drop(labels='Y', axis=1);
df_fm = df_fm.drop(labels='month', axis=1);

# Smooth these predictors to get rid of the skew.
df_fm['ISI'] = np.log1p(df_fm['ISI']);
df_fm['FFMC'] = np.log1p(df_fm['FFMC']);

num_samples = df_fm.shape[0];

# Inverting some of the variables here, just so that way 
# all the variables are "pointing" in the same direction, ie, as variables
# get larger, fires should get larger too.

# Relative Humidity(RH) will be replaced with relative dryness (RD)
# and computed as 100 - RH, and rain will be replaced by 
# "anti_rain", which is 6.4 - rain (since 6.4 is the most amount of rain
# for a given sample in this dataset)

# This will make it easier to interpret cross and quadratic terms later on.
relative_dryness = 100 - df_fm.RH;
anti_rain = 6.4 - df_fm.rain;
df_fm.insert(8, "RD", relative_dryness);
df_fm.insert(9, "antiRain", anti_rain);

df_fm = df_fm.drop(labels='RH', axis=1);
df_fm = df_fm.drop(labels='rain', axis=1);

# Drop the response variable.
area_orig = df_fm.area;
df_fm = df_fm.drop(labels='area', axis=1);

input_matricies = [];

scaler = preprocessing.MinMaxScaler();
# Make the original one, rescaled to (0,1) to start.
df_scaled = scaler.fit_transform(df_fm);
input_matricies.append(df_scaled);
# Drop isSummer and add it backwhen we're adding it to the matrix list.
passed_features = list(df_fm.columns);

passed_features_cross = copy.deepcopy(passed_features);
# Now add interaction terms (eg. feat 1 x feat 2 ) for each feature.
# Model all interaction terms.
df_cross = df_scaled.copy();
col_num = len(passed_features) + 1;
for i in range(len(passed_features)):
    name_first = passed_features[i];
    for j in range(i + 1, len(passed_features)):
        name_second = passed_features[j];
        name_feat = '{}_{}'.format(name_first, name_second);
        new_feat = df_scaled[:,i] * df_scaled[:,j];
        df_cross = np.column_stack((df_cross, new_feat));
        passed_features_cross.append(name_feat);
        

input_matricies.append(df_cross);

# In addition to interaction terms, add quadratic terms for each 
passed_features_quad = copy.deepcopy(passed_features_cross);
df_quad = df_cross.copy();
for i in range(len(passed_features)):
    name_first = passed_features[i];
    name_feat = '{}^2'.format(name_first);
    new_feat = df_scaled[:,i] ** 2;
    df_quad = np.column_stack((df_quad, new_feat));
    passed_features_quad.append(name_feat);

input_matricies.append(df_quad);

# Now make the transformations for each response variable.
area_orig = df_orig.area;
area_vals = df_orig.area.transform(area_transform_fn);
area_l2 = df_orig.area.transform(area_transform_fn_log2);
area_l2_sqrt = df_orig.area.transform(area_transform_fn_log2_sqrt);

area_transforms = [area_transform_fn, area_transform_fn_log2, area_transform_fn_log2_sqrt];
inverse_area_transforms = [invert_area_transform_fn, invert_area_transform_fn_log2, invert_area_transform_fn_log2_sqrt];


#  We'll be using 5-fold cross validation.  This part begins the grid search.  

# Start off by using the params in the paper for the SVM to replicate their results.
C = 3
n = len(area_vals);
sigma = area_noise_estimation(area_vals, n/3);
epsilon = 3 * sigma * math.sqrt(np.log(n)/n)
gamm = 1/8;

stats_f = open('{}experiment_stats.txt'.format(save_loc), 'w');

# Write out the original model from the paper as a control/baseline.
centerer = preprocessing.StandardScaler(with_std=True);
svr_orig = SVR(C=C, epsilon=epsilon, gamma=gamm);
svm_save = '{}{}/'.format(save_loc, 'SVM_Original');
tech_name = 'FromPaper';
Path(svm_save).mkdir(parents=True, exist_ok=True);

df_scaled = centerer.fit_transform(df_fm);

AUC, MAD, RMSE  = evalute_regressor(svr_orig, df_scaled, area_vals, invert_area_transform_fn, svm_save, 'SVM_{}'.format(tech_name), 'SVM');
line = 'SVM_Orig stats:  AUC: {} , MAD: {} , RMSE: {} \n'.format(AUC, MAD, RMSE);
print(line);
stats_f.write(line)



# These tupples are for naming files / headers.
input_names = ('original', 'crossOnly', 'quadratic');
fn_names = ('natLog', 'log2', 'log2Sqrt');


for i in range(len(input_matricies)):
    for f in range(len(area_transforms)):
        input_data_matrix = input_matricies[i];
        my_transform = area_transforms[f];
        my_inverse = inverse_area_transforms[f];
        my_areas = my_transform(area_orig);
        
        # Do an SVM for each input type / response variable transformation.
        svr_orig = SVR(C=C, epsilon=epsilon, gamma=gamm);
        svm_save = '{}{}/'.format(save_loc, 'SVM_Model');
        tech_name = '{}_{}'.format(input_names[i], fn_names[f]);
        Path(svm_save).mkdir(parents=True, exist_ok=True);
        
        AUC, MAD, RMSE  = evalute_regressor(svr_orig, input_data_matrix, my_areas, my_inverse, svm_save, 'SVM_{}'.format(tech_name), 'SVM');
        
        line = 'SVM_{} stats:  AUC: {} , MAD: {} , RMSE: {}  \n'.format(tech_name, AUC, MAD, RMSE);
        print(line);
        stats_f.write(line)
        
        
        p_powers = [(1.02 + 0.05*i) for i in range(0,4)];
        #eta_vals = [0.1, 0.2, 0.3, 0.4, 0.5];
        eta_vals = [0.2, 0.4, 0.6, 0.8, 1];
        min_child_weight_vals = [2, 4, 6, 8];
        #min_child_weight will be useful here since 
        
        # Now sweep across the XGBoost possibilities.
        for ptest in p_powers:
            for eta in eta_vals:
                for min_child_weight in min_child_weight_vals:
                    prepl = '{}'.format(ptest).replace('.', '_');
                    etarepel = '{}'.format(eta).replace('.', '_');

                    xgboo_save = '{}xgboost_p{}/eta_{}/min_child_{}'.format(save_loc, prepl, etarepel, min_child_weight);
                    techname = 'p{}_eta{}_min_child{}'.format(prepl, etarepel, min_child_weight);
            
                    Path(xgboo_save).mkdir(parents=True, exist_ok=True);

                    kf = KFold(n_splits=5, shuffle=True);
                    inxes = [i for i in range(num_samples)];

                    splits = kf.split(inxes);
                    test_out = pd.DataFrame(columns=['y_pred', 'y_actual']);
        
                    if (i == 0):
                        names = passed_features;
                    elif (i == 1):
                        names = passed_features_cross;
                    else:
                        names = passed_features_quad;

                    itr = 1;

                    # Run the configuration for each fold in the cross validation.
                    for train_inxes, test_inxes in splits:
            
                        my_areas_train = list(my_areas[train_inxes]);
                        my_areas_test = list(my_areas[test_inxes]);
                        dmat_train = xgb.DMatrix(input_data_matrix[train_inxes,:], my_areas_train, feature_names=names);
                        dmat_test = xgb.DMatrix(input_data_matrix[test_inxes,:], my_areas_test, feature_names=names);

                        # Start with max depth of 8 and possibly do a parameter sweep 
                        #eval_metric  : tweedie-nloglik
                        tweedie_booster = xgb.train({'max_depth': 5, 'eta': eta, 'gamma': 0, 'min_child_weight': min_child_weight, 'early_stopping_rounds': 5, 'objective': 'reg:tweedie','eval_metric' : 'rmse', 'tweedie_variance_power': ptest, 'num_boost_round': 100, 'tree_method':'hist', 'verbose': False, 'nthread':4},
                                                    dmat_train,
                                                    evals=[(dmat_train, "train"), (dmat_test, "test")])
    
                        y_pred_orig = tweedie_booster.predict(dmat_test)                    
                        y_predict = my_inverse(y_pred_orig);
                        y_actual = np.array(area_orig[test_inxes]);
    
                        temp_df = pd.DataFrame();
                        temp_df.insert(0, 'y_pred', y_predict);
                        temp_df.insert(1, 'y_actual', y_actual);
                        temp_df.insert(2, 'y_pred_orig', y_pred_orig);
                        test_out = pd.concat([test_out, temp_df]);
                        itr = itr + 1;

                    test_out = test_out.sort_values(by='y_actual', ascending=True);
                        
                    # Save all performance info across all the test folds.
                    AUC, MAD, RMSE  = plot_performance(test_out.y_pred.to_list(), test_out.y_actual.to_list(), test_out.y_pred_orig.to_list(),'xboost8_{}'.format(tech_name), 'XGBoost', xgboo_save);
                    line = 'Xgboost8_{}_p_{}_eta_{}_min_child_{} stats:,  AUC: {} , MAD: {} , RMSE: {} \n'.format(tech_name, prepl, etarepel, min_child_weight, AUC, MAD, RMSE);
                    print(line);
                    stats_f.write(line)
stats_f.close();
print('done!')