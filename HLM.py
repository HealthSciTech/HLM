import random
import numpy as np
import pandas as pd
import matplotlib.ticker
import scipy.stats as stats
import statsmodels.api as sm
from datetime import datetime
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
import seaborn as sns
sns.set_style("white")

def mixedlm_fit_2level(axis, data, method):
                        
    model = sm.MixedLM.from_formula(axis, data, re_formula="week", groups='User_ID')
    results = model.fit(method=method) # could be one of powell, lbfgs, cg or bfgs. Try the one which converges.
    print(results.summary())
    print(results.resid.values)
    fig = sm.qqplot(results.resid.values, stats.t, fit=True, line='45')
    return model, results


def combined_plot_HLM(before_df, after_df, 
                      before_x_min, before_x_max, after_x_min, after_x_max,
                      y_min, y_max, x_feature, y_feature, IDs,
                      before_slope, before_intercept, after_slope, after_intercept, name=None):
    fig, axs = plt.subplots(ncols=2, figsize=(12, 5), gridspec_kw={'width_ratios': [1.5, 1.5]})
    sns.scatterplot(x=x_feature, y=y_feature,
                    data=before_df, alpha=0.1, palette='Paired', legend='full', x_jitter=0.5, ax=axs[0])
    g = sns.scatterplot(x=x_feature, y=y_feature,
                    data=after_df, alpha=0.1, palette='Paired', legend='full', x_jitter=0.5, ax=axs[1])
    
    for idx, _id in enumerate(IDs):
        c=(random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
        axs[0].plot(before_df[before_df['User_ID'] == _id].week, 
                 before_df[before_df['User_ID'] == _id].ActivityEstimate,
                 c=c, label=str(idx + 1), linewidth=1, alpha=1)
        axs[1].plot(after_df[after_df['User_ID'] == _id].week, 
                 after_df[after_df['User_ID'] == _id].ActivityEstimate,
                 c=c, label=str(idx + 1), linewidth=1, alpha=1)
        
    x_vals = np.array(axs[0].get_xlim())
    y_vals = before_intercept + before_slope * x_vals
    axs[0].plot(x_vals, y_vals, '-', c=(0.0, 0.0, 0.0), alpha=1, linewidth=4, label='Total')
    x_vals = np.array(axs[1].get_xlim())
    y_vals = after_intercept + after_slope * x_vals
    axs[1].plot(x_vals, y_vals, '-', c=(0.0, 0.0, 0.0), alpha=1, linewidth=4, label='Total')
    
    axs[0].set(xlim=(before_x_min, before_x_max), ylim=(y_min, y_max))
    axs[1].set(xlim=(after_x_min, after_x_max), ylim=(y_min, y_max))
    
    g.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), ncol=2)
    
    locator = matplotlib.ticker.MultipleLocator(5)
    axs[0].xaxis.set_major_locator(locator)
    formatter = matplotlib.ticker.StrMethodFormatter("{x:.0f}")
    axs[0].xaxis.set_major_formatter(formatter)
    
    locator = matplotlib.ticker.MultipleLocator(5)
    axs[1].xaxis.set_major_locator(locator)
    formatter = matplotlib.ticker.StrMethodFormatter("{x:.0f}")
    axs[1].xaxis.set_major_formatter(formatter)
        
    #axs[0].set_xlabel('day')
    #axs[1].set_xlabel('day')

    if name == None:
        name = y_feature
    #plt.title(y_feature)
    plt.savefig("./results/HLM_combined-"+str(name)+".png", bbox_inches='tight')


# example for TST
agg_df_before = pd.read_csv('agg_sleep_before.csv')
agg_df_after = pd.read_csv('agg_sleep_after.csv')

agg_df_before['time'] = 0
agg_df_after['time'] = 1

# necessary to start from zero for both groups when you want to report stats
agg_df_before['week'] = agg_df_before['week'] - agg_df_before['week'].min()
agg_df_after['week'] = agg_df_after['week'] - agg_df_after['week'].min()
agg_df_before = agg_df_before[agg_df_before['week'] < 28]
agg_df_after = agg_df_after[agg_df_after['week'] < 28]

total = pd.concat([agg_df_before, agg_df_after])

# outlier removal
z = np.abs(stats.zscore(total['TST']))
outlier_idx = list(np.where(z > 3)[0])
total_temp = total.drop(total.index[outlier_idx]).copy()

total_temp.rename(columns = {'subjectID':'User_ID'}, inplace = True) 
total_temp = total_temp.reset_index()

# learning and prediction
model1, results1 = mixedlm_fit_2level("TST ~ week + time*week", total_temp, "powell")
total_temp['ActivityEstimate'] = results1.fittedvalues

# spagetti charts
combined_plot_HLM(total_temp[total_temp['time']==0].copy(), total_temp[total_temp['time']==1].copy(), 
                      total_temp[total_temp['time']==0]['week'].min(), total_temp[total_temp['time']==0]['week'].max(),
                      total_temp[total_temp['time']==1]['week'].min(), total_temp[total_temp['time']==1]['week'].max(),
                      300, 700, 'week', 'TST', IDs,
                      results1.params['week'], results1.params['Intercept'],
                      results1.params['week']+results1.params['time:week'], results1.params['Intercept']+results1.params['time'], 'TST-2level')


'''
Interpreting the results:

User ID var: var of Intercept random effect 
Week Var: var of slope random effect

Before Covid: TST = Intercept + week * (x=week) //// time = 0
After Covid: TST = Intercept + time + (week + time:week) * (x=week) //// time = 1
'''