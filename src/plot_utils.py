import matplotlib.pyplot as plt
from scipy.stats import norm as norm_dist
import numpy as np
import pandas as pd

def get_weights(experiment_df):
    last_round = experiment_df['round'].max()
    n_sims = experiment_df['simul_id'].max() + 1

    experiment_df_last_round = experiment_df[(experiment_df['round'] == last_round - 1) & (experiment_df['simul_id'] == n_sims - 1)]
    weights_df = experiment_df_last_round.groupby(['policy'], sort = False).agg({'reward': 'sum',                                                   
                                                    'regret': ['count','sum'],
                                                   'true_weights':'first',
                                                   'sample_weights':'first',
                                                   'mean_weights':'first',
                                                   'precision_weights':'first'})
    return weights_df

def plot_weight_distribution(weights_df):
    # plot the mean and std of the weights
    plt.close('all')
    plt.figure(figsize=(10, 3), dpi=150)
    # get the final true weights
    true_weights = weights_df.iloc[-1]['true_weights']['first']
    policies = weights_df.index.values
    for i in range(len(true_weights)):
        plt.figure(figsize=(10, 3), dpi=150)
        m = true_weights[i]
        sigma = 5
        # loop for each decision policy
        for policy in policies:
            X_pdf = np.linspace(m - sigma, m + sigma, 1000)
            online_lr = weights_df.loc[policy]
            pdf = norm_dist(loc=online_lr['mean_weights']['first'][i], scale=online_lr['precision_weights']['first'][i]**(-1.0)).pdf(X_pdf)
            plt.plot(X_pdf, pdf, label=policy, linewidth=1.5)

        # plot a vertical line at the true weights
        plt.axvline(x=m, color='black', linestyle='--', label='True weights', linewidth=1.5)
        plt.xlabel('Weights')
        plt.ylabel('Density') 
        plt.legend(fontsize=8); plt.xticks(fontsize=10); plt.yticks(fontsize=10)   
    plt.show()

def plot_regret(regret_df, n_sims, regret_type='instant'):
    '''It requires the mean'''
    # closing all past figures
    plt.close('all')

    # opening figure to plot regret
    plt.figure(figsize=(10, 3), dpi=150)

    # loop for each decision policy
    policies = regret_df.index.get_level_values('policy').unique()
    for policy in policies:        
        if regret_type == 'instant':
            plt.plot(regret_df.loc[policy,:].values, label=policy, linewidth=1.5)
        elif regret_type == 'cumulative':
            plt.plot(np.cumsum(regret_df.loc[policy,:].values), label=policy, linewidth=1.5)
        else:
            raise ValueError('regret_type must be either instant or cumulative')
        
    # adding title
    plt.title('Mean comparison of {} regret for each method in {} simulation'.format(regret_type, n_sims), fontsize=10)

    # adding legend
    plt.legend(fontsize=8); plt.xticks(fontsize=10); plt.yticks(fontsize=10)

    # showing plot
    plt.show()

def plot_regret_cl(regret_df, n_sims, regret_type='instant'):
    '''Requires all the regret values'''
    # closing all past figures
    plt.close('all')

    # opening figure to plot regret
    plt.figure(figsize=(10, 3), dpi=150)

    # loop for each decision policy
    policies = regret_df['policy'].unique()
    regret_df['regret_cumsum'] = regret_df.groupby(['policy', 'simul_id'], sort = False)['regret'].cumsum()

    for policy in policies:        
        if regret_type == 'instant':
            plot_col = 'regret'
        elif regret_type == 'cumulative':
            plot_col = 'regret_cumsum'
        else:
            raise ValueError('regret_type must be either instant or cumulative')

        regret_df_mean = regret_df.reset_index().groupby(['policy','round'], sort = False)[plot_col].mean()
        regret_df_std = regret_df.reset_index().groupby(['policy','round'], sort = False)[plot_col].std()
        y = regret_df_mean.loc[policy,:].values
        y_upper = y + regret_df_std.loc[policy,:].values * 1.96
        y_lower = y - regret_df_std.loc[policy,:].values * 1.96

        plt.plot(y, label=policy, linewidth=1.5)
        plt.fill_between(range(len(y)), y_lower, y_upper, alpha=0.3)
        
    # adding title
    plt.title('Confidence level comparison of {} regret for each method in {} simulation'.format(regret_type, n_sims), fontsize=10)

    # adding legend
    plt.legend(fontsize=8); plt.xticks(fontsize=10); plt.yticks(fontsize=10)

    # showing plot
    plt.show()

def plot_regret_max(regret_df, n_sims, regret_type='instant'):
    '''Requires all the regret values'''
    # closing all past figures
    plt.close('all')

    # opening figure to plot regret
    plt.figure(figsize=(10, 3), dpi=150)

    # loop for each decision policy
    policies = regret_df['policy'].unique()
    regret_df['regret_cumsum'] = regret_df.groupby(['policy', 'simul_id'], sort = False)['regret'].cumsum()

    for policy in policies:        
        if regret_type == 'instant':
            plot_col = 'regret'
        elif regret_type == 'cumulative':
            plot_col = 'regret_cumsum'
        else:
            raise ValueError('regret_type must be either instant or cumulative')

        regret_df_max = regret_df.reset_index().groupby(['policy','round'], sort = False)[plot_col].max()
        y = regret_df_max.loc[policy,:].values

        # regret_df_std = regret_df.reset_index().groupby(['policy','round'], sort = False)[plot_col].std()
        # y_upper = y + regret_df_std.loc[policy,:].values * 1.96
        # y_lower = y - regret_df_std.loc[policy,:].values * 1.96

        plt.plot(y, label=policy, linewidth=1.5)
        # plt.fill_between(range(len(y)), y_lower, y_upper, alpha=0.3)
        
    # adding title
    plt.title('Max comparison of {} regret for each method in {} simulation'.format(regret_type, n_sims), fontsize=10)

    # adding legend
    plt.legend(fontsize=8); plt.xticks(fontsize=10); plt.yticks(fontsize=10)

    # showing plot
    plt.show()

def plot_total_regret(experiment_params_nums, result_folder):
    # load the results from pickle
    total_regret = {'ts_lr':[], 'exploit_lr':[], 'ucb_lr':[]}
    alphas = []
    for i in range(len(experiment_params_nums)):
        n_providers, n_data_point_per_round, n_rounds, n_dim, lambda_, alpha, n_sims, seed = experiment_params_nums[i]
        print(f"Analyzing simulation with {n_providers} arms, {n_data_point_per_round} data points per round, {n_rounds} rounds, {n_dim} features, lambda {lambda_}, alpha {alpha}, {n_sims} simulations")
        try:
            experiment_df = pd.read_pickle(f"../{result_folder}/experiment_{n_providers}_{n_data_point_per_round}_{n_rounds}_{n_dim}_{lambda_}_{alpha}_{n_sims}_{seed}.pkl")
        except:
            print(f"File not found for {n_providers}_{n_data_point_per_round}_{n_rounds}_{n_dim}_{lambda_}_{alpha}_{n_sims}_{seed}")
            continue
        regret_mean = experiment_df.reset_index().groupby(['policy','round'])['regret'].mean()
        # print(regret_mean.shape)
        policies = regret_mean.index.get_level_values('policy').unique()

        for policy in policies:
            regret_final = np.cumsum(regret_mean.loc[policy,:].values)[-1]
            total_regret[policy].append(regret_final)
        alphas.append(alpha)
    # plot total regret
    total_regret_df = pd.DataFrame(total_regret)

    plt.plot(total_regret_df)
    plt.xlabel('Alpha')
    plt.xticks(ticks=range(len(alphas)), labels=alphas)
    plt.legend(total_regret_df.columns)
    plt.show()
    plt.figure(figsize=(10, 3), dpi=150)


def winning_policy(experiment_df):
    # only plot the one that has ts-lr as the smallest cumulative mean
    experiment_df['regret_cumsum'] = experiment_df.groupby(['policy', 'simul_id'], sort = False)['regret'].cumsum()
    experiment_df_cum_sum_mean = experiment_df.groupby(['policy','round'], sort = False)['regret_cumsum'].mean()
    last_round = experiment_df['round'].max()
    experiment_df_cum_sum_mean_last_round = experiment_df_cum_sum_mean[experiment_df_cum_sum_mean.index.get_level_values('round') == last_round]
    print(experiment_df_cum_sum_mean_last_round)
    min_idx = experiment_df_cum_sum_mean_last_round.argmin()
    winning_policy = experiment_df_cum_sum_mean_last_round.index[min_idx][0]
    return winning_policy
    
