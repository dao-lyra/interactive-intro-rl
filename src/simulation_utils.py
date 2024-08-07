import pandas as pd

def round_simulation(n_data_point_per_round, n_providers, n_dim, dp_dict, simulation, round_idx, X_full, X):
    round_df = pd.DataFrame({'k': [], 'x': [], 'reward': [], 'regret': [],
                            'true_weights': [], 'sample_weights': [], 
                            'mean_weights': [], 'precision_weights': [],
                            'round': [],
                            'simul_id': [],
                            'policy': []})
                # generate data    
    for key, online_lr in dp_dict.items():
        scores = online_lr.calculate_score(X)
        best_arms = online_lr.get_best_arm(scores)
        # print(f"best_arms: {best_arms}")
        best_arm_features = online_lr.get_best_arm_features(best_arms, X)
        # fix the reward randomness to test alpha = 0
        # np.random.seed(100)
        rewards, _ = online_lr.get_reward(best_arm_features)
        # np.random.seed(None)
        # delay the reward
        # rewards = online_lr.get_reward_with_delayed_feedback(rewards, prob = prob_delayed)
        regrets = online_lr.calculate_regret(X, best_arm_features)

        # log the data
        if key == 'ts_lr':
            W = list(online_lr.W)
        else:
            W = [online_lr.w]*n_data_point_per_round

        temp_df = pd.DataFrame({'k': best_arms, 'x': list(best_arm_features),
                                'reward': rewards, 'regret': regrets,
                                'true_weights': [online_lr.true_weights]*n_data_point_per_round,
                                'sample_weights': W,
                                'mean_weights': [online_lr.m]*n_data_point_per_round,
                                'precision_weights': [online_lr.q]*n_data_point_per_round,
                                'round':[round_idx]*n_data_point_per_round}, 
                                index=[round_idx]*n_data_point_per_round)

        # fit th e model
        online_lr.fit(best_arm_features, rewards)

        # append to round_df
        temp_df = temp_df.assign(simul_id = simulation, policy=key)

        round_df = pd.concat([round_df, temp_df])
    return round_df  