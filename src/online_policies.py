# defining a class for our online bayesian logistic regression
import numpy as np
from scipy.optimize import minimize

def random_argmax(vector):
  """Helper function to select argmax at random... not just first one."""
  index = np.random.choice(np.where(vector == vector.max())[0])
  return index

class OnlineLogisticRegression:
    
    # initializing
    def __init__(self, lambda_ = 1., alpha = 1., n_dim = 1, true_weights_full=None, n_data_point_per_round=1):
        # true weights are the weights of the true model + plus 1 for the new feature
        assert (len(true_weights_full) == n_dim + 1) | (true_weights_full is None)
        self.true_weights_full = true_weights_full
        self.true_weights = true_weights_full[:-1]
        # the only hyperparameter is the deviation on the prior (L2 regularizer)
        self.lambda_ = lambda_ 
        # for exploration
        self.alpha = alpha
        # set high to look for long term effect of the policy
        # set to 1 to look for the immediate effect of the policy
        self.n_data_point_per_round = n_data_point_per_round
                
        # initializing parameters of the model
        self.n_dim = len(self.true_weights)
        self.m = np.zeros(self.n_dim)
        self.q = np.ones(self.n_dim) * self.lambda_
        
        # initializing weights
        self.w = self.m
        
    # the loss function
    def loss(self, w, *args):
        X, y = args
        return 0.5 * (self.q * (w - self.m)).dot(w - self.m) + np.sum([np.log(1 + np.exp(-y[j] * w.dot(X[j]))) for j in range(y.shape[0])])
        
    # the gradient
    def grad(self, w, *args):
        X, y = args
        return self.q * (w - self.m) + (-1) * np.array([y[j] *  X[j] / (1. + np.exp(y[j] * w.dot(X[j]))) for j in range(y.shape[0])]).sum(axis=0)
    
    # method for sampling weights
    def sample_weights(self):
        return np.random.normal(self.m, self.alpha * (self.q)**(-1.0), size = (self.n_data_point_per_round, self.n_dim))
    
    # fitting method
    def fit(self, X, y):
                
        # step 1, find w
        self.w = minimize(self.loss, self.w, args=(X, y), jac=self.grad, method="L-BFGS-B", options={'maxiter': 20, 'disp':False}).x
        self.m = self.w
        
        # step 2, update q
        P = (1 + np.exp(-1*X.dot(self.m))) ** (-1)
        self.q = self.q + (P*(1-P)).dot(X ** 2)
                
    # probability output method, using weights sample
    def get_weights(self):
        pass

    def calculate_score(self, X):
        '''
        score used to select the best arm/providers
        '''
        pass

    def get_best_arm(self, scores):
        '''
        returns the best arm/provider
        scores size: (n_data_per_round, n_arms)
        '''
        # Fix argmax
        best_arms = np.argmax(scores, axis=1).squeeze()
        # Random argmax
        # best_arms = np.apply_along_axis(random_argmax, 1, scores)
        return best_arms
    
    def get_best_arm_features(self, best_arms, X):
        '''
        returns the best features
        X size: (n_data_points, n_arms, n_dim)
        '''
        return X[np.arange(X.shape[0]), best_arms]


    
    def predict_proba(self, w, X):
        '''
        w: weights size (n_dim,)
        X: features size (n_data_points, n_arms, n_dim)
        dot works when Last Axis of AA This axis must match the size of the second-to-last axis of BB.
        '''
        # calculating probabilities
        # proba = 1 / (1 + np.exp(-1 * X.dot(w)))
        dots = np.dot(X, w)
        proba = 1/(1 + np.exp(-dots))
        return proba # 
    
    def get_reward(self, X_best):
        '''
        Given an x, which is an arm feature vector, returns the reward for that arm
        the reward is a probability of success
        It works with the matrix of winning arms
        X size: (n_data_points, 1 x n_features)
        There are 2 phases of reward, before and after introducing new feature
        '''
        dots = np.dot(X_best, self.true_weights)
        probs = 1/(1 + np.exp(-dots))
        rewards = np.array([np.random.choice([-1,1], p=[1 - prob, prob]) for prob in probs])
        return rewards, probs
    
    def get_reward_with_delayed_feedback(self, rewards, prob = 0.0):
        '''
        We simulate the delayed feedback by flipping the reward p%
        Delay cause 0 (-1) flip to 1 but not the other way around
        
        2024/7/23
        In reality, the reward network generates a 1 but due to the delay, it is not shown
        which means, the reward is 0, which means the flipping is from 1 to -1

        We want: an array that has -1 flipped to 1 with probability p

        Also lyra may have the storage that is can store the delayed feedback
        Most importanly is a showes with a 1 will leads to the rest of them 0.
        Only problem is all 0 booking list.
        Also
        '''
        # flip 0
        # how this works: if rewards is -1, we select -1 with 1-prob, and 1 with prob. therefore simululate the flip at prob
        # delayed_rewards = np.array([np.random.choice([-1,1], p=[1 - prob, prob]) if reward == -1 else reward for reward in rewards])
        # flip 1
        delayed_rewards = np.array([np.random.choice([-1,1], p=[prob, 1 - prob]) if reward == 1 else reward for reward in rewards])

        return delayed_rewards

    def calculate_regret(self, X, X_best):
        '''
        regrets is the difference between the best proba and the proba of the selected arm
        There are 2 phases of regret, before and after introducing new feature

        # chaing the the weithgs in 2 phases some how surges the regret in the exploration algorithms
        # than the exploit. 

        2024/7/23:
        The new regret function is :
        phase 1: y = true_weight*X + 1 
            max_proba = max(proba(y))

        phase 2: y = true_weight_full*X_full
            max_proba = max(proba(y))

        In this way: the regret is phase 1 always >0, while regret in phase 2 can be 0.

        The other thing that maybe easier to implment is just use the observed true weights
        in each phase. In this case we the regret is 0 and surge and 0

        '''
        probas_of_X = self.predict_proba(self.true_weights, X)
        max_probas_of_X = np.max(probas_of_X, axis=1)

        probas_of_best_arms = self.predict_proba(self.true_weights, X_best)
        regrets = np.clip(max_probas_of_X - probas_of_best_arms, 0, None)
        return regrets
    
    def add_new_feature(self, m = 0, lambda_ = None):
        self.true_weights = self.true_weights_full
        self.n_dim = len(self.true_weights)
        # m and lambda_ is the mean and precision prior of the new feature
        self.m = np.append(self.m, m)
        if lambda_ is None:
            lambda_ = self.lambda_
        self.q = np.append(self.q, lambda_)
        self.w = self.m

class ThompsonSamplingPolicy(OnlineLogisticRegression):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def get_weights(self):
        W = self.sample_weights()
        # how does it reflect in the loss/fit function, probably just use self.m
        self.W = W
        return W
    
    def calculate_score(self, X):
        '''
        score used to select the best arm/providers
        W: multiple weights as it is TS size: (n_data_points, n_dim)
        X: features size (n_data_points, n_arms, n_dim)
        scores size: (n_data_points, n_arms)
        '''
        W = self.get_weights() # generate multiple weights as it is TS
        scores = np.einsum('ijk,ki->ij', X, W.T)
        # scores = np.dot(X, w.T)
        return scores

class UCBPolicy(OnlineLogisticRegression):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def calculate_score(self, X):
        '''
        score used to select the best arm/providers
        '''
        scores_mean = np.dot(X, self.m) # dims = n_data_points x n_arms
        scores_std = self.alpha * np.sqrt(np.dot(X ** 2, self.q**(-1))) # # dims = n_data_points x n_arms
        return scores_mean + scores_std # dims = n_data_points x n_arms

class GreedyPolicy(OnlineLogisticRegression):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def get_weights(self):
        self.w = self.m
        return self.w
    
    def calculate_score(self, X):
        '''
        score used to select the best arm/providers
        '''
        w = self.get_weights()
        scores = np.dot(X, w)
        return scores
    
class RandomPolicy(OnlineLogisticRegression):
        
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            
        def calculate_score(self, X):
            '''
            Random policy return random scores
            '''
            return np.random.normal(size=X.shape[0:2])                
        

    
