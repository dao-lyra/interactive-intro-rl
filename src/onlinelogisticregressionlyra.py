# defining a class for our online bayesian logistic regression
import numpy as np
from scipy.optimize import minimize

class OnlineLogisticRegression:
    
    # initializing
    def __init__(self, lambda_ = 1., alpha = 1., n_dim = 1, true_weights_full=None, n_data_point_per_round=1):
        # true weights are the weights of the true model + plus 1 for the new feature
        assert (len(true_weights_full) == n_dim + 1) | (true_weights_full is None)
        self.true_weights_full = true_weights_full
        self.true_weights = true_weights_full[:-1]
        # the only hyperparameter is the deviation on the prior (L2 regularizer)
        self.lambda_ = lambda_; self.alpha = alpha
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
        result = minimize(self.loss, self.w, args=(X, y), jac=self.grad, method="L-BFGS-B", options={'maxiter': 20, 'disp':False})
        print(result)
        self.w = result.x
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
        '''
        best_arms = np.argmax(scores, axis=1).squeeze()
        return best_arms
    
    def get_best_arm_features(self, best_arms, X):
        '''
        returns the best features
        '''
        return X[np.arange(X.shape[0]), best_arms]


    
    def predict_proba(self, w, X):
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
        There are 2 phase of reward, before and after introducing new feature
        '''
        dots = np.dot(X_best, self.true_weights)
        probs = 1/(1 + np.exp(-dots))
        rewards = np.array([np.random.choice([0,1], p=[1 - prob, prob]) for prob in probs])
        return rewards, probs
    
    def calculate_regret(self, X, X_best):
        '''
        regrets is the difference between the best proba and the proba of the selected arm
        There are 2 phasese of regret, before and after introducing new feature
        '''
        probas_of_X = self.predict_proba(self.true_weights_full, X)
        max_probas_of_X = np.max(probas_of_X, axis=1)

        probas_of_best_arms = self.predict_proba(self.true_weights, X_best)
        regrets = np.clip(max_probas_of_X - probas_of_best_arms, 0, None)
        return regrets
    
    def add_new_feature(self):
        self.true_weights = self.true_weights_full
        self.n_dim = len(self.true_weights)
        self.m = np.append(self.m, 0)
        self.q = np.append(self.q, self.lambda_)
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
        

    
              
        
class OnlineLogisticRegressionOnlineFeature(OnlineLogisticRegression):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def add_new_feature(self, true_weight):
        self.true_weights = np.append(self.true_weights, true_weight)
        self.n_dim = len(self.true_weights)
        self.m = np.append(self.m, 0)
        self.q = np.append(self.q, self.lambda_)
        self.w = np.append(self.w, np.random.normal(self.m, self.alpha * (self.q[-1]), size = self.n_dim))

    
