import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# from pylab import rcParams
import scipy.stats as stats
from scipy.stats import chi2
import time,tqdm
from imblearn.over_sampling import SMOTE
import lightgbm as lgb  
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


class Eda():
    '''
    inputs are dataframes
    ''' 
    def __init__(self, predictors, target):
        self.predictors = predictors
        self.target = target

    @staticmethod
    def normalization(data):
        '''
        Squeeze values to -1, 1
        Alternative: from sklearn.preprocessing import MinMaxScaler
        '''
        data = np.array(data)

        max_predictors = []
        min_predictors = []

        normalized_predictors= np.empty(shape=data.shape)

        for i in range(data.shape[1]):
            max_predictors.append(max([_[i] for _ in data]))
            min_predictors.append(min([_[i] for _ in data]))

        for index, arr_predictors in enumerate(data):
            predictors = [(e - mi)/(ma - mi) for e,mi,ma in list(zip(arr_predictors,min_predictors,max_predictors))]
            normalized_predictors[index] = predictors

        return normalized_predictors

    @staticmethod
    def standardization(data):
        '''
        Squeeze values to -1, 1
        Alternative: from sklearn.preprocessing import StandardScaler
        '''
        data = np.array(data)

        mean_predictors = []
        std_predictors = []

        standardized_predictors = np.empty(shape=data.shape)

        for i in range(data.shape[1]):
            mean_predictors.append(np.mean([_[i] for _ in data]))
            std_predictors.append(np.std([_[i] for _ in data]))

        for index, arr_predictors in enumerate(data):
            predictors = [(e - mean)/std for e,mean,std in list(zip(arr_predictors,mean_predictors,std_predictors))]
            standardized_predictors[index] = predictors

        return standardized_predictors

    def pca(self, n_pca:int):

        '''
        Return an array of new pca components as well as important features
        '''
        scaled_data = self.standardization(self.predictors)
        
        n_pcs = n_pca

        pca = PCA(n_components=n_pcs)
        pca.fit(scaled_data)
        x_pca = pca.transform(scaled_data)

        # get the index of the most important feature on EACH component
        most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]

        initial_feature_names = self.predictors.columns
        # get the feature names
        most_important_features = [initial_feature_names[most_important[i]] for i in range(n_pcs)]

        # build the dataframe
        dic = {f'PC{i}': most_important_features[i] for i in range(n_pcs)}
        df = pd.DataFrame(dic.items())

        return x_pca, df

    def kmeans(self, data, n_cluster:int):
        '''
        Return an array of cluster labels with plot. Maximum cluster size is 10
        It's recommended to run pca funcation first before clustering
        '''
        km = KMeans(n_clusters=n_cluster)

        pred = km.fit_predict(data)

        color_list = ['green','navy','yellow','orange','brown','pink','red', 'purple','black','blue']

        for i in range(n_cluster):
            idx = np.where(pred==i)
            color = color_list.pop()
            plt.scatter(data[idx][:,0], data[idx][:,1], color=str(color), label=f"cluster {i}")

        plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], color='green', marker='*', label='centroid')

        plt.xlabel('mean radius')
        plt.ylabel('mean texture')
        plt.legend()

    @staticmethod
    def kmeans_elbow(data):
        parameters = range(1,10)
        sse = []
        for k in parameters:
            km = KMeans(n_clusters=k)
            km.fit(data)
            sse.append(km.inertia_)
        plt.xlabel('k number')
        plt.ylabel('sse')
        plt.plot(parameters, sse)

    def woe_iv_continuous(self):
        """
        Finding weight of importance and informational value for binary classification tasks
        """
        df = self.predictors.copy()
        df['target'] = self.target.copy()
        IV_dict = {}
        woe_dict = {}

        for col in self.predictors.columns:
            # binning values
            bins = np.linspace(df[col].min()-0.1, df[col].max()+0.1, int(0.05* self.predictors.shape[0]))  # each bin should have at least 5% of the observation
            groups = df.groupby(np.digitize(df[col], bins))
            df[col] = pd.cut(df[col], bins)

            # getting class counts for each bin
            count_series = df.groupby([col, 'target']).size()
            new_df = count_series.to_frame(name = 'size').reset_index()

            new_df['size'] = new_df['size'] + 0.5
            df1  = new_df[new_df['target']==0].reset_index(drop=True)
            df2  = new_df[new_df['target']==1].reset_index(drop=True)
            df1['size1'] = df2['size']
            new_df = df1.drop(columns=['target'])
            sum_ = new_df['size'].sum()
            sum1 = new_df['size1'].sum()
            # Calculate woe and IV
            new_df['woe'] = np.log((new_df['size']/sum_)/(new_df['size1']/sum1))
            new_df['IV'] = ((new_df['size']/sum_) - (new_df['size1']/sum1)) * new_df['woe']
            new_df = new_df.replace([np.inf, -np.inf], np.nan)
            new_df.dropna(inplace=True)
            woe_dict[col] = new_df.drop(columns=['size','size1'])
            IV_dict[col] = new_df['IV'].sum()
        return woe_dict, IV_dict


    def woe_iv_categ(self):
        """
        Finding weight of importance and informational value for binary classification tasks
        """
        df = self.predictors.copy()
        df['target'] = self.target.copy()
        IV_dict = {}
        woe_dict = {}

        for col in self.predictors.columns:
            # binning values
            bins = np.linspace(df[col].min()-0.1, df[col].max()+0.1, len(set(df[col])))  # each bin should have at least 5% of the observation
            groups = df.groupby(np.digitize(df[col], bins))
            df[col] = pd.cut(df[col], bins)

            # getting class counts for each bin
            count_series = df.groupby([col, 'target']).size()
            new_df = count_series.to_frame(name = 'size').reset_index()

            new_df['size'] = new_df['size'] + 0.5
            df1  = new_df[new_df['target']==0].reset_index(drop=True)
            df2  = new_df[new_df['target']==1].reset_index(drop=True)
            df1['size1'] = df2['size']
            new_df = df1.drop(columns=['target'])
            sum_ = new_df['size'].sum()
            sum1 = new_df['size1'].sum()
            # Calculate woe and IV
            new_df['woe'] = np.log((new_df['size']/sum_)/(new_df['size1']/sum1))
            new_df['IV'] = ((new_df['size']/sum_) - (new_df['size1']/sum1)) * new_df['woe']
            new_df = new_df.replace([np.inf, -np.inf], np.nan)
            new_df.dropna(inplace=True)
            woe_dict[col] = new_df.drop(columns=['size','size1'])
            IV_dict[col] = new_df['IV'].sum()
        return woe_dict, IV_dict

    @staticmethod
    def barchart_dict(iv_dict:dict):
        d = dict(sorted(iv_dict.items(), key=lambda item: item[1]))
        # rcParams['figure.figsize'] = 20, 10
        plt.bar(range(len(d)), d.values(), align='center')
        plt.xticks(range(len(d)), list(d.keys()))
        plt.xticks(rotation=90)
        plt.show()

    def heatmap(self):
        df = self.predictors.copy()
        df['target'] = self.target.copy()
        corrmat = df.corr()
        top_corr_features = corrmat.index
        plt.figure(figsize=(20,20))
        g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")
        
    def chi_square(self, alpha:float):
        result = pd.DataFrame(columns=['Independent_Variable','Alpha','Degree_of_Freedom', 'Chi_Square','P_value','Conclusion'])
        for col in self.predictors.columns:
            table = pd.crosstab(self.target,self.predictors[col])
            print(f"Null hypothesis: there's no relationship between {col} and the response variable")
            observed_freq = table.values
            val = stats.chi2_contingency(observed_freq)
            expected_freq = val[3]
            dof = (table.shape[0]-1) * (table.shape[1]-1)
            chi_square = sum([(o-e)**2/e for o,e in zip(observed_freq,expected_freq)])
            chi_square_statistic = chi_square[0] + chi_square[1]
            p_value = 1-chi2.cdf(x=chi_square_statistic,df=dof)
            if p_value <= alpha:
                print(f"Test result rejects the null hypothesis. There is a relationship between the {col} and the response variable")
                conclusion = "There's a relationship"
            else:
                print(f"Test result fails to reject the null hypothesis. There is no evidence to prove there's a relationship between {col} and the response variable")
                conclusion = "There's no relationship"
            result = result.append(pd.DataFrame([[col,alpha, dof,chi_square_statistic, p_value,conclusion]],columns=result.columns))
        return result

class ML_models():
    def __init__(self, train_inputs, train_targets, test_inputs, test_targets):
        self.train_inputs = train_inputs
        self.train_targets = train_targets
        self.test_inputs = test_inputs
        self.test_targets = test_targets

    def plot_cost_function(self,error,hyperparameters):
        plt.plot(hyperparameters,error)
        plt.ylabel('error')
        plt.xlabel('parameter')
        plt.show()

    def concept_drift(self):
        '''
        Evaluate concept drift between train and test data and return sample_weights as a hyperparameter to model's training
        '''

        trn, tst = self.train_inputs.copy(), self.test_inputs.copy()
        trn = np.concatenate((trn,np.ones((trn.shape[0],1))),1)
        tst = np.concatenate((tst,np.zeros((tst.shape[0],1))),1)

        merged_array = np.vstack((trn,tst))

        X_ = np.asarray([e[:-1] for e in list(merged_array)])
        y_ = np.asarray([e[-1] for e in list(merged_array)])

        predictions = np.zeros(y_.shape)
        
        np.random.seed(123)
        lgb_model = lgb.LGBMClassifier(n_jobs=-1, max_depth=-1, n_estimators=500, learning_rate=0.1, num_leaves=30, colsample_bytree=0.28, objective='binary')

        skf = StratifiedKFold(n_splits=20, shuffle=True, random_state=123)
        for fold, (train_idx, test_idx) in enumerate(skf.split(X_, y_)):
            X_train, X_test = X_[train_idx], X_[test_idx]
            y_train, y_test = y_[train_idx], y_[test_idx]

            lgb_model.fit(X_train, y_train, eval_metric='auc', eval_set = [(X_test, y_test)], verbose=False)
            probs_train = lgb_model.predict_proba(self.train_inputs)[:, 1] 
            probs_test = lgb_model.predict_proba(self.test_inputs)[:, 1] 

        predictions = np.append(probs_train, probs_test)
        score = round(roc_auc_score(y_, predictions),2)

        if score >= 0.7:
            print(f"The model can differentiate between train and test data with an AUC of {score}. There seem to exist a significant level of concept drift between the test and train data")
        else:
            print(f"The model cannot strongly differentiate between train and test data with as its AUC is only {score}. There doesn't seem to exist a significant level of concept drift between the test and train data")


        sns.kdeplot(probs_train, label='train', legend=True)
        sns.kdeplot(probs_test, label='test', legend=True)        

        predictions_train = predictions[:len(self.train_inputs)] # getting the training array
        sample_weights = (1 - predictions_train) / predictions_train
        sample_weights /= np.mean(sample_weights) # Normalizing the weights

        return sample_weights

class Gaussian_process(ML_models):

    measurement_variance = 1

    def __init__(self, train_inputs, train_targets, test_inputs, test_targets, k_folds, hyperparameters):
        super().__init__(train_inputs, train_targets, test_inputs, test_targets)
        self.k_folds = k_folds
        self.hyperparameters = hyperparameters

    def __str__(self):
        return(f'train_input size is {self.train_inputs.shape}, test_input size is {self.test_inputs.shape},train_target size is {self.train_targets.shape}, test_target size is {self.test_targets.shape}')
    
    __repr__ = __str__

    @staticmethod
    def predict_gaussian_process(inputs, posterior):
        mean,variance = posterior(inputs)
        return mean, variance

    def eval_gaussian_process(self,inputs, posterior, targets):
        mean, variance = self.predict_gaussian_process(inputs,posterior)
        errors = mean - targets
        mean_squared_error = np.sum(errors ** 2) / len(targets)

        return mean_squared_error
    
    def train_gaussian_process(self,train_inputs, train_targets, param):
        K = self.gaussian_kernel(train_inputs,train_inputs, param)
        regularized_gram_matrix = K + self.measurement_variance * np.identity(K.shape[0])
        inverse_regularized_gram_matrix = np.linalg.inv(regularized_gram_matrix)

        def posterior(inputs, train_inputs=train_inputs, train_targets=train_targets, inverse_regularized_gram_matrix=inverse_regularized_gram_matrix):
            mean = np.matmul(np.matmul(self.gaussian_kernel(inputs,train_inputs,param),inverse_regularized_gram_matrix),train_targets)
            variance = np.diag(self.gaussian_kernel(inputs,inputs,param) + self.measurement_variance * np.identity(inputs.shape[0]) - np.matmul(self.gaussian_kernel(inputs,train_inputs,param),inverse_regularized_gram_matrix).dot(self.gaussian_kernel(train_inputs,inputs,param)))
            return mean, variance

        return posterior

    def gaussian_kernel(self,inputs1,inputs2,width):
        euclidean_distance = np.sum(np.square(inputs1),1)[:,None] - 2 * np.matmul(inputs1,inputs2.transpose()) + np.sum(np.square(inputs2),1).transpose()
        gram_matrix = np.exp(-euclidean_distance / (2 * np.square(width)))
        return gram_matrix

    def identity_kernel(self, dummy_param=None):
        gram_matrix = np.matmul(self.train_inputs,self.train_inputs.transpose())
        return gram_matrix

    def cross_validation_gaussian_process(self):
        fold_size = len(self.train_targets)/self.k_folds
        mean_squared_errors = np.zeros(len(self.hyperparameters))
        for id, hyperparam in enumerate(self.hyperparameters):
            for fold in tqdm.tqdm(range(self.k_folds)):
                time.sleep(0.00001)
            
                validation_inputs = self.train_inputs[int(round(fold*fold_size)):
                                            int(round((fold+1)*fold_size))]
                validation_targets = self.train_targets[int(round(fold*fold_size)):
                                            int(round((fold+1)*fold_size))]
                train_inputs = np.concatenate((self.train_inputs[:int(round(fold*fold_size))],
                                                self.train_inputs[int(round((fold+1)*fold_size)):]))
                train_targets = np.concatenate((self.train_targets[:int(round(fold*fold_size))],
                                                self.train_targets[int(round((fold+1)*fold_size)):]))
                posterior = self.train_gaussian_process(train_inputs, train_targets, hyperparam)
                mean_squared_errors[id] += self.eval_gaussian_process(self.train_inputs, posterior, self.train_targets)
        mean_squared_errors /= self.k_folds
        best_mean_squared_error = np.min(mean_squared_errors)
        best_hyperparam = self.hyperparameters[np.argmin(mean_squared_errors)]
        return best_hyperparam, best_mean_squared_error, mean_squared_errors


    def training_gaussian(self):

        best_width, best_mean_squared_error, errors = self.cross_validation_gaussian_process()
        posterior = self.train_gaussian_process(self.train_inputs, self.train_targets, best_width)
        mse = self.eval_gaussian_process(self.test_inputs, posterior, self.test_targets)

        return posterior, mse, errors


class LogisticReg(ML_models):

    def __init__(self, train_inputs, train_targets, test_inputs, test_targets, k_folds, hyperparameters, imbalanced=False, upsampling=False, optimizer='newton'):
        super().__init__(train_inputs, train_targets, test_inputs, test_targets)
        self.k_folds = k_folds
        self.hyperparameters = hyperparameters
        self.imbalanced = imbalanced
        self.upsampling = upsampling
        self.optimizer = optimizer


    def __str__(self):
        return(f'train_input size is {self.train_inputs.shape}, test_input size is {self.test_inputs.shape},train_target size is {self.train_targets.shape}, test_target size is {self.test_targets.shape}')
    
    __repr__ = __str__

    @staticmethod
    def conjGrad(A,x0,b,tol,maxit):
        rold = b.reshape(-1) - np.asarray(np.matmul(A,x0)).reshape(-1)
        rold = rold.reshape(len(rold),1) 
        pold = rold
        # Store values
        steps = 0
        X = x0 
        res = []
        init_normR0 = norm(rold)
        ## tolerance check 
        tol_check = 1
        for k in range(maxit):
            rt_old = np.transpose(rold)
            pt_old = np.transpose(pold)
            mult1 = np.matmul(rt_old,rold)
            mult2 = np.matmul(np.matmul(pt_old,A),pold)
            alpha = mult1/mult2
            # Update the solution
            alpha_pold = alpha*pold.reshape(-1)
            X= X.reshape(-1) + alpha_pold 

            rnew = rold - alpha*np.matmul(A,pold)

            rt_new = np.transpose(rnew)
            mult3 = np.matmul(rt_new,rnew)
            beta = mult3/mult1
            pnew = rnew + beta*pold
            ResCal = norm(rnew)
            res.append(ResCal)
            tol_check = ResCal/init_normR0
            # update p and r
            pold = pnew
            rold = rnew
            steps = steps + 1 
            if (tol_check<tol):
                break 
            
        return X

    @staticmethod
    def identity_kernel(train_inputs):
        gram_matrix = np.matmul(train_inputs,train_inputs.transpose())
        return gram_matrix

    @staticmethod
    def sigmoid(input):
        output = 1/(1+np.exp(-input))
        return output

    def predict_logistic_regression(self,inputs, weights):
        logits = np.matmul(inputs,weights)
        sigma = self.sigmoid(logits)
        predicted_probabilities = np.column_stack((1-sigma,sigma))
        return predicted_probabilities

    @staticmethod
    def calculate_f1(labels,prediction):
        labels = np.append(labels, 0)
        labels = np.append(labels, 1)
        prediction = np.append(prediction, 0)
        prediction = np.append(prediction, 1)

        df_confusion = pd.crosstab(labels, prediction)

        TN = df_confusion.iloc[0,0]
        FN = df_confusion.iloc[1,0]
        TP = df_confusion.iloc[1,1]
        FP = df_confusion.iloc[0,1]
        
        recall_0 = TN / (TN + FP)
        recall_1 = TP / (TP + FN)

        precision_0 = TN / TN + FN
        precision_1 = TP / TP + FP

        f1_class0 = (2 * recall_0 * precision_0) / (recall_0 + precision_0)
        f1_class1 = (2 * recall_1 * precision_1) / (recall_1 + precision_1)
        # ave_f1 = round((f1_class0 + f1_class1) / 2,2)

        return f1_class0

    def eval_logistic_regression(self,inputs, weights, labels):

        predicted_probabilities = self.predict_logistic_regression(inputs,weights)
        prediction = np.argmax(predicted_probabilities,axis=1)

        # accuracy_check = prediction - labels
        # accuracy = 1-(np.count_nonzero(accuracy_check) / labels.shape[0])
        if self.imbalanced == True:
            score = self.calculate_f1(labels, prediction)
        else:
            score = -sum(labels * np.log(self.sigmoid(inputs.dot(weights))) + (1-labels) * np.log(1 - self.sigmoid(inputs.dot(weights))))
        
        return score

    @staticmethod
    def initialize_weights(n_weights):
        np.random.seed(123)
        random_weights = np.random.rand(n_weights)/10 - 0.05
        return random_weights

    def train_logistic_regression(self,lambda_hyperparam, train_inputs=None, train_targets=None):
        if train_inputs is None:
            train_inputs = self.train_inputs
            train_targets = self.train_targets

        weights = self.initialize_weights(train_inputs.shape[1])

        if self.optimizer == 'gd':
            transformed_train_inputs = np.matmul(train_inputs.T, train_inputs)
            transformed_train_targets = np.matmul(train_inputs.T, train_targets)
            tol = 10**(-10)
            maxit = 500
            weights = self.conjGrad(transformed_train_inputs, weights, transformed_train_targets ,tol,maxit).reshape(-1)

        else:
            weights = self.initialize_weights(train_inputs.shape[1])
            R = np.identity(train_inputs.shape[0])
            # Finding optimal weights
            max_change = 1
            while max_change > 0.001:
            # Creating R matrix of size N-N
                for i in range(train_inputs.shape[0]):
                    R[i][i] = self.sigmoid(train_inputs[i].dot(weights)) * (1 - self.sigmoid(train_inputs[i].dot(weights)))

                # Creating H with the lambda param
                H = train_inputs.T.dot(R).dot(train_inputs) + lambda_hyperparam * np.identity(train_inputs.shape[1])
                inverse_H = np.linalg.inv(H)
                # Finding gradient of the loss function with the lambda param
                gradient_L = train_inputs.T.dot(self.sigmoid(train_inputs.dot(weights)) - train_targets) + lambda_hyperparam * weights
                # Find new weights
                delta = inverse_H.dot(gradient_L)
                weights = weights - delta
                max_change = max(delta)

        return weights  

    def cross_validation_logistic_regression(self):
        fold_size = len(self.train_targets)/self.k_folds
        scores = np.zeros(len(self.hyperparameters))

        # Oversample the under-represented class
        if self.upsampling == True:
            os = SMOTE(random_state=123)
            os_data_X,os_data_y=os.fit_sample(self.train_inputs, self.train_targets)
            self.train_inputs = np.asarray(os_data_X)
            self.train_targets = np.asarray(os_data_y)

        for id, hyperparam in enumerate(self.hyperparameters):
            for fold in range(self.k_folds):
                validation_inputs = self.train_inputs[int(round(fold*fold_size)):int(round((fold+1)*fold_size))]
                validation_targets = self.train_targets[int(round(fold*fold_size)):int(round((fold+1)*fold_size))]
                train_inputs = np.concatenate((self.train_inputs[:int(round(fold*fold_size))],self.train_inputs[int(round((fold+1)*fold_size)):]))
                train_targets = np.concatenate((self.train_targets[:int(round(fold*fold_size))],self.train_targets[int(round((fold+1)*fold_size)):]))
                weights = self.train_logistic_regression(hyperparam, train_inputs, train_targets)
                score = self.eval_logistic_regression(validation_inputs, weights, validation_targets)
                scores[id] += score

        scores /= self.k_folds

        if self.imbalanced == True:
            best_score = np.max(scores)
            best_hyperparam = self.hyperparameters[np.argmax(scores)]
            return best_hyperparam, best_score, scores
        else:
            best_score = np.min(scores)
            best_hyperparam = self.hyperparameters[np.argmin(scores)]
            return best_hyperparam, best_score, scores


class Stack():
    def __init__(self):
        self.stack = list()
    def __str__(self):
        return str(self.stack)
    __repr__ = __str__

    def push(self,n):
        self.stack.append(n)
    def pop(self):
        if len(self.stack) > 0:
            return self.stack.pop()
        else:
            return None
    def peek(self):
        return self.stack[len(self.stack)-1]

class Queue():
    def __init__(self):
        self.queue = list()

    def __str__(self):
        return str(self.queue)

    __repr__ = __str__

    def enqueue(self,n):
        self.queue.append(n)
    def dequeue(self):
        if len(self.queue) > 0:
            return self.queue.pop(0)
        else:
            return None
    def peek(self):
        return self.queue[0]

class MaxHeap():
    def __init__(self, items =[]):
        # super().__init__()
        self.heap = [0]
        for item in items:
            self.heap.append(item)
            self.__floatUp(len(self.heap)-1)

    def __str__(self):
        return str(self.heap)
    
    __repr__ = __str__

    def push(self, data):
        self.heap.append(data)
        self.__floatUp(len(self.heap)-1)

    def peek(self):
        if len(self.heap) > 1:
            return self.heap[1]
        else:
            return None

    def pop(self):
        if len(self.heap) > 2:
            self.__swap(1, len(self.heap)-1)
            max = self.heap.pop()
            self.__bubbleDown(1)
        elif len(self.heap) == 2:
            max = self.heap.pop()
        else:
            max = None
        return max

    def __swap(self, i, j):
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]

    def __floatUp(self,index):
        parent_index = index // 2
        if index <= 1:
            return
        elif self.heap[index] > self.heap[parent_index]:
            self.__swap(index, parent_index)
            self.__floatUp(parent_index)
        
    def __bubbleDown(self, index):
        left_sub = index * 2
        right_sub = left_sub + 1
        largest = index
        if len(self.heap) > left_sub and self.heap[left_sub] > self.heap[largest]:
            largest = left_sub
        if len(self.heap) > right_sub and self.heap[right_sub] > self.heap[largest]:
            largest = right_sub
        if largest != index:
            self.__swap(index,largest)
            self.__bubbleDown(largest)

class MinHeap():
    def __init__(self, items =[]):
        # super().__init__()
        self.heap = [0]
        for item in items:
            self.heap.append(item)
            self.__floatUp(len(self.heap)-1)

    def __str__(self):
        return str(self.heap)

    __repr__ = __str__
        
    def push(self, data):
        self.heap.append(data)
        self.__floatUp(len(self.heap)-1)

    def peek(self):
        if len(self.heap) > 1:
            return self.heap[1]
        else:
            return None

    def pop(self):
        if len(self.heap) > 2:
            self.__swap(1, len(self.heap)-1)
            max = self.heap.pop()
            self.__bubbleDown(1)
        elif len(self.heap) == 2:
            max = self.heap.pop()
        else:
            max = None
        return max

    def __swap(self, i, j):
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]

    def __floatUp(self,index):
        parent_index = index // 2
        if index <= 1:
            return
        elif self.heap[index] < self.heap[parent_index]:
            self.__swap(index, parent_index)
            self.__floatUp(parent_index)
        
    def __bubbleDown(self, index):
        left_sub = index * 2
        right_sub = left_sub + 1
        largest = index
        if len(self.heap) > left_sub and self.heap[left_sub] < self.heap[largest]:
            largest = left_sub
        if len(self.heap) > right_sub and self.heap[right_sub] < self.heap[largest]:
            largest = right_sub
        if largest != index:
            self.__swap(index,largest)
            self.__bubbleDown(largest)

class Node:
    def __init__(self, d, n=None, p=None):
        self.data = d
        self.next = n 
        self.prev = p

class LinkedList():
    def __init__(self, r=None):
        self.root = r
        self.size=0

    def __str__(self):
        this_node = self.root
        while this_node is not None:
            print(this_node.data, end=" -> ")
            this_node = this_node.next
        print('end of list')

    __repr__ = __str__
    

    def add(self,d):
        new_node = Node(d, self.root)
        self.root = new_node
        self.size += 1

    def find(self,d):
        this_node = self.root
        while this_node is not None:
            if this_node.data == d:
                return(f'Item found')
            else:
                this_node = this_node.next
        return(f"this item doesn't exist in the list")

    def remove(self,d):
        this_node = self.root
        prev_node = None

        while this_node is not None:
            if this_node.data == d: 
                if prev_node is None:
                    self.root = this_node.next 
                else:
                    prev_node.next = this_node.next
                self.size -= 1
                return(f'item removed')
            else:
                this_node = this_node.next
                prev_node = this_node
        return(f'no item in the list')














