import seaborn as sns
import matplotlib.pyplot as plt
# from sklearn.feature_selection import f_classif
import pandas as pd
import numpy as np
from pylab import rcParams
import scipy.stats as stats
from scipy.stats import chi2
import time,tqdm


class Gaussian_process():

    measurement_variance = 1

    def __init__(self, train_inputs, train_targets, test_inputs, test_targets, k_folds):
        self.train_inputs = train_inputs
        self.train_targets = train_targets
        self.test_inputs = test_inputs
        self.test_targets = test_targets
        self.k_folds = k_folds

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

    def cross_validation_gaussian_process(self, hyperparameters):
        fold_size = len(self.train_targets)/self.k_folds
        mean_squared_errors = np.zeros(len(hyperparameters))
        for id, hyperparam in enumerate(hyperparameters):
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
        best_hyperparam = hyperparameters[np.argmin(mean_squared_errors)]
        return best_hyperparam, best_mean_squared_error, mean_squared_errors

    def training_gaussian(self, hyperparams):
        best_width, best_mean_squared_error, errors = self.cross_validation_gaussian_process(hyperparams)
        posterior = self.train_gaussian_process(self.train_inputs, self.train_targets, best_width)
        mse = self.eval_gaussian_process(self.test_inputs, posterior, self.test_targets)
        return posterior, mse, errors

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



def woe_iv_continuous(X,y,regularize:str):
    """
    Finding weight of importance and informational value for binary classification tasks
    """
    df = X.copy()
    df['target'] = y.copy()
    IV_dict = {}
    woe_dict = {}

    for col in X.columns:
        # binning values
        bins = np.linspace(df[col].min()-0.1, df[col].max()+0.1, int(0.05* X.shape[0]))  # each bin should have at least 5% of the observation
        groups = df.groupby(np.digitize(df[col], bins))
        df[col] = pd.cut(df[col], bins)

        # getting class counts for each bin
        count_series = df.groupby([col, 'target']).size()
        new_df = count_series.to_frame(name = 'size').reset_index()
        if regularize == True:
            new_df['size'] = new_df['size'] + 0.5
        df1  = new_df[new_df['target']==0].reset_index(drop=True)
        df2  = new_df[new_df['target']==1].reset_index(drop=True)
        df1['size1'] = df2['size']
        new_df = df1.drop(columns=['target'])
        sum = new_df['size'].sum()
        sum1 = new_df['size1'].sum()
        # Calculate woe and IV
        new_df['woe'] = np.log((new_df['size']/sum)/(new_df['size1']/sum1))
        new_df['IV'] = ((new_df['size']/sum) - (new_df['size1']/sum1)) * new_df['woe']
        new_df = new_df.replace([np.inf, -np.inf], np.nan)
        new_df.dropna(inplace=True)
        woe_dict[col] = new_df.drop(columns=['size','size1'])
        IV_dict[col] = new_df['IV'].sum()
    return woe_dict, IV_dict


def woe_iv_categ(X,y,regularize:str):
    """
    Finding weight of importance and informational value for binary classification tasks
    """
    df = X.copy()
    df['target'] = y.copy()
    IV_dict = {}
    woe_dict = {}

    for col in X.columns:
        # binning values
        bins = np.linspace(df[col].min()-0.1, df[col].max()+0.1, len(set(df[col])))  # each bin should have at least 5% of the observation
        groups = df.groupby(np.digitize(df[col], bins))
        df[col] = pd.cut(df[col], bins)

        # getting class counts for each bin
        count_series = df.groupby([col, 'target']).size()
        new_df = count_series.to_frame(name = 'size').reset_index()
        if regularize == True:
            new_df['size'] = new_df['size'] + 0.5
        df1  = new_df[new_df['target']==0].reset_index(drop=True)
        df2  = new_df[new_df['target']==1].reset_index(drop=True)
        df1['size1'] = df2['size']
        new_df = df1.drop(columns=['target'])
        sum = new_df['size'].sum()
        sum1 = new_df['size1'].sum()
        # Calculate woe and IV
        new_df['woe'] = np.log((new_df['size']/sum)/(new_df['size1']/sum1))
        new_df['IV'] = ((new_df['size']/sum) - (new_df['size1']/sum1)) * new_df['woe']
        new_df = new_df.replace([np.inf, -np.inf], np.nan)
        new_df.dropna(inplace=True)
        woe_dict[col] = new_df.drop(columns=['size','size1'])
        IV_dict[col] = new_df['IV'].sum()
    return woe_dict, IV_dict

def barchart_dict(d):
    d = dict(sorted(d.items(), key=lambda item: item[1]))
    rcParams['figure.figsize'] = 20, 10
    plt.bar(range(len(d)), d.values(), align='center')
    plt.xticks(range(len(d)), list(d.keys()))
    plt.xticks(rotation=90)
    plt.show()

def heatmap(df):
    corrmat = df.corr()
    top_corr_features = corrmat.index
    plt.figure(figsize=(20,20))
    g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")

# def pvalues_plot(X,y):
#     """
#     Plotting p-values for all the univariate tests
#     """
#     sel = f_classif(X,y)
#     p_values = pd.Series(sel[1])
#     p_values.index = X.columns
#     p_values.sort_values(ascending=True, inplace=True)
#     # Visualize p-values
#     p_values.plot.bar(figsize=(16,5), title='p-values')
    
def chi_square(X,y,alpha:float):
    result = pd.DataFrame(columns=['Independent_Variable','Alpha','Degree_of_Freedom', 'Chi_Square','P_value','Conclusion'])
    for col in X.columns:
        table = pd.crosstab(y,X[col])
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






