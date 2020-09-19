import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
import pandas as pd
import numpy as np
from pylab import rcParams
import statsmodels.api as sm
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.metrics import mean_squared_error
from math import sqrt
import eli5
from eli5.sklearn import PermutationImportance
import datetime
import scipy.stats as stats
from scipy.stats import chi2


def heatmap(df):
    corrmat = df.corr()
    top_corr_features = corrmat.index
    plt.figure(figsize=(20,20))
    g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")

def datetime_obj(df,objformat:str):
    return df.apply(lambda x : datetime.datetime.strptime(x, objformat))

def pvalues_plot(X,y):
    """
    Plotting p-values for all the univariate tests
    """
    sel = f_classif(X,y)
    p_values = pd.Series(sel[1])
    p_values.index = X.columns
    p_values.sort_values(ascending=True, inplace=True)
    # Visualize p-values
    p_values.plot.bar(figsize=(16,5), title='p-values')
    

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


def F_test(X,y):
    """
    Apply SelectKBest class to extract top 10 best features using univariate ANOVA F test (scoring based on p-values)
    """
    bestfeatures = SelectKBest(score_func=f_classif, k=10)
    fit = bestfeatures.fit(X,y)
    dfcolumns = pd.DataFrame(X.columns)
    dfscores = pd.DataFrame(fit.scores_)
    dfpvalues = pd.DataFrame(fit.pvalues_)
    #concat two dataframes for better visualization 
    featureScores = pd.concat([dfcolumns,dfscores,dfpvalues],axis=1)
    featureScores.columns = ['Variables','Score','P-values']  #naming the dataframe columns
    return featureScores


def rsquares_iter(df,y_col,important_vars):
    """
    Plotting rsquares for pairwise regression
    """
    d = {}
    for var in important_vars:
        X = df[var]
        y = df[y_col]
        model = sm.OLS(y, X).fit()
        d[var] = model.rsquared
    rcParams['figure.figsize'] = 20, 10
    plt.bar(range(len(d)), d.values(), align='center')
    plt.xticks(range(len(d)), list(d.keys()))
    plt.xticks(rotation=90)
    plt.show()
    

def randomforest(train_X, train_y,val_X,val_y):
    my_model = RandomForestClassifier(random_state=123).fit(train_X, train_y)
    pred_rf = my_model.predict_proba(val_X)
    prediction_rf = np.asarray([np.argmax(line) for line in pred_rf])
    accuracy = accuracy_score(val_y, prediction_rf)
    rms = sqrt(mean_squared_error(val_y, prediction_rf))
    feat_importances = pd.Series(my_model.feature_importances_, index=train_X.columns)
    feat_importances.nlargest(10).plot(kind='barh')
    rcParams['figure.figsize'] = 20, 10
    plt.show()


def permutation_importance(my_model,val_X, val_y):
    """
    Plotting permutation importance
    """
    perm = PermutationImportance(my_model, random_state=1).fit(val_X, val_y)
    eli5.show_weights(perm, feature_names = val_X.columns.tolist())
