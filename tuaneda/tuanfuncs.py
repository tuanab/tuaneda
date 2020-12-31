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
# from sklearn.metrics import mean_squared_error
import scipy.stats as stats
from scipy.stats import chi2


def heatmap(df):
    corrmat = df.corr()
    top_corr_features = corrmat.index
    plt.figure(figsize=(20,20))
    g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")

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
    

# def quick_rf_importance(train_X, train_y,val_X,val_y):
#     my_model = RandomForestClassifier(random_state=123).fit(train_X, train_y)
#     pred_rf = my_model.predict_proba(val_X)
#     prediction_rf = np.asarray([np.argmax(line) for line in pred_rf])
#     accuracy = accuracy_score(val_y, prediction_rf)
#     rms = np.sqrt(mean_squared_error(val_y, prediction_rf))
#     feat_importances = pd.Series(my_model.feature_importances_, index=train_X.columns)
#     feat_importances.nlargest(10).plot(kind='barh')
#     rcParams['figure.figsize'] = 20, 10
#     plt.show()

def barchart_dict(d):
    d = dict(sorted(d.items(), key=lambda item: item[1]))
    rcParams['figure.figsize'] = 20, 10
    plt.bar(range(len(d)), d.values(), align='center')
    plt.xticks(range(len(d)), list(d.keys()))
    plt.xticks(rotation=90)
    plt.show()

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

        # df_woe_iv = (pd.crosstab(df[col],df['target'],
        #                   normalize='columns')
        #          .assign(woe=lambda dfx: np.log(dfx[1] / dfx[0]))
        #          .assign(iv=lambda dfx: np.sum(dfx['woe']*
        #                                        (dfx[1]-dfx[0]))))
        # df_woe_iv = df_woe_iv.replace([np.inf, -np.inf], np.nan)
        # df_woe_iv.dropna(inplace=True)
        # woe_dict[col] = df_woe_iv
        # IV_dict[col] = df_woe_iv['iv'].sum()

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

        # df_woe_iv = (pd.crosstab(df[col],df['target'],
        #                   normalize='columns')
        #          .assign(woe=lambda dfx: np.log(dfx[1] / dfx[0]))
        #          .assign(iv=lambda dfx: np.sum(dfx['woe']*
        #                                        (dfx[1]-dfx[0]))))
        # df_woe_iv = df_woe_iv.replace([np.inf, -np.inf], np.nan)
        # df_woe_iv.dropna(inplace=True)
        # woe_dict[col] = df_woe_iv
        # IV_dict[col] = df_woe_iv['iv'].sum()

    return woe_dict, IV_dict


