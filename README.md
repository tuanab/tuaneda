This library is a quick way to do exploratory statistical analysis to evaluate variable relationship and importance

From your notebook environment:

    !pip install tuaneda
    
Import the library

    from tuaneda import tuanfuncs
    
Show available functions in the library:

    help(tuanfuncs)

Example on information value:

    from sklearn.datasets import load_iris
    iris = load_iris()
    iris_data = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                         columns= iris['feature_names'] + ['target'])

    X = iris_data.iloc[:,:5]
    y = iris_data['target']

    woe_dict, iv_dict = tuanfuncs.woe_iv_continuous(X,y,True)
    iv_graph = tuanfuncs.barchart_dict(iv_dict) 
    

    

