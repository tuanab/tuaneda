This library is a quick way to do exploratory statistical analysis to evaluate variable relationship and importance

From your notebook environment:

    !pip install tuaneda
    
Import the library

    from tuaneda import tuanfuncs
    
Show available functions in the library:

    help(tuanfuncs)

Example on chi-square analysis of categorical variables:

    titanic = pd.read_csv('path/titanic_train.csv')
    titanic_categorical_cols = ['Pclass','Sex','SibSp','Parch','Embarked','Survived']

    df = titanic[titanic_categorical_cols]

    X = df.iloc[:,:5]
    y = df['Survived']

    b = tuaneda.functions.chi_square(X,y,0.05)
    b
    

    
