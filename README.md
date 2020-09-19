This library is a quick way to do exploratory statistical analysis to evaluate variable relationship and importance

From your notebook environment:

    !pip install tuaneda

Show available functions in the library:

    help(tuaneda.functions)

Example:

    titanic = pd.read_csv('path/titanic_train.csv')
    titanic_categorical_cols = ['Pclass','Sex','SibSp','Parch','Embarked','Survived']

    df = titanic[titanic_categorical_cols]

    X = df.iloc[:,:5]
    y = df['Survived']

    b = tuaneda.functions.chi_square(X,y,0.05)
    b
    

    
