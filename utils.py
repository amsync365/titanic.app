from sklearn.base import  BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
import re
class PREPROCESSOR(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.ageimputer = SimpleImputer()
        self.ageimputer.fit(X[['Age']])
        return self

    def transform(self, X, y=None):
        X['Age'] = self.ageimputer.transform(X[['Age']])
        X['Cabin_Class'] = X.Cabin.fillna('M').apply(lambda x: str(x).replace(' ', '')).apply(lambda x: re.sub(r'[^a-zA-z]','',x))
        X['Cabin_num'] = X.Cabin.fillna('M').apply(lambda x: str(x).replace(' ', '')).apply(lambda x: re.sub(r'[^0-9]', '', x)).replace('',0)
        X.Embarked = X.Embarked.fillna('M')
        X = X.drop(['PassengerId', 'Name', 'Ticket','Cabin'], axis=1)
        return X

columns = ['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']