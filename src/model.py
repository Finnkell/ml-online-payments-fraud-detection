from os.path import abspath
import sys
sys.path.append(abspath('./../'))


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np


df = pd.read_parquet('database/credit_card.parquet.gzip')

X = np.array(df[['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig']])
y = np.array(df['isFraud'])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

tree_model = DecisionTreeClassifier()
tree_model.fit(X_train, y_train)

print(f'{round(tree_model.score(X_test, y_test)*100, 6)}%')

y_pred = tree_model.predict(X_test)
print(f'{round(accuracy_score(y_test, y_pred)*100, 6)}%')
