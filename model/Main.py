import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle
#clean data
data=pd.read_csv("C:/Users/harsh/OneDrive/Desktop/data.csv")
data=data.drop(['Unnamed: 32','id'],axis=1)
data['diagnosis']= data['diagnosis'].map({'M':1, 'B':0})
data.head()

#model

X=data.drop(['diagnosis'],axis=1)
y=data['diagnosis']

scaler=StandardScaler()
X=scaler.fit_transform(X)

X_train,X_test, y_train, y_test= train_test_split(
    X,y, test_size=0.2, random_state=42
)
#training
model= LogisticRegression()
model.fit(X_train, y_train)

#testing
y_pred=model.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Classification Report:\n', classification_report(y_test, y_pred))

# saving model and scaler

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)