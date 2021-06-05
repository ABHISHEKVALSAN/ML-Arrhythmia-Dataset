import pandas as pd
import constants as const
from imblearn.over_sampling import SMOTE
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.metrics import confusion_matrix,classification_report
import pickle

classes = [2,3,4,5,6,7,8,9,10,14,15,16]
oversampling = SMOTE()
logreg = LogisticRegression(max_iter=5000)


for data_class in classes:
	try:
		data = pd.read_csv('data/data_class_'+str(data_class)+'.csv')
		data.drop(['J'],1,inplace=True)
		data.diagnosis.replace(1,0,inplace=True)
		data.diagnosis.replace(data_class,1,inplace=True)

		print('Number of normal cases : ',data[data.diagnosis==0].shape)
		print('Number of sick cases   : ',data[data.diagnosis==1].shape)

		data.replace('?',np.nan,inplace=True)
		data.dropna(inplace=True)

		X=data.drop(const.LABEL,1).values
		y=data[const.LABEL].values

		X_new,y_new = oversampling.fit_resample(X,y)

		X_train,X_test,y_train,y_test = train_test_split(
											X_new,
											y_new,
											test_size=0.3,
											random_state=0)
		logreg.fit(X_train,y_train)

		y_pred = logreg.predict(X_test)

		cm = confusion_matrix(y_test,y_pred)
		score = logreg.score(X_test,y_test)
		print('Confusion Matrix : ',cm)
		print('Score : ',score)
		print('Classification_report : ',classification_report(y_test,y_pred))

		pickle.dump(logreg,open('models/Model_class_'+str(data_class)+'.pkl','wb'))
	except:
		print('Avoid')
