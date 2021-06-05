import pickle
import numpy as np

def predict(input_data): #data with the all feature values [without 'J']

	classes = [2,3,4,5,6,9,10,16]
	results = []
	for data_class in classes:
		model = pickle.load(
					open('models/Model_class_'+str(data_class)+'.pkl','rb'))
		results.append(model.predict(input_data))
	final_result = max(results)
	print(final_result)
	return final_result

if __name__=='__main__':
	input_data=np.zeros((1,278))
	input_data.reshape(1,278)
	predict(input_data)
