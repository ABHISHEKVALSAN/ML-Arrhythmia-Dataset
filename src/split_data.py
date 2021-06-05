import pandas as pd
import constants as const
data = pd.read_csv('data/data_with_headers.csv')
data_class_1 = data[data.diagnosis==1]
for name,grouped_data in data[data.diagnosis!=1].groupby(const.LABEL):
    data_class =grouped_data.append(data_class_1)
    #print(data_class.head())
    #print(data_class.tail())
    data_class.reset_index(drop=True,inplace=True)
    data_class.to_csv('data/data_class_'+str(name)+'.csv',index=False)
