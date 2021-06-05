import pandas as pd
from constants import HEADERS
data = pd.read_csv('data/data.csv',sep=';')
data.columns=HEADERS
data.to_csv('data/data_with_headers.csv',index=False)
