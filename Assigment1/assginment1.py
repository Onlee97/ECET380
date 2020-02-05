import pickle
import pandas as pd

# with open('appml-assignment1-dataset.pkl', 'rb') as f:
#     data = pickle.load(f)

# print(data)

# X = data
# print(X)
data = pd.read_pickle('appml-assignment1-dataset.pkl')
print(data)

y = data['y']
print(y)

X = data['X'][['CAD-open', 'CAD-high', 'CAD-low', 'CAD-close']]
print(X)