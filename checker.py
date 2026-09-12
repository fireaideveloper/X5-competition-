import pandas as pd

train = pd.read_csv(r"C:\Users\anyon\Desktop\ds projects\x5-gradient\data\raw\train.csv")
test = pd.read_csv(r"C:\Users\anyon\Desktop\ds projects\x5-gradient\data\raw\test.csv")

print(train.shape)
print(test.shape)
print(train.columns.tolist())
print(test.columns.tolist())
