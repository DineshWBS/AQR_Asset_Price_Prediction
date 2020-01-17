import numpy as np
import pandas as pd


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform()
X_test = sc.transform(X_test)
