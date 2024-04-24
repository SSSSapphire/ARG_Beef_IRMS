import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
def Location_draw(df):
    locationResult = {}
    for index, string in enumerate(df['Scatter_Index']):
        if string not in locationResult:
            locationResult[string] = index
    print(locationResult)

    