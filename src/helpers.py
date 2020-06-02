import pandas as pd
import matplotlib.pyplot as plt

def plot(data):
    s = pd.Series(data)
    s.plot()
    plt.show()
