import pandas as pd
import matplotlib.pyplot as plt 
import glob

file = glob.glob("logs/*.csv")
file = file[0]

data = pd.read_csv(file)

plt.plot(data['stepcount'], data['current_temp'])
plt.show()