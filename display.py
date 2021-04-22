def display(zoom=False):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt 
    import glob
    #%matplotlib inline

    files = glob.glob("logs/*")
    for file in files:
        #if "Cycles.1" in file and "Temp.65" in file:
        if "Tuning01_P005_I3_D0_" in file:
            data = pd.read_csv(file)
            print(np.max(data['current_temp']))
            data.replace("False", "0", inplace=True)
            data.replace("True", "1", inplace=True)
            data.set_index("stepcount", inplace=True)
            plt.figure(figsize=(20,5))
            fig, ax = plt.subplots()
            if zoom:
                end, _ = data.shape
                ax.set_xlim([end-100, end])
            ax.plot(data.index, data['current_temp'], c='black', 
                    label="Actual temperature")
            ax.plot(data.index, data['target_temp'], c='black', 
                    label="Target temperature", linestyle=':', 
                    alpha=0.4)
            ax.set_xlabel(file)
            ax.set_ylabel("Temperature")
            ax.tick_params(axis='x', labelrotation = 90)
            plt.legend(bbox_to_anchor=(0.4,1.3))
            ax2 = ax.twinx()
            ax2.plot(data.index, data['outcome'], c='g', 
                    label="Drive Output", alpha=0.75, linestyle='--')
            ax2.plot(data.index, data['Ival'], c='b', 
                    label="I VAL", alpha=0.75, linestyle='--')
            ax2.plot(data.index, data['Dval'], c='r', 
                    label="D VAL", alpha=0.75, linestyle='--')
            ax2.set_ylabel("IVal")
            plt.legend(bbox_to_anchor=(1.1,1.3))
            plt.show()
            