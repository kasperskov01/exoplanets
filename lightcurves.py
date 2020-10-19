# $ conda activate sop38
import matplotlib.pyplot as plt
import pandas as pd
print("loading data...")
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_colwidth', -1)

data = pd.read_table('lightcurves/UID_0003093_RVC_008.tbl', comment='#')

print(data.head(100))

data = pd.read_table('lightcurves/UID_0003093_RVC_008.tbl', delim_whitespace=True, sep=",", header=None, skiprows=22)
data.columns = ["time", "velocity", "velocity_error"]

# print(data.head(100))


data.sort_values(by='time', axis=0)
print(data)
data.plot(kind='scatter', x=0, y=1, title='Volocity/time')
plt.show()
