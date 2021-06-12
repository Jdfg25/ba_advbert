import pandas as pd
import matplotlib.pyplot as plt

path = '/tmp/accuracy/accuracy.txt'

with open(path, 'r') as f:
    lines = f.readlines()
    x = [float(line.split('accuracy ')[1].split('\n')[0]) for line in lines]

values = pd.Series(x, dtype=object)

f = plt.figure()
f.set_figwidth(15)

values.plot()
plt.savefig(
    fname='/tmp/accuracy/acc_plot'
)
plt.show()
