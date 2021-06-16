import pandas as pd
import matplotlib.pyplot as plt

path = r'E:\Uni\9. Trimester (Bachelorarbeit)\Dateien vom Monacum One\notypos_40_custom\accuracy_files\accuracy_train.txt'

with open(path, 'r') as f:
    lines = f.readlines()
    x = [float(line.split('accuracy ')[1].split('\n')[0]) for i, line in enumerate(lines) if i % 2]

values = pd.Series(data=x, dtype=object)
print(values)

f = plt.figure()
f.set_figwidth(15)

values.plot()
plt.savefig(
    fname=r'E:\Uni\9. Trimester (Bachelorarbeit)\Dateien vom Monacum One\notypos_40_custom\accuracy_files\accuracy_train_plot'
)
plt.show()
