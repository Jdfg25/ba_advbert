import pandas as pd
import matplotlib.pyplot as plt

path = r'E:\Uni\9. Trimester (Bachelorarbeit)\Dateien vom Monacum One\notypos_20_custom\loss_files\losses_eval.txt'

with open(path, 'r') as f:
    lines = f.readlines()
    x = [float(line.split('loss ')[1].split('\n')[0]) for line in lines]

values = pd.Series(x, dtype=object)

f = plt.figure()
f.set_figwidth(15)

values.plot()
plt.savefig(
    fname=r'E:\Uni\9. Trimester (Bachelorarbeit)\Dateien vom Monacum One\notypos_20_custom\loss_files\losses_eval_plot'
)
plt.show()
