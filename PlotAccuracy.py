import argparse
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        required=True
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    path = args.path + '/accuracy_eval_per_epoch.txt'

    with open(path, 'r') as f:
        lines = f.readlines()
        x = [float(line.split('accuracy ')[1].split('\n')[0]) for i, line in enumerate(lines) if i % 2]

    values = pd.Series(data=x, dtype=object)
    print(values)

    f = plt.figure()
    f.set_figwidth(15)

    values.plot()
    plt.savefig(fname=args-path + '/accuracy_eval_plot')
    plt.show()


if __name__ == '__main__':
    main()
