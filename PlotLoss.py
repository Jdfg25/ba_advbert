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

    parser.add_argument(
        "--eval",
        type=bool,
        default=False,
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.eval:
        path_part = 'eval'
    else:
        path_part = 'train'

    path = args.path + '/losses_' + path_part + '.txt'

    with open(path, 'r') as f:
        lines = f.readlines()
        x = [float(line.split('loss ')[1].split('\n')[0]) for line in lines]

    values = pd.Series(x, dtype=object)
    print(values)

    f = plt.figure()
    f.set_figwidth(15)

    values.plot()
    plt.savefig(fname=args.path + '/losses_' + path_part + '_plot')
    plt.show()
