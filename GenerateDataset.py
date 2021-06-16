import argparse
import datasets

import CleanDataset
import InsertTypos


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--total_split_percentage",
        type=float,
        default=100,
    )
    parser.add_argument(
        "--validation_split_percentage",
        type=float,
        default=5,
    )
    parser.add_argument(
        "--insert_typos",
        type=int,
        default=0,
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    wikipedia = datasets.load_dataset(
        'wikipedia',
        '20200501.de',
        cache_dir='/data'
    )

    dataset_length = len(wikipedia['train'])
    validation_samples = int(round(
        args.validation_split_percentage / 100 * args.total_split_percentage / 100 * dataset_length
    ))
    total_samples = int(round(
        args.total_split_percentage / 100 * dataset_length
    ))

    wikipedia_valid = datasets.load_dataset(
        'wikipedia',
        '20200501.de',
        split=f"train[:{validation_samples}]",
        cache_dir='/data'
    )
    wikipedia_train = datasets.load_dataset(
        'wikipedia',
        '20200501.de',
        split=f"train[{validation_samples}:{total_samples}]",
        cache_dir='/data'
    )

    wikipedia_valid = CleanDataset.clean_dataset(wikipedia_valid)
    wikipedia_train = CleanDataset.clean_dataset(wikipedia_train)

    if args.insert_typos == 0:
        # clean dataset
        wikipedia_valid.save_to_disk('/data/wikipedia_clean/validation')
        wikipedia_train.save_to_disk('/data/wikipedia_clean/train')
    elif args.insert_typos == 1:
        # dataset with typos
        wikipedia_valid = InsertTypos.insert_typos(wikipedia_valid, 0.01, False)
        wikipedia_train = InsertTypos.insert_typos(wikipedia_train, 0.01, False)

        wikipedia_valid.save_to_disk('/data/wikipedia_with_typos/validation')
        wikipedia_train.save_to_disk('/data/wikipedia_with_typos/train')
    else:
        # dataset with one third clean and two thirds typos
        wikipedia_valid = InsertTypos.insert_typos(wikipedia_valid, 0.001, True)
        wikipedia_train = InsertTypos.insert_typos(wikipedia_train, 0.001, True)

        wikipedia_valid.save_to_disk('/data/wikipedia_with_typos_alt/validation')
        wikipedia_train.save_to_disk('/data/wikipedia_with_typos_alt/train')


if __name__ == '__main__':
    main()
