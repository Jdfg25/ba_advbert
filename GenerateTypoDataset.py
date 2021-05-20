import datasets

import InsertTypos


def generate_typo_dataset(total_split_percentage, validation_split_percentage=5):
    raw_datasets = datasets.load_dataset(
        'wikipedia',
        '20200501.de',
        cache_dir='/data'
    )

    dataset_length = len(raw_datasets['train'])
    validation_samples = int(round(
        validation_split_percentage / 100 * total_split_percentage / 100 * dataset_length
    ))
    total_samples = int(round(
        total_split_percentage / 100 * dataset_length
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

    wikipedia_valid = InsertTypos.insert_typos(wikipedia_valid, 0.05)
    wikipedia_train = InsertTypos.insert_typos(wikipedia_train, 0.05)

    wikipedia_valid.save_to_disk('/data/wikipedia_with_typos/validation')
    wikipedia_train.save_to_disk('/data/wikipedia_with_typos/train')


if __name__ == '__main__':
    generate_typo_dataset(10)



