import numpy
import random

from datasets import load_dataset

characters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
              'u', 'v', 'w', 'x', 'y', 'z', 'ä', 'ö', 'ü', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

w: int


def insert_typos(dataset, true_prob):
    global w
    w = -1

    tmp_text = []

    total_samples = len(dataset)

    for i, sample in enumerate(dataset):
        tmp_text.append(sample['text'])
        print(f'Typo: {i}/{total_samples - 1}')
        for k, char in enumerate(sample['text']):
            if numpy.random.choice(numpy.arange(0, 2), p=[1 - true_prob, true_prob]):
                tmp_text[i] = make_mistakes(random.randint(1, 5), k, tmp_text[i])

    def update_dataset(example):
        global w
        w = w + 1
        example['text'] = tmp_text[w]
        return example

    dataset = dataset.map(update_dataset)

    return dataset


def make_mistakes(choice, i, org_sample):
    # einfügen
    if choice == 1:
        tmp_sample = org_sample[:i] + \
                     characters[random.randint(0, len(characters) - 1)] + \
                     org_sample[i:]
    # löschen
    elif choice == 2:
        try:
            tmp_sample = org_sample[:i] + \
                         org_sample[i + 1:]
        except IndexError:
            tmp_sample = org_sample[:i]
    # vertauschen
    elif choice == 3:
        try:
            tmp_sample = org_sample[:i] + \
                         org_sample[i + 1] + \
                         org_sample[i] + \
                         org_sample[i + 2:]
        except IndexError:
            try:
                tmp_sample = org_sample[:i] + \
                             org_sample[i + 1] + \
                             org_sample[i]
            except IndexError:
                try:
                    tmp_sample = org_sample[:i - 1] + \
                                 org_sample[i] + \
                                 org_sample[i - 1]
                except IndexError:
                    tmp_sample = org_sample[:i]
    # austauschen
    else:
        try:
            tmp_sample = org_sample[:i] + \
                         characters[random.randint(0, len(characters) - 1)] + \
                         org_sample[i + 1:]
        except IndexError:
            tmp_sample = org_sample[:i] + \
                         characters[random.randint(0, len(characters) - 1)]

    return tmp_sample


if __name__ == '__main__':
    gnad = load_dataset(
        path='gnad10',
        split='train',
        cache_dir='/data'
    )

    new_gnad = insert_typos(gnad, 0.01)

    new_gnad.save_to_disk('/data/gnad10_with_typos')
