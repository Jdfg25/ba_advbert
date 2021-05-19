import datasets
import numpy
import random

characters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
              'u', 'v', 'w', 'x', 'y', 'z', 'ä', 'ö', 'ü', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

w: int


def insert_typos(dataset, true_prob):
    global w
    w = -1

    tmp_title = []
    tmp_text = []

    for i, sample in enumerate(dataset):
        print(i)
        tmp_title.append(sample['title'])
        tmp_text.append(sample['text'])
        for j, char in enumerate(sample['title']):
            if numpy.random.choice(numpy.arange(0, 2), p=[1 - true_prob, true_prob]):
                tmp_title[i] = make_mistakes(random.randint(1, 5), j, tmp_title[i])
        for k, char in enumerate(sample['text']):
            if numpy.random.choice(numpy.arange(0, 2), p=[1 - true_prob, true_prob]):
                tmp_text[i] = make_mistakes(random.randint(1, 5), k, tmp_text[i])

    def update_dataset(example):
        global w
        w = w + 1
        example['title'] = tmp_title[w]
        example['text'] = tmp_text[w]
        return example

    dataset = dataset.map(update_dataset)

    return dataset


def make_mistakes(choice, i, org_sample):
    # insert
    if choice == 1:
        tmp_sample = org_sample[:i] + \
                     characters[random.randint(0, len(characters) - 1)] + \
                     org_sample[i:]
    # delete
    elif choice == 2:
        try:
            tmp_sample = org_sample[:i] + \
                         org_sample[i + 1:]
        except IndexError:
            tmp_sample = org_sample[:i]
    # swap
    elif choice == 3:
        try:
            tmp_sample = org_sample[:i] + \
                         org_sample[i + 1] + \
                         org_sample[i] + \
                         org_sample[i + 2:]
        except IndexError:
            tmp_sample = org_sample[:i] + \
                         characters[random.randint(0, len(characters) - 1)]
    # mistype
    else:
        try:
            tmp_sample = org_sample[:i] + \
                         characters[random.randint(0, len(characters) - 1)] + \
                         org_sample[i + 1:]
        except IndexError:
            tmp_sample = org_sample[:i] + \
                         characters[random.randint(0, len(characters) - 1)]

    return tmp_sample


"""
if __name__ == '__main__':
    # wikipedia_de = datasets.load_dataset(path='wikipedia', name='20200501.de', split=f'train[:5]')
    # print(wikipedia_de[1]['text'])
    # print(wikipedia_de['title'])
    # bad_wikipedia_de = insert_typos(wikipedia_de, 0.05)
    # print(bad_wikipedia_de[1]['text'])
    # print(bad_wikipedia_de['title'])

    validation_samples = 10
    total_samples = 210

    raw_datasets = datasets.load_dataset(
        'wikipedia',
        '20200501.de',
    )

    raw_datasets["validation"] = datasets.load_dataset(
        'wikipedia',
        '20200501.de',
        split=f"train[:{validation_samples}]",
    )
    raw_datasets["train"] = datasets.load_dataset(
        'wikipedia',
        '20200501.de',
        split=f"train[{validation_samples}:{total_samples}]",
    )

    raw_datasets["validation"] = insert_typos(raw_datasets["validation"], 0.05)
    raw_datasets["train"] = insert_typos(raw_datasets["train"], 0.05)

    print(raw_datasets["validation"]["title"])
"""
