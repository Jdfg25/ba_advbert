import datasets
import numpy
import random

characters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
              'u', 'v', 'w', 'x', 'y', 'z', 'ä', 'ö', 'ü', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


def insert_typos(dataset_len, true_prob):
    wikipedia_de = datasets.load_dataset(path='wikipedia', name='20200501.de', split=f'train[0:{dataset_len}]')
    print(wikipedia_de[0]['text'])

    for i, sample in enumerate(wikipedia_de):
        tmp_title = sample['title']
        tmp_text = sample['text']
        for j, char in enumerate(sample['title']):
            if numpy.random.choice(numpy.arange(0, 2), p=[1 - true_prob, true_prob]):
                tmp_title = make_mistakes(random.randint(1, 5), j, tmp_title)
        for k, char in enumerate(sample['text']):
            if numpy.random.choice(numpy.arange(0, 2), p=[1 - true_prob, true_prob]):
                tmp_text = make_mistakes(random.randint(1, 5), k, tmp_text)

    wikipedia_de = wikipedia_de.map(lambda example: {'title': tmp_title, 'text': tmp_text})

    return wikipedia_de


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


if __name__ == '__main__':
    bad_wikipedia_de = insert_typos(1, 0.05)
    print(bad_wikipedia_de[0]['text'])
