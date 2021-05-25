import datasets

w: int


def clean_dataset(dataset):
    global w
    w = -1

    tmp = []

    for i, sample in enumerate(dataset):
        print(f'Clean: {i}')
        tmp.append(sample['text'])

        try:
            tmp[i] = tmp[i].split("Literatur")[0]
        except NameError:
            pass

        try:
            tmp[i] = tmp[i].split("Weblinks")[0]
        except NameError:
            pass

    def update_dataset(example):
        global w
        w = w + 1
        example['text'] = tmp[w]
        return example

    dataset = dataset.map(update_dataset)

    return dataset


"""
if __name__ == '__main__':
    wikipedia = datasets.load_dataset(path='wikipedia', name='20200501.de', split=f'train[:10%]')
    print(wikipedia_de[0]['text'])
    clean_wikipedia = clean_dataset(wikipedia)
    print(clean_wikipedia_de[0]['text'])
    clean_wikipedia.save_to_disk('/data/wikipedia_with_typos')
"""




