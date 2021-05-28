w: int


def clean_dataset(dataset):
    global w
    w = -1

    tmp = []

    total_samples = len(dataset)

    for i, sample in enumerate(dataset):
        print(f'Clean: {i}/{total_samples-1}')
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
