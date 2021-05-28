import argparse
from transformers import pipeline, AutoTokenizer, AutoModelForMaskedLM


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--typos",
        type=bool,
        default=False,
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if not args.typos:
        model = AutoModelForMaskedLM.from_pretrained('/model/bert-base-german-no-typos')
    else:
        model = AutoModelForMaskedLM.from_pretrained('/model/bert-base-german-typos')
    tokenizer = AutoTokenizer.from_pretrained('dbmdz/bert-base-german-uncased')

    classifier = pipeline('fill-mask', model=model, tokenizer=tokenizer)

    print(classifier('berlin ist die [MASK] von deutschland'))
    print(classifier('mein [MASK] ist peter'))
    print(classifier('ich habe diesen [MASK] geschaut und er war großartig'))
    print(classifier('dieser film stammt aus dem [MASK] 1980'))
    print(classifier('heute scheint die [MASK]'))


if __name__ == '__main__':
    main()
