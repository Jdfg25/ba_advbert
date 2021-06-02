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
    print(classifier('ich habe diesen [MASK] geschaut und er war fantastisch'))
    print(classifier('dieser film stammt aus dem [MASK] 1980'))
    print(classifier('heute scheint die [MASK]'))
    print(classifier('die nähe zum meer sorgt für eine [MASK] luftfeuchtigkeit'))
    print(classifier('angela merkel ist eine deutsche [MASK]'))
    print(classifier('Christentum, [MASK] und Judentum sind Religionen'))
    print(classifier('ein automobil ist ein von einem [MASK] angetriebenes straßenfahrzeug'))
    print(classifier('das alphabet besteht aus [MASK]'))


if __name__ == '__main__':
    main()
