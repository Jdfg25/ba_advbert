import argparse
from transformers import pipeline, AutoTokenizer, AutoModelForMaskedLM


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--typos",
        type=bool,
        default=False,
    )

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default='bert-base-german-dbmdz-uncased',
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    model = AutoModelForMaskedLM.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained('dbmdz/bert-base-german-uncased')

    classifier = pipeline('fill-mask', model=model, tokenizer=tokenizer)

    if not args.typos:
        print(classifier('berlin ist die [MASK] von deutschland'))
        print(classifier('mein [MASK] ist peter'))
        print(classifier('ich habe diesen [MASK] geschaut und er war fantastisch'))
        print(classifier('dieser film stammt aus dem [MASK] 1980'))
        print(classifier('die nähe zum meer sorgt für eine [MASK] luftfeuchtigkeit'))
        print(classifier('christentum, [MASK] und judentum sind religionen'))
        print(classifier('ein automobil ist ein von einem [MASK] angetriebenes straßenfahrzeug'))
    else:
        print(classifier('berl4n ist die [MASK] von deutschland'))
        print(classifier('mein [MASK] istn peter'))
        print(classifier('ih habe diesen [MASK] geschaut und er war fantastisch'))
        print(classifier('dieser fil mstammt aus dem [MASK] 1980'))
        print(classifier('die nähe zum meer sorgt für eine [MASK] luftfeuchtiäkeit'))
        print(classifier('Christentum, [MASK] und Judentum sind Religionne'))
        print(classifier('ein automobil istein von einem [MASK] angetriebenes straßenfahrzeug'))


if __name__ == '__main__':
    main()
