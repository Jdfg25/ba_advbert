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

    model = AutoModelForMaskedLM.from_pretrained(
        r'E:\Uni\9. Trimester (Bachelorarbeit)\Dateien vom Monacum One\notypos_20_custom\model'
    )
    tokenizer = AutoTokenizer.from_pretrained('dbmdz/bert-base-german-uncased')

    classifier = pipeline('fill-mask', model=model, tokenizer=tokenizer)

    if not args.typos:
        print(classifier('berlin ist die [MASK] von deutschland'))
        print('\n')
        print(classifier('mein [MASK] ist peter'))
        print('\n')
        print(classifier('ich habe diesen [MASK] geschaut und er war fantastisch'))
        print('\n')
        print(classifier('dieser film stammt aus dem [MASK] 1980'))
        print('\n')
        print(classifier('heute scheint die [MASK]'))
        print('\n')
        print(classifier('die nähe zum meer sorgt für eine [MASK] luftfeuchtigkeit'))
        print('\n')
        print(classifier('angela merkel ist eine deutsche [MASK]'))
        print('\n')
        print(classifier('Christentum, [MASK] und Judentum sind Religionen'))
        print('\n')
        print(classifier('ein automobil ist ein von einem [MASK] angetriebenes straßenfahrzeug'))
        print('\n')
        print(classifier('das alphabet besteht aus [MASK]'))
    else:
        print(classifier('berl4n ist die [MASK] von deutschland'))
        print('\n')
        print(classifier('mein [MASK] istn peter'))
        print('\n')
        print(classifier('ih habe diesen [MASK] geschaut und er war fantastisch'))
        print('\n')
        print(classifier('dieser fil mstammt aus dem [MASK] 1980'))
        print('\n')
        print(classifier('heute scheit die [MASK]'))
        print('\n')
        print(classifier('die nähe zum meer sorgt für eine [MASK] luftfeuchtiäkeit'))
        print('\n')
        print(classifier('angela merkel ist eine deutsc1he [MASK]'))
        print('\n')
        print(classifier('Christentum, [MASK] und Judentum sind Religionne'))
        print('\n')
        print(classifier('ein automobil istein von einem [MASK] angetriebenes straßenfahrzeug'))
        print('\n')
        print(classifier('das alphabet becsteht aus [MASK]'))


if __name__ == '__main__':
    main()
