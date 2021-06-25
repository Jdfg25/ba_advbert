import argparse
import torch

from datasets import load_dataset
from torch.utils.data.dataloader import DataLoader

from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=8,
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.model_path is None:
        args.model_path = r'E:\Uni\9. Trimester (Bachelorarbeit)\Dateien vom Monacum One\typosfine_50_custom\model'

    raw_datasets = load_dataset(
        path='gnad10',
        split='train[:50]'
    )

    tokenizer = AutoTokenizer.from_pretrained('bert-base-german-dbmdz-uncased')
    model = AutoModelForMaskedLM.from_pretrained(args.model_path)

    model.resize_token_embeddings(len(tokenizer))

    column_names = raw_datasets.column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    max_seq_length = tokenizer.model_max_length

    # Tokenize every text, then concatenate them together before splitting them in smaller parts.
    # We use `return_special_tokens_mask=True` because DataCollatorForLanguageModeling (see below) is more
    # efficient when it receives the `special_tokens_mask`.
    def tokenize_function(examples):
        return tokenizer(examples[text_column_name], return_special_tokens_mask=True)

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
    )

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of
    # max_seq_length.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        total_length = (total_length // max_seq_length) * max_seq_length
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + max_seq_length] for i in range(0, total_length, max_seq_length)]
            for k, t in concatenated_examples.items()
        }
        return result

    tokenized_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
    )

    eval_dataset = tokenized_datasets

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.eval_batch_size)

    model.eval()
    real_labels = 0
    right_preds = 0

    for step, batch in enumerate(eval_dataloader):
        print(f'Batch {step}')
        with torch.no_grad():
            outputs = model(**batch)

        for i, sent_logits in enumerate(outputs.logits):
            for j, word_logits in enumerate(sent_logits):
                if batch.labels[i][j] != -100:
                    real_labels += 1
                    if torch.argmax(word_logits) == batch.labels[i][j]:
                        right_preds += 1

        loss = outputs.loss

        with open(args.model_path + '/losses_eval.txt', 'a') as f:
            f.write(f'Batch {step} loss {loss}\n')

    accuracy = 100 * right_preds / real_labels
    with open(args.model_path + '/accuracy.txt', 'a') as f:
        f.write(f'Right Predicitons {right_preds} Masked Labels {real_labels}\n')
        f.write(f'accuracy {accuracy}\n')


if __name__ == "__main__":
    main()