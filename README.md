# [Kaggle: LLMs - You Can't Please Them All](https://www.kaggle.com/competitions/llms-you-cant-please-them-all)

- [11th Place Write-up](https://www.kaggle.com/competitions/llms-you-cant-please-them-all/discussion/566386)

# Data
## Synthetic 
- `data/topics.json`: 137 topics (the first 3 are from comp's `test.csv`, while the others are generated)
- `data/evaluation_prompts.json`: 4 evaluation prompts using `{topic}` and `{full_text}`.

## Word Lists
- [Brown Corpus](https://www.kaggle.com/datasets/nltkdata/brown-corpus)
- [MIT 10000 Word list](https://www.mit.edu/~ecprice/wordlist.10000)

# Utility Scripts

## Requirements

(tested with poetry `2.1.1`)

```bash
poetry install --no-root
```

## Creating word list from brown corpus

```bash
poetry run python -m utils.create_word_list \
--brown_file ./data/brown.csv \
--out_file './data/filtered_brown_max_df_3_isalpha_only.txt' \
--max_df 3 \
--min_word_length 0 \
--max_word_length 100
```

## Generate Random Essays

```bash
poetry run python -m utils.generate_random_essays \
--word_list_txt ./data/filtered_brown_max_df_3_isalpha_only.txt \
--out_file ./data/random_essays.json \
--n_words 50 \
--max_char 450
```

## Check Split
Given some public LB scores, guess the possible combination for each attack and then prints some possible number of swaps needed for best improvements.

```bash
poetry run python -m utils.check_split --n_samples 300 --target_score 29.97415
```

```plaintext
total 9 combinations found!
x1=99, x2=99, x3=102, Score=29.97415274492199
x1=99, x2=100, x3=101, Score=29.97415274492199
x1=99, x2=101, x3=100, Score=29.97415274492199
x1=99, x2=102, x3=99, Score=29.97415274492199
x1=100, x2=99, x3=101, Score=29.974152744921994
x1=100, x2=101, x3=99, Score=29.97415274492205
x1=101, x2=99, x3=100, Score=29.974152744922005
x1=101, x2=100, x3=99, Score=29.97415274492205
x1=102, x2=99, x3=99, Score=29.974152744922005
...
```




