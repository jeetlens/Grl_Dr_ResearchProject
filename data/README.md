# Data Directory

Place the Murarka et al. Mental Health Classification Dataset CSV files here:

- `both_train.csv`
- `both_val.csv`
- `both_test.csv`

Expected Murarka columns are:

- `ID`
- `title`
- `post`
- `class_name`
- `class_id`

The project builds `combined_text` from `title` and `post`, drops rows where
the combined text is empty, and uses `class_name` as the target label.

The default GRL setting is proxy-domain adversarial training with
`domain.strategy=text_length_bins`. It computes the median word length from the
training split only and applies that threshold to validation and test.
