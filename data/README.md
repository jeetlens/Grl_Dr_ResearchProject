# Data Directory

Place the Murarka et al. Mental Health Classification Dataset CSV files here:

- `both_train.csv`
- `both_val.csv`
- `both_test.csv`

Dataset references:

- https://github.com/amurark/mental-health-classification
- https://drive.google.com/drive/folders/11aW_fpXjA-O51uv3xYY3xj6NWGh1VYh_?usp=sharing
- https://huggingface.co/datasets/ourafla/Mental-Health_Text-Classification_Dataset

Expected Murarka columns are:

- `ID`
- `title`
- `post`
- `class_name`
- `class_id`

Expected classes are ADHD, Anxiety, Bipolar, Depression, PTSD, and None. The
CSV files and saved reports store these labels in lowercase.

The project builds `combined_text` from `title` and `post`, drops rows where
the combined text is empty, and uses `class_name` as the target label.

The default GRL setting is proxy-domain adversarial training with
`domain.strategy=text_length_bins`. It computes the median word length from the
training split only and applies that threshold to validation and test.
