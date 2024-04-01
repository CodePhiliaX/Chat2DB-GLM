# Chat2DB-GLM

Languages: 中文 | [English](README.md)

## Introduction

Chat2DB-GLM is a part of the open-source project [Chat2DB](https://github.com/chat2db/Chat2DB/), aimed at providing an efficient way to convert natural language queries into structured SQL statements. The open-sourced [Chat2DB-SQL-7B](https://huggingface.co/Chat2DB/Chat2DB-GLM-7B) model, with 7B parameters, has been fine-tuned based on CodeLlama. This model is specifically designed for the task of converting natural language to SQL, supports various SQL dialects, and is capable of handling up to 16k of context length.

## Dialect Support

The Chat2DB-SQL-7B model supports a wide range of SQL dialects, including but not limited to MySQL, PostgreSQL, SQLite, and other common SQL dialects. This cross-dialect capability ensures the model's broad applicability and flexibility.

## Model Performance

The Chat2DB-SQL-7B model has shown excellent performance across multiple dialects and key parts of SQL. Below is an overview of the model's performance on different SQL key parts, using generic SQL as an example, based on evaluations using the spider dataset, demonstrating the model's capability in handling various SQL functions (such as date functions, string functions, etc.).

| Dialect      | select | where | group | order | function | total |
|:-------------|:------:|:-----:|:-----:|:-----:|:--------:|:-----:|
| Generic SQL  | 91.5   | 83.7  | 80.5  | 98.2  | 96.2     | 77.3  |

## Model Limitations and Usage Notes

The Chat2DB-SQL-7B was mainly fine-tuned for the dialects MySQL, PostgreSQL, and generic SQL. Although the model can provide basic conversion capabilities for other SQL dialects, inaccuracies may occur when dealing with special functions of specific dialects (such as date functions, string functions, etc.). Performance may vary with changes in the dataset.

Please note that this model is primarily intended for academic research and learning purposes. While we strive to ensure the accuracy of the model's output, its performance in a production environment is not guaranteed. Any potential losses incurred from using this model are not the responsibility of this project or its contributors. We encourage users to carefully evaluate its applicability in specific use cases before use.

## Model Inference

You can load the model via transformers and use the Chat2DB-SQL-7B model with the following sample code snippet. The model's performance may vary depending on the prompt, so please try to follow the prompt format provided in the example below. The `model_path` in the code block can be replaced with your local model path.

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
model_path = "Chat2DB-GLM/Chat2DB-SQL-7B" # This can be replaced with your local model path
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True, torch_dtype=torch.float16, use_cache=True)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, return_full_text=False, max_new_tokens=100)
prompt = "### Database Schema\n\n['CREATE TABLE \"stadium\" (\\n\"Stadium_ID\" int,\\n\"Location\" text,\\n\"Name\" text,\\n\"Capacity\" int,\\n\"Highest\" int,\\n\"Lowest\" int,\\n\"Average\" int,\\nPRIMARY KEY (\"Stadium_ID\")\\n);', 'CREATE TABLE \"singer\" (\\n\"Singer_ID\" int,\\n\"Name\" text,\\n\"Country\" text,\\n\"Song_Name\" text,\\n\"Song_release_year\" text,\\n\"Age\" int,\\n\"Is_male\" bool,\\nPRIMARY KEY (\"Singer_ID\")\\n);', 'CREATE TABLE \"concert\" (\\n\"concert_ID\" int,\\n\"concert_Name\" text,\\n\"Theme\" text,\\n\"Stadium_ID\" text,\\n\"Year\" text,\\nPRIMARY KEY (\"concert_ID\"),\\nFOREIGN KEY (\"Stadium_ID\") REFERENCES \"stadium\"(\"Stadium_ID\")\\n);', 'CREATE TABLE \"singer_in_concert\" (\\n\"concert_ID\" int,\\n\"Singer_ID\" text,\\nPRIMARY KEY (\"concert_ID\",\"Singer_ID\"),\\nFOREIGN KEY (

\"concert_ID\") REFERENCES \"concert\"(\"concert_ID\"),\\nFOREIGN KEY (\"Singer_ID\") REFERENCES \"singer\"(\"Singer_ID\")\\n);']\n\n\n### Task \n\nBased on the provided database schema information, How many singers do we have?```sql\n"
response = pipe(prompt)[0]["generated_text"]
print(response)
```

## Hardware Requirements

| Model           | Minimum GPU Memory (Inference) | Minimum GPU Memory (Efficient Parameter Fine-Tuning) |
|:----------------|:-------------------------------:|:----------------------------------------------------:|
| Chat2DB-SQL-7B  |             14GB               |                         20GB                         |


## Contribution Guide
We welcome and encourage community members to contribute to the Chat2DB-GLM project. Whether it's by reporting issues, proposing new features, or directly submitting code fixes and improvements, your help is invaluable.

If you're interested in contributing, please follow our contribution guidelines:

Report Issues: Report any issues or bugs encountered via GitHub Issues.
Submit Pull Requests: If you wish to contribute directly to the codebase, please fork the repository and submit a pull request (PR).
Improve Documentation: Contributions to best practices, example code, documentation improvements, etc., are welcome.


## License
The model weights in this project are licensed under the CC BY-SA 4.0 license. You are free to use this model, but if you make modifications to it (such as fine-tuning), you must open source your modified weights under the same license terms. For specific terms, please refer to the LICENSE file.

Before using this software, please ensure you have fully understood the terms of the license.
