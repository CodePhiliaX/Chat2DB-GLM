# Chat2DB-GLM

Languages: 中文 | [English](README.md)

## 简介

Chat2DB-GLM是[Chat2DB](https://github.com/chat2db/Chat2DB/)开源项目的组成部分，旨在提供一个高效的途径，将自然语言查询转换为结构化的SQL语句。此次开源的[Chat2DB-SQL-7B](https://huggingface.co/Chat2DB/Chat2DB-SQL-7B)模型，拥有7B参数，基于CodeLlama进行了精心微调。这一模型专为自然语言转SQL任务设计，支持多种SQL方言，并且具有高达16k的上下文长度处理能力。

## 方言支持

Chat2DB-SQL-7B模型支持广泛的SQL方言，包括但不限于Mysql、Postgres、Sqlite，以及其他通用的SQL方言。这一跨方言支持能力确保了模型的广泛适用性和灵活性。

## 模型效果

Chat2DB-SQL-7B模型在多个方言和SQL关键部分上都展现出了优异的性能。以下是模型在不同的SQL关键部分的表现概览，以通用SQL为例，基于spider数据集进行的评测结果展示了模型在处理SQL各个关键部分和各类SQL函数（如日期函数、字符串函数等）上的能力。

| 方言         | select | where | group | order | function | total |
|:-----------|:------:|:-----:|:-----:|------:|:--------:|------:|
| Generic SQL | 91.5   | 83.7  | 80.5  | 98.2  | 96.2     | 77.3  |

## 模型局限性与使用须知

Chat2DB-SQL-7B主要针对方言MySql、PostgreSQL和通用SQL进行了微调。尽管对于其他SQL方言，此模型仍可提供基本的转换能力，但在处理特定方言的特殊函数（如日期函数、字符串函数等）时，可能会出现误差。随着数据集的变化，模型的性能也可能会有所不同。

请注意，此模型主要供学术研究和学习目的使用。虽然我们努力确保模型输出的准确性，但不保证其在生产环境中的表现。使用此模型所产生的任何潜在损失，本项目及其贡献者概不负责。我们鼓励用户在使用模型时，应谨慎评估其在特定用例中的适用性。

## 模型推理

您可以通过transformers加载模型，参考如下样例代码段使用Chat2DB-SQL-7B模型，模型表现会随着prompt不同而有所不同，请尽量使用以下样例中的prompt范式。以下代码块中的model_path可以替换成你的本地模型路径。

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
model_path = "Chat2DB/Chat2DB-SQL-7B" # 此处可换成模型的本地路径
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto",trust_remote_code=True, torch_dtype=torch.float16,use_cache=True)
pipe = pipeline(  "text-generation",model=model,tokenizer=tokenizer,return_full_text=False,max_new_tokens=100)
prompt = "### Database Schema\n\n['CREATE TABLE \"stadium\" (\\n\"Stadium_ID\" int,\\n\"Location\" text,\\n\"Name\" text,\\n\"Capacity\" int,\\n\"Highest\" int,\\n\"Lowest\" int,\\n\"Average\" int,\\nPRIMARY KEY (\"Stadium_ID\")\\n);', 'CREATE TABLE \"singer\" (\\n\"Singer_ID\" int,\\n\"Name\" text,\\n\"Country\" text,\\n\"Song_Name\" text,\\n\"Song_release_year\" text,\\n\"Age\" int,\\n\"Is_male\" bool,\\nPRIMARY KEY (\"Singer_ID\")\\n);', 'CREATE TABLE \"concert\" (\\n\"concert_ID\" int,\\n\"concert_Name\" text,\\n\"Theme\" text,\\n\"Stadium_ID\" text,\\n\"Year\" text,\\nPRIMARY KEY (\"concert_ID\"),\\nFOREIGN KEY (\"Stadium_ID\") REFERENCES \"stadium\"(\"Stadium_ID\")\\n);', 'CREATE TABLE \"singer_in_concert\" (\\n\"concert_ID\" int,\\n\"Singer_ID\" text,\\nPRIMARY KEY (\"concert_ID\",\"Singer_ID\"),\\nFOREIGN KEY (\"concert_ID\") REFERENCES \"concert\"(\"concert_ID\"),\\nFOREIGN KEY (\"Singer_ID\") REFERENCES \"singer\"(\"Singer_ID\")\\n);']\n\n\n### Task \n\n基于提供的database schema信息，How many singers do we have?[SQL]\n"
response = pipe(prompt)[0]["generated_text"]
print(response)
```

## 硬件要求

| 模型           | 最低GPU显存(推理) | 最低GPU显存(高效参数微调) |
|:--------------:|:-----------------:|:-------------------------:|
| Chat2DB-SQL-7B |       14GB        |            20GB           |

## 模型下载
- huggingface：[Chat2DB-SQL-7B](https://huggingface.co/Chat2DB/Chat2DB-SQL-7B)
- modelscope：[Chat2DB-SQL-7B](https://modelscope.cn/models/Chat2DB/Chat2DB-SQL-7B/summary)

## 贡献指南
我们欢迎并鼓励社区成员对Chat2DB-GLM项目进行贡献。无论是通过报告问题、提出新功能，还是直接提交代码修复和改进，您的帮助都是非常宝贵的。

如果您有兴趣贡献，请遵循我们的贡献指南：

报告问题：通过GitHub Issues报告遇到的任何问题或错误。
提交拉取请求：如果您想直接为代码库贡献，请先fork仓库，然后提交拉取请求(PR)。
改进文档：欢迎提供最佳实践、示例代码、文档改进等。


## 许可证
本项目中的模型权重受Code Llama的自定义商业许可证约束。详情请访问：[自定义商业许可证](https://llama.meta.com/llama-downloads/)

在使用此软件之前，请确保您已充分理解许可证的条款。

