# Dynamic OpenAI 颜值比较

基于 OpenAI GPT-5.2-chat 模型进行动态图片颜值比较。

## 功能

- 并行调用 OpenAI API 进行 triplet 颜值比较
- 使用 EloManager 进行分数管理和 triplet 提案
- 支持断点续传
- 记录所有响应日志

## 配置

配置文件: `config.yaml`

主要配置项:
- `data.input_dir`: 输入图片目录 (`local_data/univer_female`)
- `data.save_path`: 输出保存路径 (`local_data/openai_female`)
- `data.sample_n`: 采样数量 (300)
- `parallel.max_workers`: 最大并行 worker 数量 (15)
- `parallel.k_one_time`: 每次调用 propose_triplet 的 K 值 (5)
- `api.model`: OpenAI 模型名称 (`gpt-5.2-chat-latest`)
- `api.budget`: API 调用预算 (1800)

## 环境变量

需要在 `.env` 文件中设置:
- `LUMOS_API`: API 密钥
- `OPENAI_BASE_URL`: OpenAI API 基础 URL

## 运行

```bash
# 使用默认配置
python src/dynamic_openai/dynamic_openai.py

# 使用自定义配置
python src/dynamic_openai/dynamic_openai.py --config path/to/config.yaml
```

## 输出

所有输出保存到 `local_data/openai_female`:
- `nodes.json`: 节点信息和分数
- `comparison_logs.jsonl`: 比较日志
- `response_logs.jsonl`: API 响应日志
- `meta.json`: 元数据
- `elo_config.yaml`: EloManager 配置

## 断点续传

程序随时可以关闭和重启，只要配置保持稳定，重启后会自动从上次断点继续执行。
