# StableReSearcher

## 环境安装

### 1. 配置 docker 环境

```bash
docker compose up -d
docker compose exec -it rllm bash
```

### 2. 配置 python 环境

```bash
uv sync --extra vllm --extra gpu
uv pip install --no-deps -e verl/
```

## 运行

```bash
bash examples/search/train_search_agent.sh
```
