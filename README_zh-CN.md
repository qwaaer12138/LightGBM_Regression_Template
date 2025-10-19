# LightGBM 回归模板

该仓库提供了一个轻量级模板，用于训练 LightGBM 回归模型，支持 k 折交叉验证，并在保留的测试集上执行推理。

## 项目结构

```
├── scripts/
│   └── run_pipeline.py   # 完整流程的命令行入口
├── src/
│   └── lightgbm_regression/
│       ├── __init__.py   # 公共 API
│       ├── _typing.py    # 共享类型别名
│       ├── data_utils.py # 数据集加载与预处理辅助函数
│       ├── infer.py      # 推理相关辅助函数
│       └── train.py      # 交叉验证训练循环
├── outputs/              # 预测产物（自动创建）
├── logs/                 # 训练日志（自动创建）
└── README.md
```

## 环境依赖

使用 pip 安装必要依赖：

```bash
pip install lightgbm pandas scikit-learn
```

## 使用方法

通过命令行脚本运行完整流程。需要提供训练集和测试集的 CSV 路径、目标列名称以及可选的配置参数。

```bash
python scripts/run_pipeline.py \
    --train-path /path/to/df_train.csv \
    --test-path /path/to/df_test.csv \
    --target-column target
```

主要命令行参数包括：

- `--drop-columns`：训练前需要从两个数据集中同时删除的列。
- `--output-dir`：保存预测结果（`predictions.csv`）和交叉验证汇总（`cv_results.json`）的目录（默认：`outputs/`）。
- `--log-dir`：保存带时间戳日志文件的目录（默认：`logs/`）。
- `--n-splits`：交叉验证的折数（默认：`5`）。
- `--learning-rate`、`--num-leaves`、`--max-depth`、`--n-estimators`、`--subsample`、
  `--colsample-bytree`、`--reg-alpha`、`--reg-lambda`：LightGBM 超参数。
- `--early-stopping-rounds`：提前停止的耐心轮数（默认：`100`）。
- `--log-level`：日志输出等级（默认：`INFO`）。

脚本会在控制台打印每一折的 MAE 指标，并同步保存日志到 `logs/` 目录下的文件中；同时在指定的 `output` 目录下写入交叉验证指标汇总 JSON，以及将测试集预测结果写入 `outputs/predictions.csv`。

### 作为 Python 库调用

该工具集同样可以在 Python 代码（脚本、notebook 或其他包）中直接导入使用。
`train_regression_model` 支持传入内存中的 pandas DataFrame 或者 CSV 文件路径，
并返回包含最佳模型与每折指标的 `TrainingResult` 对象。

```python
from lightgbm_regression import (
    TrainingConfig,
    generate_predictions,
    train_regression_model,
)
import pandas as pd

train_df = pd.read_csv("/path/to/train.csv")
test_df = pd.read_csv("/path/to/test.csv")

result = train_regression_model(
    train_data=train_df,
    target_column="target",
    test_data=test_df,
    drop_columns=["id"],
    config=TrainingConfig(n_splits=3, random_state=7),
    lgbm_params={"learning_rate": 0.1, "num_leaves": 63},
)

predictions = generate_predictions(
    result.best_model,
    test_df,
    feature_columns=result.feature_columns,
)
```

如果数据已经以 CSV 形式存放在磁盘上，也可以直接传入文件路径，
例如 `train_regression_model(train_data="train.csv", test_data="test.csv", ...)`
而无需手动加载到内存。

#### 在 Jupyter Notebook 中调用

如果希望在外部 `ipynb` 笔记本中重用训练/预测逻辑，可以先把仓库根目录加入 `sys.path`，
然后导入 `lightgbm_regression` 包暴露的 API。以下示例展示了如何在 notebook 中加载
项目自带的 Boston Housing 数据集，执行训练并生成预测：

```python
from pathlib import Path
import sys
import pandas as pd

repo_root = Path("/path/to/LightGBM_Regression_Template")
sys.path.append(str(repo_root / "src"))

from lightgbm_regression import (
    TrainingConfig,
    train_regression_model,
    generate_predictions,
)

train_df = pd.read_csv(repo_root / "tests" / "data" / "boston_housing_train.csv")
test_df = pd.read_csv(repo_root / "tests" / "data" / "boston_housing_test.csv")

result = train_regression_model(
    train_data=train_df,
    target_column="MEDV",
    test_data=test_df,
    drop_columns=["CHAS"],
    config=TrainingConfig(n_splits=5, random_state=42),
    lgbm_params={"learning_rate": 0.05, "num_leaves": 31},
)

pred_df = generate_predictions(
    model=result.best_model,
    data=test_df,
    feature_columns=result.feature_columns,
)
pred_df.head()
```

如果更偏向使用磁盘上的 CSV 文件，可以直接把 `train_data` 与 `test_data` 参数替换为相应的路径，
或借助 `load_datasets` / `split_features_and_target` 等辅助函数来完成读取与特征列对齐。

## 示例：在 UCI Boston Housing 数据集上运行

要在公开数据集上体验完整流程，可使用 `examples/` 目录下的辅助脚本。该脚本会下载 [Boston Housing](https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv) 数据集（公共领域），构建可复现的训练/测试划分，并执行 LightGBM 流程。

```bash
python examples/run_on_public_dataset.py
```

脚本会在控制台输出进度，并将产物写入 `examples/data/` 与 `examples/outputs/boston_housing/`：

- `examples/data/` —— 缓存的原始数据集、训练集、测试集以及保留的真实标签。
- `examples/outputs/boston_housing/cv_results.json` —— 流程生成的交叉验证 MAE 汇总。
- `examples/outputs/boston_housing/predictions.csv` —— 对保留测试集的预测结果。
- `examples/outputs/boston_housing/holdout_metrics.json` —— 基于保留标签计算的 MAE 指标。
- `examples/outputs/boston_housing/holdout_predictions.csv` —— 包含目标值与预测值的汇总表。
- `examples/outputs/boston_housing/logs/` —— 带时间戳的训练日志。

可以通过 `--force-download` 重新下载数据集，或使用 `--output-dir` / `--data-dir` 自定义产物存放路径。
