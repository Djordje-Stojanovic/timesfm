# TimesFM

TimesFM (Time Series Foundation Model) is a pretrained time-series foundation
model developed by Google Research for time-series forecasting.

*   Paper: [A decoder-only foundation model for time-series forecasting](https://arxiv.org/abs/2310.10688), ICML 2024.
*   Checkpoints: [TimesFM Hugging Face Collection](https://huggingface.co/collections/google/timesfm-release-66e4be5fdb56e960c1e482a6).
*   [Google Research blog](https://research.google/blog/a-decoder-only-foundation-model-for-time-series-forecasting/).

**Model Version:** TimesFM 2.5

## Key Features

- 200M parameters (down from 500M in 2.0)
- Up to 16k context length (up from 2048)
- Continuous quantile forecast up to 1k horizon via optional 30M quantile head
- No frequency indicator needed
- Flip invariance for negative-valued time series
- Covariate support via XReg

## Install

1.  Clone the repository:
    ```shell
    git clone https://github.com/Djordje-Stojanovic/timesfm.git
    cd timesfm
    ```

2.  Create a virtual environment and install dependencies:
    ```shell
    uv venv --python 3.11
    source .venv/bin/activate       # Linux/Mac
    source .venv/Scripts/activate   # Windows/Git Bash

    uv pip install -e .[torch]
    uv pip install yfinance
    ```

3.  [Optional] Install your preferred PyTorch backend:
    - [Install PyTorch](https://pytorch.org/get-started/locally/) (CUDA recommended for GPU).

## Code Example

```python
import torch
import numpy as np
import timesfm

torch.set_float32_matmul_precision("high")

model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")

model.compile(
    timesfm.ForecastConfig(
        max_context=1024,
        max_horizon=256,
        normalize_inputs=True,
        use_continuous_quantile_head=True,
        force_flip_invariance=True,
        infer_is_positive=True,
        fix_quantile_crossing=True,
    )
)
point_forecast, quantile_forecast = model.forecast(
    horizon=12,
    inputs=[
        np.linspace(0, 1, 100),
        np.sin(np.linspace(0, 20, 67)),
    ],
)
point_forecast.shape  # (2, 12)
quantile_forecast.shape  # (2, 12, 10): mean, then 10th to 90th quantiles.
```

## NVIDIA Prediction Test

```shell
python predict_nvidia.py
```

Predicts next-day NVDA close price using 5 years of historical data. See `CLAUDE.md` for details.
