# Cart Pole Balance - PPO Training

This project implements **Proximal Policy Optimization (PPO)** to train an agent on the DeepMind Control Suite's **cartpole-swingup** task. The agent learns to swing up and balance a pole on a cart using continuous control actions.

## 📁 Project Structure

```
cart_pole_balance/
├── train_cart_pole.py      # Main training and evaluation script
├── rl_results/              # Training outputs
│   ├── logs/                # Monitor CSV logs for each seed
│   ├── models/              # Trained PPO models (.zip files)
│   ├── plots/               # Generated visualization plots
│   └── eval_results.csv     # Evaluation metrics summary
└── report/                  # LaTeX report and PDF documentation
    ├── cart_pole_report.tex # Report source
    ├── cart_pole_report.pdf # Compiled report
    └── Report.pdf           # Final report
```

## 🚀 Quick Start

### Prerequisites

Install required dependencies:

```bash
pip install "gymnasium[other]" stable-baselines3 "shimmy>=2.0" dm-control matplotlib pandas
```

### Running the Training

Execute the training script:

```bash
python train_cart_pole.py
```

This will:
1. Train PPO agents with **3 different seeds** (0, 1, 2) for reproducibility
2. Train each agent for **100,000 timesteps**
3. Evaluate all trained models on a fixed **evaluation seed (10)**
4. Generate learning curves and performance plots
5. Save models, logs, and visualizations to `rl_results/`

## 📊 Training Configuration

| Parameter | Value |
|-----------|-------|
| Environment | `dm_control/cartpole-swingup` |
| Algorithm | PPO (Actor-Critic) |
| Policy | MultiInputPolicy |
| Learning Rate | 3e-4 |
| Total Timesteps | 100,000 |
| Training Seeds | [0, 1, 2] |
| Evaluation Seed | 10 |
| Evaluation Episodes | 10 per seed |

## 📈 Generated Outputs

### Models
Trained PPO models saved in `rl_results/models/`:
- `ppo_cartpole_seed_0.zip`
- `ppo_cartpole_seed_1.zip`
- `ppo_cartpole_seed_2.zip`

### Logs
Episode-level training data in `rl_results/logs/`:
- `seed_0.monitor.csv`
- `seed_1.monitor.csv`
- `seed_2.monitor.csv`

### Plots
Visualization plots in `rl_results/plots/`:
- **`learning_curve.png`** - Training progress (mean ± std) with evaluation band
- **`evaluation_summary.png`** - Per-seed evaluation performance comparison

### Evaluation Results
Summary CSV in `rl_results/eval_results.csv` containing:
- Per-seed evaluation metrics
- Mean reward and standard deviation
- Number of evaluation episodes

## 🔬 Key Features

- **Multi-seed Training**: Trains with 3 different random seeds to assess reproducibility
- **Standardized Evaluation**: All models evaluated on the same fixed seed for fair comparison
- **Automatic Logging**: Monitor wrapper tracks episode rewards and lengths
- **Learning Curves**: Smoothed training progress with statistical aggregation
- **Performance Visualization**: Clear plots showing training dynamics and final performance

## 📝 Code Structure

The `train_cart_pole.py` script is organized into logical components:

1. **Configuration**: Environment setup and hyperparameters
2. **Utilities**: Helper functions for environment creation and data smoothing
3. **Training**: Single-seed and multi-seed training functions
4. **Evaluation**: Model evaluation with fixed seed
5. **Data Loading**: Training curve extraction from Monitor logs
6. **Plotting**: Learning curve and evaluation summary visualizations

## 🎯 Expected Results

The trained agent should:
- Successfully swing up the pole from hanging position
- Maintain balance at the top
- Achieve high episode rewards (typically 600-900 range)
- Show consistent performance across different training seeds

## 📖 Documentation

For detailed analysis and results, refer to the report in `report/cart_pole_report.pdf`.

## 🔧 Customization

To modify the training:
- Change `TOTAL_TIMESTEPS` for longer/shorter training
- Adjust `TRAIN_SEEDS` to train with different or more seeds
- Modify PPO hyperparameters in the `train_single_seed()` function
- Change smoothing window in `load_training_curves(window=10)`

## 📊 Monitoring Training

Training progress is printed to the console and includes:
- Episode rewards and lengths
- Training timesteps completed
- Model save locations
- Evaluation results with mean ± std

All artifacts are automatically saved to the `rl_results/` directory for reproducibility and analysis.
