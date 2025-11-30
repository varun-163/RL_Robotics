# RL_Robotics

Reinforcement Learning projects for robotics applications, focusing on continuous control and policy optimization algorithms.

## 📁 Repository Structure

```
RL_Robotics/
├── cart_pole_balance/           # PPO on DeepMind Control Suite Cart-Pole Swingup
│   ├── train_cart_pole.py       # Main training and evaluation script
│   ├── rl_results/              # Training outputs and artifacts
│   │   ├── logs/                # Episode logs for each seed
│   │   ├── models/              # Trained PPO models (.zip files)
│   │   ├── plots/               # Learning curves and evaluation plots
│   │   └── eval_results.csv     # Evaluation metrics summary
│   ├── report/                  # LaTeX report and documentation
│   │   ├── cart_pole_report.tex # Report source
│   │   └── cart_pole_report.pdf # Compiled report
│   └── README.md                # Project-specific documentation
├── LICENSE
└── README.md                    # This file
```

## 🚀 Projects

### 1. Cart-Pole Balance (PPO)

**Algorithm:** Proximal Policy Optimization (PPO)  
**Environment:** DeepMind Control Suite - `cartpole-swingup`  
**Framework:** Stable-Baselines3, Gymnasium

Implementation of an actor-critic agent using PPO to solve the continuous control cart-pole swingup task. The agent learns to swing up a pole from a hanging position and balance it upright.

**Key Features:**
- Multi-seed training (seeds 0, 1, 2) for reproducibility analysis
- Standardized evaluation on fixed seed (10)
- Automated logging and visualization
- Learning curves with statistical aggregation
- Comprehensive performance analysis

**Quick Start:**
```bash
cd cart_pole_balance
pip install "gymnasium[other]" stable-baselines3 "shimmy>=2.0" dm-control matplotlib pandas
python train_cart_pole.py
```

See [`cart_pole_balance/README.md`](cart_pole_balance/README.md) for detailed documentation.

## 📊 Results

The cart-pole PPO agent achieves:
- Successful pole swingup and balancing
- Mean evaluation reward: ~467 (±224 across seeds)
- Training converges in ~100,000 timesteps
- Generated learning curves and per-seed evaluation summaries

## 🛠️ Technologies

- **Python 3.x**
- **Stable-Baselines3** - RL algorithm implementations
- **Gymnasium** - Environment API
- **DeepMind Control Suite** - Physics-based control tasks
- **Matplotlib** - Visualization
- **Pandas** - Data analysis
- **LaTeX** - Report generation

## 📖 Course Information

**Course:** EEE598 - Reinforcement Learning in Robotics  
**Institution:** Arizona State University  
**Academic Year:** 2025

## 🔗 Resources

- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [DeepMind Control Suite](https://github.com/deepmind/dm_control)
- [PPO Paper](https://arxiv.org/abs/1707.06347)

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Varun Karthik**  
ASU ID: 1234608702

---

*Repository maintained as part of coursework for EEE598 - Reinforcement Learning in Robotics*
