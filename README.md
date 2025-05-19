# Phy-SSM

This repository contains the code for the paper **“A Generalizable Physics-Enhanced State Space Model for Long-Term Dynamics Forecasting in Complex Environments”**, accepted at ICML 2025.  
OpenReview: [Paper Link](https://openreview.net/forum?id=9NrUIaH1sx&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICML.cc%2F2025%2FConference%2FAuthors%23your-submissions))

---

## ⚙️ Installation

**Tested environment**  
- Python 3.11  
- CUDA 11.8 or 12.1 
- PyTorch 2.2.1+cu121  
Note that other variants that support s5-pytorch should also work

1. Clone this repo and enter its root directory:  
   ```bash
   git clone https://github.com/yourusername/Phy-SSM.git
   cd Phy-SSM

2. Install **s5-pytorch** locally (do **not** use the official package):

   ```bash
   cd s5-pytorch
   pip install -e .
   cd ..
   ```

3. Install the rest of the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## 🔍 Minimal Example

See [Phy\_SSM\_example.py](Phy_SSM_example.py) for a step-by-step usage demo.

---

## 📦 Dataset Preparation

1. **Drone data**

   * Source: [ARPLaboratory/sysid-data](https://github.com/arplaboratory/data-driven-system-identification)
   * Download the `logs_crazyflie` archive.
   * Run the preprocessor:

     ```bash
     python utils/drone_data_preprocess.py
     ```
   * Preprocessed files will be saved under `data/drone_data/`.

2. **COVID-19 SIR data**

   * Preprocessed via [lisphilar/covid19-sir](https://github.com/lisphilar/covid19-sir).
   * Files already stored in `data/sir_data/`.

3. **Pendulum-friction data**

   * Generate synthetic trajectories by running:

     ```bash
     python utils/create_pendulum_data_script.py
     ```
   * Output saved in `data/pendulum_data/`.

---

## 🚀 Running Experiments

All training scripts accept the `--model` flag:

```bash
# Drone
python pissm_train.py --model drone

# COVID-19 (SIR)
python pissm_train.py --model SIR

# Pendulum-friction
python pissm_train.py --model pendulum_friction
```

If you’ve already trained a model and only want to evaluate:

1. In the script’s argument parser, ensure

   ```python
   parser.add_argument('--resume', '-r', type=bool, default=True, help='Resume from checkpoint')
   parser.add_argument('--evaluate-only', type=bool, default=True, help='Only evaluate the model locally')
   ```
2. Run:

   ```bash
   python pissm_train.py --resume True --evaluate-only True
   ```

---

## Acknowledgements

This project builds upon the [GOKU library](https://github.com/orilinial/GOKU). Thanks to the authors for their excellent work!

---

<!-- ## Citation

If you find this work useful, please cite: -->

<!-- ```bibtex
@inproceedings{yourlastname2025physsm,
  title     = {A Generalizable Physics-Enhanced State Space Model for Long-Term Dynamics Forecasting in Complex Environments},
  author    = {Your Name and Coauthor Name and ...},
  booktitle = {Proceedings of the 2025 International Conference on Machine Learning},
  year      = {2025},
  url       = {https://openreview.net/forum?id=9NrUIaH1sx}
}
``` -->
