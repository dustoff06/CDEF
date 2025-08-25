# Concordance–Dispersion–Extremeness Framework (CDEF)

This repository contains the implementation, analysis, and supporting materials for the paper:

> **Balancing Phantom Concordance and Genuine Reliability in Ranking Systems: A Copula-Based Framework**

CDEF is a novel statistical approach for evaluating ranking reliability that jointly models **concordance**, **dispersion**, and **extremeness**. Using copula-based dependence structures, the framework distinguishes between *genuine consensus* and what we define as **phantom concordance**—apparent agreement that arises from systematic dependencies rather than true consensus.

---

## 📌 Key Features
- **Copula-based dependence modeling**  
  - Gumbel copula for upper-tail dependence  
  - Robustness checks with Gaussian and Clayton copulas  

- **Distributional flexibility for dispersion**  
  - Multinomial distribution (with-replacement sampling)  
  - Multivariate Hypergeometric distribution (without-replacement sampling)  

- **Simulation design for benchmarking**  
  - Synthetic datasets with controlled correlation structures  
  - Monte Carlo replications with reproducibility controls  

- **Validation metrics**  
  - Joint and conditional probability estimates  
  - Mutual information and chi-squared tests  
  - Comparisons with Kendall’s \(W\)  

---

## 📊 Empirical Application
We demonstrate CDEF using **pre-season NCAA football rankings** from four major polling organizations:

- Associated Press Poll (AP)  
- Coaches Poll  
- Congrove Computer Rankings  
- ESPN Power Index  

Figures include:  
- **3D Copula Surface Plots** – illustrating upper-tail dependence among ranking features.  
- **Contour Plots** – comparing dispersion under multinomial and hypergeometric assumptions.

---

## ⚙️ Implementation
The framework is implemented in **Python 3.11** using a custom class:

- `numpy` – numerical operations  
- `scipy` – distribution fitting, statistical tests  
- `pandas` – data handling  
- `matplotlib` – visualization  

All analysis is conducted in **Jupyter Notebooks**, ensuring transparency and reproducibility.  

> 📂 The main notebook is: **`CDEF_8_25_Copula.ipynb`**

---

## 🚀 Quick Start
Clone the repository and install required packages:

```bash
git clone https://github.com/your-username/CDEF-Copula.git
cd CDEF-Copula
pip install -r requirements.txt

📖 Citation
If you use this framework in your research, please cite:

@article{yourname2025CDEF,
  title={Balancing Phantom Concordance and Genuine Reliability in Ranking Systems: A Copula-Based Framework},
  author={Fulton, Lawrence and Merritt, Lien Lea},
  year={2025},
  journal={Under Review}
}

