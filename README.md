# Neural Networks for Iterative Matrix Algorithms

> Teaching neural networks to perform matrix inversion and PCA using iterative algorithms like Oja's rule

This educational Jupyter notebook demonstrates how neural networks can learn to perform fundamental matrix operations through iterative algorithms. Instead of computing matrix inversion or Principal Component Analysis (PCA) directly, the notebook shows how simple neural network architectures can iteratively converge to these solutions. You'll explore Oja's rule for PCA and iterative methods for matrix inversion, bridging the gap between classical linear algebra and neural computation.

## ✨ Features

- **Iterative Matrix Inversion** — Demonstrates how neural networks can learn to invert matrices through iterative refinement rather than direct computation, showing the convergence process step-by-step.
- **Oja's Rule for PCA** — Implements Oja's learning rule, a biologically-inspired algorithm that allows a single neuron to extract the principal component from data through iterative weight updates.
- **Interactive Visualizations** — Includes matplotlib-based visualizations showing convergence behavior, error reduction over iterations, and the evolution of learned weights.
- **Educational Code Examples** — Step-by-step implementations with clear explanations, making complex concepts accessible to learners new to neural computation and iterative algorithms.

## 📦 Installation

### Prerequisites

- Python 3.7+
- Jupyter Notebook or JupyterLab (or Google Colab account)
- Basic understanding of linear algebra (matrices, vectors)
- Familiarity with Python and NumPy

### Setup

1. Clone or download this repository to your local machine
   - Get the notebook file onto your computer
2. pip install numpy matplotlib scipy
   - Install required Python packages for numerical computation and visualization
3. pip install jupyter jupyterlab
   - Install Jupyter environment to run the notebook (skip if using Google Colab)
4. jupyter lab
   - Launch JupyterLab in your browser
5. Open notebook.ipynb in the JupyterLab interface
   - Navigate to the notebook file and start exploring

## 🚀 Usage

### Running Locally with JupyterLab

Launch the notebook on your local machine using JupyterLab for full interactive experience

```
# In your terminal/command prompt:
jupyter lab notebook.ipynb

# Then in the notebook interface:
# 1. Click 'Run' -> 'Run All Cells' to execute the entire notebook
# 2. Or use Shift+Enter to run cells one at a time
```

**Output:**

```
Interactive plots showing convergence of iterative algorithms, numerical results comparing learned vs. true solutions
```

### Running on Google Colab

Use Google Colab for a cloud-based environment without local installation

```
# 1. Go to https://colab.research.google.com/
# 2. Click 'File' -> 'Upload notebook'
# 3. Upload notebook.ipynb
# 4. Run the first cell to install dependencies:
!pip install numpy matplotlib scipy

# 5. Execute cells sequentially with Shift+Enter
```

**Output:**

```
Same interactive visualizations and results as local execution, rendered in your browser
```

### Running Individual Sections

Execute specific sections to focus on particular algorithms (matrix inversion or PCA)

```
# After opening the notebook:
# 1. Read the markdown cells for context
# 2. Run setup cells (imports and helper functions)
# 3. Jump to specific sections:
#    - Matrix Inversion section for iterative inversion
#    - Oja's Rule section for PCA demonstration
# 4. Modify parameters (learning rate, iterations) to experiment
```

**Output:**

```
Targeted results for specific algorithms, allowing experimentation with hyperparameters
```

## 🏗️ Architecture

The notebook is structured as an educational progression through iterative matrix algorithms. It begins with foundational concepts, then implements iterative matrix inversion, followed by Oja's rule for PCA. Each section includes theory, implementation, visualization, and analysis. The 21 cells are organized to build understanding incrementally, with code cells alternating with explanatory markdown.

### File Structure

```
Notebook Structure:

┌─────────────────────────────────────┐
│  1. Introduction & Imports          │
│     - NumPy, Matplotlib, SciPy      │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  2. Helper Functions & Setup        │
│     - Visualization utilities       │
│     - Matrix generation functions   │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  3. Iterative Matrix Inversion      │
│     - Theory explanation            │
│     - Implementation                │
│     - Convergence visualization     │
│     - Error analysis                │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  4. Oja's Rule for PCA              │
│     - Single neuron learning        │
│     - Weight update dynamics        │
│     - Principal component extraction│
│     - Comparison with scipy PCA     │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  5. Experiments & Visualizations    │
│     - Parameter sensitivity         │
│     - Convergence plots             │
│     - Performance metrics           │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  6. Conclusions & Extensions        │
│     - Summary of findings           │
│     - Suggestions for exploration   │
└─────────────────────────────────────┘
```

### Files

- **notebook.ipynb** — Main Jupyter notebook containing all code, explanations, and visualizations for iterative matrix algorithms.

### Design Decisions

- Iterative approach chosen over direct computation to demonstrate how neural networks naturally solve problems through gradient-based updates
- Oja's rule selected as the PCA method because it's biologically plausible and shows clear connection between neural learning and linear algebra
- Visualizations included after each algorithm to make abstract mathematical concepts concrete and observable
- NumPy used for core computations to maintain clarity and educational value without deep learning framework overhead
- SciPy included for ground-truth comparisons, validating that iterative methods converge to correct solutions

## 🔧 Technical Details

### Dependencies

- **numpy** — Core numerical computing library for matrix operations, random number generation, and array manipulations
- **matplotlib** — Visualization library for creating plots showing convergence behavior, error curves, and weight evolution
- **scipy** — Scientific computing library used for ground-truth PCA and matrix operations to validate iterative results

### Key Algorithms / Patterns

- Oja's Rule: w(t+1) = w(t) + η·y(t)·(x(t) - y(t)·w(t)), a Hebbian learning rule for extracting principal components
- Iterative matrix inversion using Newton's method or gradient descent to approximate A^(-1) through successive refinements
- Power iteration method for finding dominant eigenvectors, related to PCA and Oja's rule convergence
- Gradient descent optimization for minimizing reconstruction error in neural network weight updates

### Important Notes

- Learning rates must be carefully tuned for convergence; too large causes divergence, too small slows learning significantly
- Matrix conditioning affects convergence speed - ill-conditioned matrices require more iterations or may not converge
- Oja's rule converges to the first principal component; extensions needed for multiple components
- Results are stochastic due to random initialization; running multiple times may yield slightly different convergence paths

## ❓ Troubleshooting

### Notebook cells fail with 'ModuleNotFoundError' for numpy, matplotlib, or scipy

**Cause:** Required Python packages are not installed in the current environment

**Solution:** Run 'pip install numpy matplotlib scipy' in your terminal, or add '!pip install numpy matplotlib scipy' as the first cell in the notebook

### Iterative algorithms diverge or produce NaN values

**Cause:** Learning rate is too high or matrix is ill-conditioned, causing numerical instability

**Solution:** Reduce the learning rate parameter (try values like 0.01, 0.001) or increase regularization. Check matrix condition number before running algorithms.

### Visualizations don't appear or show blank plots

**Cause:** Matplotlib backend not configured properly for Jupyter notebooks

**Solution:** Add '%matplotlib inline' or '%matplotlib notebook' at the top of the notebook after imports to enable inline plotting

### Convergence is very slow or doesn't reach expected accuracy

**Cause:** Insufficient iterations or suboptimal learning rate for the given problem

**Solution:** Increase the number of iterations (e.g., from 100 to 1000) or tune the learning rate. Monitor error plots to see if convergence is still progressing.

### Jupyter kernel crashes or runs out of memory

**Cause:** Matrix sizes too large or too many iterations stored in memory for visualization

**Solution:** Reduce matrix dimensions for experiments, or modify code to store fewer intermediate results. Restart kernel and clear outputs before re-running.

---

This README was generated to help learners understand the fascinating connection between neural networks and classical linear algebra. The notebook demonstrates that many matrix operations we typically compute directly can also be learned iteratively - a perspective that's central to understanding how neural networks solve complex problems. Experiment with the parameters, observe the convergence behavior, and gain intuition for how learning algorithms work!