# Emergence of Grounded Compositional Language in Multi-Agent Populations

## Project Overview

This project aims to replicate the findings of the paper "Emergence of Grounded Compositional Language in Multi-Agent Populations" by Mordtach et al. The goal is to observe how grounded compositional language can emerge in multi-agent systems through reinforcement learning.

## Directory Structure

```
project/
│
├── I iteration/
│   ├── main.py
│   ├── world.py
│   ├── requirements.txt
│   └── ...
│
├── II iteration/
│   ├── train.py
│   ├── environment.py
│   ├── requirements.txt
│   └── ...
│
├── III iteration/
│   ├── maddpg.py
│   ├── env.py
│   ├── buffer.py
│   ├── agent.py
│   └── ...
│
└── README.md               
```

## Library Requirements

The project requires specific versions of libraries to ensure compatibility and reproducibility. The required libraries are listed in the `requirements.txt` file for each iteration.

In general, the latest versions of famous packages like Numpy, PyTorch, PyGame, Gymansium, etc... should work fine.
The only important precaution is to install PyTorch version 1.4.0 if you want to run the III Iteration. We don't know exactly why, but if a newer version is installed the computational graph breaks. We speculate that across different versions the library changed how the computational graph is computed, leading to compatibility errors.

## Instructions

1. Clone the repository:
   ```sh
   git clone <repository-url>
   cd project
   ```

2. Create a virtual environment and activate it:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required libraries for the desired iteration:
   ```sh
   pip install -r <iteration>/requirements.txt
   ```

4. Run the main script to start experiments:
   ```sh
   python <iteration>/main.py
   ```