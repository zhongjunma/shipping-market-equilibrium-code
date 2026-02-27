# Modeling the Equilibrium of Global Shipping Markets: A Cyclic Space-time Supernetwork Approach

This repository contains the official Python implementation and data instances for the paper: **Modeling the Equilibrium of Global Shipping Markets: A Cyclic Space-time Supernetwork Approach**

## Project Structure

- `column_generation.py`: Core algorithm implementation, including the Restricted Master Problem (RMP), Subproblem (SP) using cyclic path generation algorithm, and the label setting algorithm.
- `main.py`: Main entry point for running the optimization using cyclic path generation algorithm across various scenarios.
- `main_benchmark.py`: Script for running benchmark tests across various scenarios, using the label setting algorithm.
- `summary.py`: Post-processing script to collect results, calculate gaps, and generate summary tables.
- `data/`: Directory containing the instance data (Network structure, demands, capacities).
<!-- * `tables/`: (Generated) Detailed computational results and summaries. -->

## Software Requirements

- **Python 3.8+**
- **Gurobi Optimizer:** This project uses Gurobi for solving the Master Problem.

## How to Run

To execute the Column Generation algorithm for the instances:

```bash
python main.py
```

This will read instances from the `./data` directory, solve them, and save the intermediate solutions and logs.

### Run Benchmarks (Optional)

To run the specialized benchmark script:

```bash
python main_benchmark.py
```

### Generate Summary Tables (Optional)

After the solver finishes, generate the final result tables used in the paper:

```bash
python summary.py
```

<!-- ## 7. Citation

If you use this code or the data instances in your research, please cite:

```bibtex
@article{YourNameYear,
  title={Your Paper Title},
  author={Your Last Name, First Name and Others},
  journal={European Journal of Operational Research},
  year={202X},
  publisher={Elsevier}
}
``` -->