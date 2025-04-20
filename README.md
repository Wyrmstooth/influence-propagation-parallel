# Parallel Influence Propagation

This project focuses on implementing parallel algorithms for influence propagation in social networks using **MPI**, **OpenMP**, and **METIS**. It addresses the computational challenges of analyzing large-scale networks by leveraging distributed and parallel computing.

## Overview

Influence propagation is widely used in:

- Viral marketing
- Epidemic modeling
- Information diffusion analysis

The project combines:

- **Graph Partitioning**: METIS for efficient graph division.
- **Distributed Computing**: MPI for communication between nodes.
- **Shared Memory Parallelism**: OpenMP for optimizing computations within nodes.

## Features

- Efficient handling of large-scale networks
- Hybrid parallelism with MPI and OpenMP
- Scalability evaluation with real-world datasets

## Usage

### Prerequisites

- **MPI** (e.g., OpenMPI)
- **METIS**
- **GCC** with OpenMP support

### Steps to Run

1. Compile the program:
   ```bash
   make
   ```
2. Run the influence propagation algorithm:
   ```bash
   mpirun -np <num_processes> ./influence_prop <dataset_path>
   ```

Example:

```bash
mpirun -np 4 ./influence_prop data/medium/twitter_network.txt
```

## Contributors

- **Naveed Ahmed** (Roll No: i220889)
- **Laraib** (Roll No: i210741)
- **Umer Jahangir** (Roll No: i210617)

## Acknowledgments

This project builds upon the foundational work in influence propagation by Kempe et al. (2003) and leverages modern tools like MPI, OpenMP, and METIS for parallelism.
d
