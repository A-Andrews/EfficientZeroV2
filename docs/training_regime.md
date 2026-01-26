# EfficientZero V2 Training Regime

This document provides a detailed explanation of how the training regime works in EfficientZero V2.

## Overview

EfficientZero V2 employs a sophisticated training regime that combines model-based reinforcement learning with Monte Carlo Tree Search (MCTS) to achieve sample-efficient learning across discrete and continuous control tasks. The training architecture uses distributed Ray-based actors to parallelize data collection, batch preparation, and model training, enabling efficient use of computational resources.

## Training Architecture

The training system consists of three main asynchronous components that communicate through a central replay buffer:

1. **Data Workers** - Perform self-play to collect trajectories using MCTS-guided policies
2. **Batch Workers** - Prepare training batches with reanalyzed targets from the latest model
3. **Training Worker** - Updates model weights using prepared batches

These components run concurrently, with data workers continuously generating new experiences while batch workers reanalyze historical data with the latest model, and the training worker optimizes the model parameters.

## Detailed Training Workflow

### 1. Self-Play Data Collection

Data workers execute environment episodes using the current policy augmented with MCTS planning:

- Each data worker maintains multiple parallel environments (`num_envs`) and a copy of the model for inference
- At every environment step, MCTS performs tree search using the model's predictions to generate improved policy targets
- The search process uses Gumbel-MCTS for discrete actions and standard MCTS for continuous actions
- Temperature scheduling controls exploration: high temperature early in training for exploration, decreasing over time for exploitation
- Trajectories are stored in the replay buffer as `GameTrajectory` objects containing: observations, actions, rewards, MCTS-generated policies, and value estimates
- Data collection is throttled to match training speed, preventing the buffer from being filled with data from outdated policies

### 2. Replay Buffer and Prioritized Experience Replay

The replay buffer stores collected trajectories and manages sampling for training:

- Trajectories are split into fixed-size segments for efficient storage and retrieval
- Prioritized Experience Replay (PER) assigns priorities to transitions based on temporal-difference (TD) error
- Higher-priority transitions (those with larger prediction errors) are sampled more frequently
- Priority alpha and beta parameters are scheduled over training to balance exploration and exploitation
- The buffer tracks total transitions for curriculum learning and manages memory efficiently by removing old trajectories when capacity is reached

### 3. Batch Preparation and Reanalysis

Batch workers prepare training data with fresh targets from the latest model:

- **Sampling**: Transitions are sampled from the replay buffer using PER priorities
- **Target Preparation**: For each sampled transition, the system computes:
  - **Value Prefixes**: Sum of discounted rewards over the next k steps (for predicting immediate cumulative rewards)
  - **Value Targets**: Bootstrapped values combining n-step returns with network value predictions, or GAE (Generalized Advantage Estimation) for policy gradient methods
  - **Policy Targets**: Originally from MCTS during self-play, but a portion is reanalyzed with the latest model
- **Reanalysis Process**: A configurable fraction of the batch (`reanalyze_ratio`, typically 100%) undergoes reanalysis:
  1. The latest model re-encodes the sampled observations to obtain fresh latent states
  2. MCTS is executed again on these states using the current model's dynamics and value predictions
  3. The improved policies from this fresh search replace the original stale policies
  4. This ensures that even old experiences contribute up-to-date learning signals
- **Off-Policy Correction**: Older transitions use shorter TD horizons to mitigate the effects of off-policy learning with stale data
- Prepared batches are pushed to a `batch_storage` queue from which the training worker consumes

### 4. Model Training Loop

The main training worker optimizes the model using prepared batches:

- **Model Architecture**: The model consists of:
  - Representation network: Encodes observations into latent states
  - Dynamics network: Predicts next latent state and immediate reward given current state and action
  - Prediction network: Outputs policy logits and value estimates from latent states
- **Training Objectives**: The model is trained with multiple losses:
  - **Value Loss**: Mean squared error between predicted and target values
  - **Reward Loss**: Cross-entropy or regression loss for immediate reward prediction
  - **Policy Loss**: Cross-entropy between predicted policy and MCTS-improved policy targets
  - **Consistency Loss**: Encourages consistency between predicted and actual next-state representations
  - **Entropy Regularization**: Prevents premature convergence to deterministic policies
- **Optimization**: Uses SGD or Adam optimizer with cosine learning rate decay and gradient clipping
- **Model Updates**: The training worker periodically distributes updated weights:
  - Every `self_play_update_interval` steps, weights are sent to data workers for self-play
  - Every `reanalyze_update_interval` steps, weights are sent to batch workers for reanalysis
  - A target model is maintained for stable value bootstrapping, updated less frequently than the main model
- **Mixed Value Targets**: The system can blend MCTS search values with network bootstrapped values, balancing model-based and model-free learning
- **Distributed Training**: Supports multi-GPU training with PyTorch DistributedDataParallel (DDP) for faster convergence

### 5. Key Design Features

**Temperature Scheduling**: Controls the exploration-exploitation tradeoff during MCTS. Early in training, high temperature encourages diverse action exploration. As training progresses, temperature decreases to exploit the learned policy.

**Value Target Mixing**: Combines MCTS search values (model-based, high variance, unbiased) with network bootstrapped values (model-free, low variance, biased) to get the best of both approaches. The mixing ratio can adapt based on training progress.

**Periodic Network Reset**: Optionally resets parts of the prediction network to avoid overfitting and maintain plasticity, though this is disabled by default.

**Asynchronous Execution**: Data collection, batch preparation, and training occur concurrently, maximizing hardware utilization and throughput.

## Training Phases

The training process can be conceptually divided into phases:

1. **Warmup Phase** (`0` to `start_transitions`): Data workers collect initial experiences until the replay buffer contains enough transitions to begin training.

2. **Active Training Phase** (`start_transitions` to `training_steps`): All components actively run - data collection continues, batches are prepared with reanalysis, and the model is trained. The model is periodically evaluated on held-out episodes.

3. **Offline Training Phase** (`training_steps` to `training_steps + offline_training_steps`): Data collection stops, but training continues on the existing buffer, extracting maximum value from collected experiences through continued reanalysis.

## Summary

The EfficientZero V2 training regime achieves sample efficiency through several key mechanisms: (1) MCTS-guided policy improvement that generates high-quality targets beyond what the network alone can produce, (2) continuous reanalysis of historical data with the latest model to avoid stale off-policy issues, (3) prioritized replay that focuses learning on high-error transitions, and (4) model-based learning that leverages learned dynamics for multi-step predictions. This combination enables the algorithm to learn effective policies from limited environment interactions, making it particularly suitable for domains where data collection is expensive or time-consuming.
