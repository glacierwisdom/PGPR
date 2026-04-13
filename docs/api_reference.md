# PGPR API Reference

This document provides detailed API reference for the PGPR library, including all main modules, classes, and functions.

## Table of Contents

1. [Models](#models)
   - [ESMEncoder](#esmencoder)
   - [NodeRepresentationExtractor](#noderepresentationextractor)
   - [TargetConditionalAttention](#targetconditionalattention)
   - [GNN_PPI](#gnn_ppi)
   - [SimilarityMatcher](#similaritymatcher)
   - [COTGenerator](#cotgenerator)
   - [MultiScaleRewardCalculator](#multiscalerewardcalculator)
   - [PPORLFramework](#pporlframework)

2. [Data Processing](#data-processing)
   - [PPIDataset](#ppidataset)
   - [ProteinPreprocessor](#proteinpreprocessor)
   - [DatasetSplitter](#datasetsplitter)
   - [BlastpUtils](#blastputils)

3. [Training](#training)
   - [TrainingPipeline](#trainingpipeline)
   - [RewardCalculator](#rewardcalculator)
   - [PPOAgent](#ppoagent)

4. [Inference](#inference)
   - [InferencePipeline](#inferencepipeline)

5. [Utilities](#utilities)
   - [Optimizer](#optimizer)
   - [Metrics](#metrics)
   - [Debugging](#debugging)

## Models

### ESMEncoder

```python
from models.esm_encoder import ESMEncoder
```

**Description**: Encodes protein sequences using ESM (Evolutionary Scale Modeling) language models.

#### Constructor

```python
esm_encoder = ESMEncoder(
    model_name: str = "facebook/esm2_t33_650M_UR50D",
    device: Optional[Union[str, torch.device]] = None,
    max_length: int = 1024
)
```

**Parameters**:
- `model_name`: Name of the ESM model to use
- `device`: Device for computation ("cuda" or "cpu")
- `max_length`: Maximum sequence length for encoding

#### Methods

##### `encode(sequence: str) -> torch.Tensor`
Encodes a single protein sequence.

**Parameters**:
- `sequence`: Protein amino acid sequence

**Returns**:
- `torch.Tensor`: Encoded protein representation (shape: [hidden_size])

##### `batch_encode(sequences: List[str]) -> torch.Tensor`
Encodes multiple protein sequences in batch.

**Parameters**:
- `sequences`: List of protein sequences

**Returns**:
- `torch.Tensor`: Batch of encoded representations (shape: [batch_size, hidden_size])

### NodeRepresentationExtractor

```python
from models.gnn_ppi import NodeRepresentationExtractor
```

**Description**: Extracts node representations from protein embeddings.

#### Constructor

```python	extractor = NodeRepresentationExtractor(
    input_size: int,
    hidden_size: int,
    output_size: int,
    dropout: float = 0.1
)
```

**Parameters**:
- `input_size`: Input feature size (ESM embedding size)
- `hidden_size`: Hidden layer size
- `output_size`: Output representation size
- `dropout`: Dropout probability

#### Methods

##### `forward(x: torch.Tensor) -> torch.Tensor`
Extracts node representations.

**Parameters**:
- `x`: Input features (shape: [batch_size, input_size])

**Returns**:
- `torch.Tensor`: Extracted node representations (shape: [batch_size, output_size])

### TargetConditionalAttention

```python
from models.gnn_ppi import TargetConditionalAttention
```

**Description**: Attention mechanism that conditions on target protein information.

#### Constructor

```python
attention = TargetConditionalAttention(
    node_feature_size: int,
    target_feature_size: int,
    num_heads: int = 4,
    dropout: float = 0.1
)
```

**Parameters**:
- `node_feature_size`: Node feature dimension
- `target_feature_size`: Target protein feature dimension
- `num_heads`: Number of attention heads
- `dropout`: Dropout probability

#### Methods

##### `forward(node_features: torch.Tensor, target_features: torch.Tensor, adj: torch.Tensor) -> torch.Tensor`
Applies target-conditional attention.

**Parameters**:
- `node_features`: Node features (shape: [batch_size, num_nodes, node_feature_size])
- `target_features`: Target protein features (shape: [batch_size, target_feature_size])
- `adj`: Adjacency matrix (shape: [batch_size, num_nodes, num_nodes])

**Returns**:
- `torch.Tensor`: Attended node features (shape: [batch_size, num_nodes, node_feature_size])

### GNN_PPI

```python
from models.gnn_ppi import GNN_PPI
```

**Description**: Graph Neural Network for Protein-Protein Interaction prediction.

#### Constructor

```python
gnn_ppi = GNN_PPI(
    node_feature_size: int,
    gnn_hidden_size: int,
    num_gnn_layers: int,
    num_attention_heads: int,
    dropout: float = 0.1
)
```

**Parameters**:
- `node_feature_size`: Node feature dimension
- `gnn_hidden_size`: GNN hidden layer size
- `num_gnn_layers`: Number of GNN layers
- `num_attention_heads`: Number of attention heads
- `dropout`: Dropout probability

#### Methods

##### `forward(node_features: torch.Tensor, adj: torch.Tensor) -> torch.Tensor`
Performs forward pass through the GNN.

**Parameters**:
- `node_features`: Node features (shape: [batch_size, num_nodes, node_feature_size])
- `adj`: Adjacency matrix (shape: [batch_size, num_nodes, num_nodes])

**Returns**:
- `torch.Tensor`: PPI prediction (shape: [batch_size, 1])

### SimilarityMatcher

```python
from models.gnn_ppi import SimilarityMatcher
```

**Description**: Matches proteins based on similarity in the embedding space.

#### Constructor

```python
matcher = SimilarityMatcher(
    feature_size: int,
    num_candidates: int = 10,
    similarity_threshold: float = 0.7
)
```

**Parameters**:
- `feature_size`: Feature dimension for similarity calculation
- `num_candidates`: Number of candidate proteins to consider
- `similarity_threshold`: Threshold for similarity matching

#### Methods

##### `find_similar_proteins(query_features: torch.Tensor, candidate_features: torch.Tensor) -> Dict[str, Any]`
Finds similar proteins to the query.

**Parameters**:
- `query_features`: Query protein features (shape: [feature_size])
- `candidate_features`: Candidate protein features (shape: [num_candidates, feature_size])

**Returns**:
- `Dict`: Dictionary containing similar proteins and their similarity scores

### COTGenerator

```python
from models.cot_generator import COTGenerator
```

**Description**: Generates Chain-of-Thought (COT) explanations for PPI predictions.

#### Constructor

```python
cot_generator = COTGenerator(
    llm_model: str = "gpt2",
    max_length: int = 100,
    temperature: float = 0.7,
    device: Optional[Union[str, torch.device]] = None
)
```

**Parameters**:
- `llm_model`: Name of the language model to use
- `max_length`: Maximum length of generated COT
- `temperature`: Generation temperature
- `device`: Device for computation

#### Methods

##### `generate_cot(protein1_emb: torch.Tensor, protein2_emb: torch.Tensor, predicted_label: int) -> str`
Generates a COT explanation for a PPI prediction.

**Parameters**:
- `protein1_emb`: Embedding of first protein
- `protein2_emb`: Embedding of second protein
- `predicted_label`: Predicted PPI label

**Returns**:
- `str`: Generated COT explanation

### MultiScaleRewardCalculator

```python
from models.rl_framework import MultiScaleRewardCalculator
```

**Description**: Calculates rewards at multiple scales for reinforcement learning.

#### Constructor

```python
reward_calculator = MultiScaleRewardCalculator(
    sequence_scale_weight: float = 0.3,
    structure_scale_weight: float = 0.4,
    interaction_scale_weight: float = 0.3,
    similarity_threshold: float = 0.7
)
```

**Parameters**:
- `sequence_scale_weight`: Weight for sequence-level rewards
- `structure_scale_weight`: Weight for structure-level rewards
- `interaction_scale_weight`: Weight for interaction-level rewards
- `similarity_threshold`: Threshold for similarity-based rewards

#### Methods

##### `calculate_reward(predictions: torch.Tensor, labels: torch.Tensor, protein_embeddings: torch.Tensor) -> torch.Tensor`
Calculates multi-scale rewards.

**Parameters**:
- `predictions`: Model predictions
- `labels`: Ground truth labels
- `protein_embeddings`: Protein embeddings for similarity calculation

**Returns**:
- `torch.Tensor`: Calculated rewards

### PPORLFramework

```python
from models.rl_framework import PPORLFramework
```

**Description**: Proximal Policy Optimization (PPO) framework for RL-based model optimization.

#### Constructor

```python
ppo_framework = PPORLFramework(
    model: nn.Module,
    learning_rate: float = 1e-5,
    ppo_epochs: int = 4,
    clip_epsilon: float = 0.2,
    value_loss_coef: float = 0.5,
    entropy_coef: float = 0.01,
    max_grad_norm: float = 0.5
)
```

**Parameters**:
- `model`: Model to optimize with PPO
- `learning_rate`: Learning rate for PPO optimizer
- `ppo_epochs`: Number of PPO epochs per update
- `clip_epsilon`: Clipping parameter for PPO
- `value_loss_coef`: Coefficient for value loss
- `entropy_coef`: Coefficient for entropy bonus
- `max_grad_norm`: Maximum gradient norm for clipping

#### Methods

##### `update(batch: Dict[str, Any]) -> Dict[str, float]`
Updates the model using PPO.

**Parameters**:
- `batch`: Batch of data for PPO update

**Returns**:
- `Dict`: Dictionary containing training metrics

##### `get_action(state: Dict[str, torch.Tensor], deterministic: bool = False) -> Dict[str, Any]`
Gets action from the policy.

**Parameters**:
- `state`: Current state
- `deterministic`: Whether to use deterministic policy

**Returns**:
- `Dict`: Dictionary containing action and related information

## Data Processing

### PPIDataset

```python
from utils.data_processing import PPIDataset
```

**Description**: Dataset class for protein-protein interaction data.

#### Constructor

```python
dataset = PPIDataset(
    data_path: str,
    preprocessor: ProteinPreprocessor,
    max_length: int = 1024,
    sliding_window: bool = False,
    window_size: int = 512,
    overlap: int = 128
)
```

**Parameters**:
- `data_path`: Path to PPI data file
- `preprocessor`: ProteinPreprocessor instance for data preprocessing
- `max_length`: Maximum sequence length
- `sliding_window`: Whether to use sliding window for long sequences
- `window_size`: Window size for sliding window approach
- `overlap`: Overlap between consecutive windows

#### Methods

##### `__getitem__(idx: int) -> Dict[str, Any]`
Gets an item from the dataset.

**Parameters**:
- `idx`: Index of the item

**Returns**:
- `Dict`: Dictionary containing protein pair data

##### `__len__() -> int`
Returns the length of the dataset.

**Returns**:
- `int`: Number of items in the dataset

### ProteinPreprocessor

```python
from utils.data_processing import ProteinPreprocessor
```

**Description**: Preprocessor for protein sequences and features.

#### Constructor

```python
preprocessor = ProteinPreprocessor(
    esm_model_name: str = "facebook/esm2_t33_650M_UR50D",
    max_length: int = 1024,
    device: Optional[Union[str, torch.device]] = None
)
```

**Parameters**:
- `esm_model_name`: Name of ESM model for encoding
- `max_length`: Maximum sequence length
- `device`: Device for computation

#### Methods

##### `preprocess_sequence(sequence: str) -> Dict[str, Any]`
Preprocesses a single protein sequence.

**Parameters**:
- `sequence`: Protein amino acid sequence

**Returns**:
- `Dict`: Dictionary containing preprocessed sequence data

##### `preprocess_batch(sequences: List[str]) -> Dict[str, Any]`
Preprocesses multiple protein sequences in batch.

**Parameters**:
- `sequences`: List of protein sequences

**Returns**:
- `Dict`: Dictionary containing batch preprocessed data

### DatasetSplitter

```python
from utils.data_processing import DatasetSplitter
```

**Description**: Splits dataset into training, validation, and test sets.

#### Constructor

```python
splitter = DatasetSplitter(
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
)
```

**Parameters**:
- `train_ratio`: Ratio of data for training
- `val_ratio`: Ratio of data for validation
- `test_ratio`: Ratio of data for testing
- `random_seed`: Random seed for reproducibility

#### Methods

##### `split(dataset: Dataset) -> Tuple[Dataset, Dataset, Dataset]`
Splits the dataset.

**Parameters**:
- `dataset`: Dataset to split

**Returns**:
- `Tuple`: Training, validation, and test datasets

### BlastpUtils

```python
from utils.data_processing import BlastpUtils
```

**Description**: Utilities for working with BLASTP results.

#### Methods

##### `parse_blast_results(blast_output: str) -> List[Dict[str, Any]]`
Parses BLASTP output into structured format.

**Parameters**:
- `blast_output`: BLASTP output string

**Returns**:
- `List`: List of parsed BLASTP results

##### `filter_blast_results(results: List[Dict[str, Any]], evalue_threshold: float = 1e-5) -> List[Dict[str, Any]]`
Filters BLASTP results based on e-value threshold.

**Parameters**:
- `results`: List of BLASTP results
- `evalue_threshold`: E-value threshold for filtering

**Returns**:
- `List`: Filtered BLASTP results

## Training

### TrainingPipeline

```python
from experiments.training.training_pipeline import TrainingPipeline
```

**Description**: Pipeline for training PGPR models.

#### Constructor

```python
training_pipeline = TrainingPipeline(
    model: nn.Module,
    optimizer: Optimizer,
    dataset: Dataset,
    loss_fn: Callable,
    config: Dict[str, Any]
)
```

**Parameters**:
- `model`: Model to train
- `optimizer`: Optimizer instance
- `dataset`: Training dataset
- `loss_fn`: Loss function
- `config`: Training configuration

#### Methods

##### `train(epochs: int, val_dataset: Optional[Dataset] = None) -> Dict[str, List[float]]`
Trains the model for specified epochs.

**Parameters**:
- `epochs`: Number of training epochs
- `val_dataset`: Validation dataset (optional)

**Returns**:
- `Dict`: Dictionary containing training and validation metrics

### RewardCalculator

```python
from experiments.training.reward_calculator import RewardCalculator
```

**Description**: Calculates rewards for reinforcement learning.

#### Constructor

```python
reward_calculator = RewardCalculator(
    accuracy_weight: float = 0.5,
    novelty_weight: float = 0.3,
    diversity_weight: float = 0.2
)
```

**Parameters**:
- `accuracy_weight`: Weight for accuracy component
- `novelty_weight`: Weight for novelty component
- `diversity_weight`: Weight for diversity component

#### Methods

##### `calculate_reward(predictions: torch.Tensor, labels: torch.Tensor, sequences: List[str]) -> torch.Tensor`
Calculates rewards for predictions.

**Parameters**:
- `predictions`: Model predictions
- `labels`: Ground truth labels
- `sequences`: Protein sequences

**Returns**:
- `torch.Tensor`: Calculated rewards

### PPOAgent

```python
from experiments.training.ppo_agent import PPOAgent
```

**Description**: PPO agent for reinforcement learning.

#### Constructor

```python
ppo_agent = PPOAgent(
    model: nn.Module,
    config: Dict[str, Any]
)
```

**Parameters**:
- `model`: Model to optimize
- `config`: PPO configuration

#### Methods

##### `train_step(batch: Dict[str, Any]) -> Dict[str, float]`
Performs a single training step.

**Parameters**:
- `batch`: Batch of training data

**Returns**:
- `Dict`: Dictionary containing training metrics

##### `evaluate(batch: Dict[str, Any]) -> Dict[str, float]`
Evaluates the agent on a batch of data.

**Parameters**:
- `batch`: Batch of evaluation data

**Returns**:
- `Dict`: Dictionary containing evaluation metrics

## Inference

### InferencePipeline

```python
from experiments.deployment.inference_pipeline import InferencePipeline
```

**Description**: Pipeline for performing PPI inference.

#### Constructor

```python
inference_pipeline = InferencePipeline(
    model_path: str,
    config_path: str,
    device: Optional[Union[str, torch.device]] = None
)
```

**Parameters**:
- `model_path`: Path to trained model checkpoint
- `config_path`: Path to inference configuration
- `device`: Device for computation

#### Methods

##### `predict(protein1_sequence: str, protein2_sequence: str, gene1: str = "", gene2: str = "") -> Dict[str, Any]`
Predicts PPI between two proteins.

**Parameters**:
- `protein1_sequence`: Amino acid sequence of first protein
- `protein2_sequence`: Amino acid sequence of second protein
- `gene1`: Name of first gene (optional)
- `gene2`: Name of second gene (optional)

**Returns**:
- `Dict`: Dictionary containing prediction results

##### `batch_predict(batch_data: List[Dict[str, str]]) -> List[Dict[str, Any]]`
Performs batch prediction.

**Parameters**:
- `batch_data`: List of protein pairs for prediction

**Returns**:
- `List`: List of prediction results

## Utilities

### Optimizer

```python
from utils.optimization import Optimizer
```

**Description**: Optimizer wrapper with support for gradient accumulation, mixed precision training, and parallelism.

#### Constructor

```python
optimizer = Optimizer(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    gradient_accumulation_steps: int = 1,
    mixed_precision: bool = False,
    parallel_mode: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None
)
```

**Parameters**:
- `model`: Model to optimize
- `optimizer`: PyTorch optimizer
- `gradient_accumulation_steps`: Number of steps for gradient accumulation
- `mixed_precision`: Whether to use mixed precision training
- `parallel_mode`: Parallelism mode ("dp", "ddp", or None)
- `device`: Device for computation

#### Methods

##### `zero_grad() -> None`
Zeros gradients (considering gradient accumulation).

##### `backward(loss: torch.Tensor) -> None`
Performs backward pass with support for gradient accumulation and mixed precision.

##### `step() -> None`
Updates model parameters (considering gradient accumulation).

### Metrics

```python
from utils.metrics import calculate_metrics
```

**Description**: Utility functions for calculating evaluation metrics.

#### Functions

##### `calculate_metrics(predictions: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]`
Calculates various evaluation metrics.

**Parameters**:
- `predictions`: Model predictions
- `labels`: Ground truth labels
- `threshold`: Threshold for binary classification

**Returns**:
- `Dict`: Dictionary containing accuracy, precision, recall, F1-score, and AUC-ROC

### Debugging

```python
from utils.debugging import GradientChecker, ActivationStats, AttentionVisualizer
```

**Description**: Debugging utilities for model analysis.

#### GradientChecker

```python
gradient_checker = GradientChecker(model)
gradient_checker.check_gradient_flow(loss)
```

**Methods**:
- `check_gradient_flow(loss: torch.Tensor) -> Dict[str, float]`: Checks gradient flow in the model

#### ActivationStats

```python
activation_stats = ActivationStats(model)
stats = activation_stats.get_activation_statistics(input_data)
```

**Methods**:
- `get_activation_statistics(input_data: Dict[str, torch.Tensor]) -> Dict[str, Any]`: Gets statistics about activations

#### AttentionVisualizer

```python
attention_visualizer = AttentionVisualizer()
attention_visualizer.visualize(attention_weights, sequence1, sequence2, output_path)
```

**Methods**:
- `visualize(attention_weights: torch.Tensor, seq1: str, seq2: str, output_path: str) -> None`: Visualizes attention weights

## Configuration

### ConfigManager

```python
from configs.config_manager import ConfigManager
```

**Description**: Manages configuration files for the project.

#### Constructor

```python
config_manager = ConfigManager(
    config_path: str,
    base_config_path: Optional[str] = None
)
```

**Parameters**:
- `config_path`: Path to main configuration file
- `base_config_path`: Path to base configuration file (optional)

#### Methods

##### `load_config() -> Dict[str, Any]`
Loads and validates configuration.

**Returns**:
- `Dict`: Loaded configuration dictionary

##### `save_config(config: Dict[str, Any], output_path: str) -> None`
Saves configuration to file.

**Parameters**:
- `config`: Configuration dictionary to save
- `output_path`: Path to save configuration file

##### `validate_config(config: Dict[str, Any]) -> bool`
Validates configuration against schema.

**Parameters**:
- `config`: Configuration dictionary to validate

**Returns**:
- `bool`: Whether configuration is valid

## Command Line Interface

### Main Entry Point

```bash
python main.py --help
```

**Description**: Main command line interface for PGPR.

**Commands**:
- `train`: Train a PGPR model
- `evaluate`: Evaluate a trained model
- `inference`: Perform PPI inference
- `preprocess`: Preprocess data
- `visualize`: Visualize model results

**Options**:
- `--config`: Path to configuration file
- `--data_path`: Path to data file
- `--model_path`: Path to model checkpoint
- `--output_path`: Path to output directory
- `--device`: Device for computation
- `--help`: Show help message
