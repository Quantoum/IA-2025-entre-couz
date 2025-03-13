# TO-DO : Fenix

### 1) encoding the board to TensorFlow
This uses multiple channels in a tensor to represent the pieces and game states

### 2) Neural Network architecture
This is the brain for move prediction (policy) and position evaluation (value).
With AlphaZero architecture :
  * channels
  * Convolutionnal Tower (ResNet Blocks) -> batch normalization and ReLu
  * Policy head : predicts move probability from the board state
  * Value head : predicts the expected outcome (win probability, [-1, 1])

### 3) Monte-Carlo Tree Search
MCTS uses a neural network to guide simulations and evaluate positions.
Components :
  * Node class
  * Select Action (PUCT Algorithm)
  * Simulation : 
    1. Expand leaf nodes once visit count exceeds a threshold.
    2. Use the network to predict (policy, value) for the leaf state.
    3. Backpropagate the value through the tree.

### 4) Training Loop
Uses self-play and training in iteratives cycles
Self-play :
  * Generate games with MCTS to select moves
Training :
  * update the neural network

### 5) Optimization
1. Move encoding : flat action space 
* exemple (linear encoding) : from_square * 64 + to_square for chess-like moves, plus flags for promotions.

2. Mixed precision training : use FP16 to speed up training (by GPU)

3. Symmetry Augmentation : rotate or flip the board (generate more training data)

4. Efficient MCTS : implement it in C(++) -> pybind11 ? or existing ? and masking illegal moves during policy prediction

5. Inference Optimization : TensorRT or ONNX to optimize neural network (inference)

### 6) Warnings

1. Training stability : 
* label smoothing (small noise policy to prevent overconfidence)
* gradient clipping (avoid exploding gradients)