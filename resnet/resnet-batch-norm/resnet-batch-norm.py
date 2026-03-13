import numpy as np

class BatchNorm:
    """Batch Normalization layer."""
    
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        self.eps = eps
        self.momentum = momentum
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        
        if training:
            mean = np.mean(x, axis=0)
            var = np.var(x, axis=0)

            x_hat = (x - mean) / np.sqrt(var + self.eps)

            # update running stats
            self.running_mean = self.momentum * mean + (1 - self.momentum) * self.running_mean
            self.running_var = self.momentum * var + (1 - self.momentum) * self.running_var

        else:
            x_hat = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)

        out = self.gamma * x_hat + self.beta
        return out

def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation."""
    return np.maximum(0, x)

def post_activation_block(x: np.ndarray, W1: np.ndarray, W2: np.ndarray, bn1: BatchNorm, bn2: BatchNorm) -> np.ndarray:
    
    out = x @ W1
    out = bn1.forward(out)
    out = relu(out)

    out = out @ W2
    out = bn2.forward(out)

    out = out + x   # skip connection
    out = relu(out)

    return out
def pre_activation_block(x: np.ndarray, W1: np.ndarray, W2: np.ndarray, bn1: BatchNorm, bn2: BatchNorm) -> np.ndarray:
    
    out = bn1.forward(x)
    out = relu(out)
    out = out @ W1

    out = bn2.forward(out)
    out = relu(out)
    out = out @ W2

    out = out + x   # identity skip

    return out