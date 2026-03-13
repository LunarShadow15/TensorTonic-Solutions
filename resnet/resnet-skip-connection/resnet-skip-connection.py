import numpy as np

def compute_gradient_without_skip(gradients_F: list, x: np.ndarray) -> np.ndarray:
    """
    Gradient flow through L layers WITHOUT skip connections.
    """
    
    grad = np.array(x).reshape(-1, 1)

    for g in gradients_F:
        grad = g @ grad

    return grad.flatten()


def compute_gradient_with_skip(gradients_F: list, x: np.ndarray) -> np.ndarray:
    """
    Gradient flow through L layers WITH skip connections.
    """
    
    grad = np.array(x).reshape(-1, 1)

    for g in gradients_F:
        I = np.eye(g.shape[0])
        grad = (I + g) @ grad

    return grad.flatten()