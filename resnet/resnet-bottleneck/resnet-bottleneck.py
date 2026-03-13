import numpy as np

def relu(x):
    return np.maximum(0, x)

class BottleneckBlock:
    """
    Bottleneck Block: 1x1 -> 3x3 -> 1x1
    Reduces computation by compressing channels.
    """
    
    def __init__(self, in_channels: int, bottleneck_channels: int, out_channels: int):
        self.in_ch = in_channels
        self.bn_ch = bottleneck_channels  # Compressed dimension
        self.out_ch = out_channels
        
        # 1x1 reduce
        self.W1 = np.random.randn(in_channels, bottleneck_channels) * 0.01
        # 3x3 (simplified as dense)
        self.W2 = np.random.randn(bottleneck_channels, bottleneck_channels) * 0.01
        # 1x1 expand
        self.W3 = np.random.randn(bottleneck_channels, out_channels) * 0.01
        
        # Shortcut (if dimensions differ)
        self.Ws = np.random.randn(in_channels, out_channels) * 0.01 if in_channels != out_channels else None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
    
        # 1x1 reduce
        z1 = x @ self.W1
        a1 = relu(z1)
    
        # 3x3 process
        z2 = a1 @ self.W2
        a2 = relu(z2)
    
        # 1x1 expand
        z3 = a2 @ self.W3
        a3 = relu(z3)
    
        # Shortcut connection
        if self.Ws is not None:
            shortcut = x @ self.Ws
        else:
            shortcut = x
    
        # Add skip connection
        y = a3 + shortcut
    
        return y
