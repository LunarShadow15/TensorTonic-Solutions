import numpy as np

def relu(x):
    return np.maximum(0, x)

class BasicBlock:
    """Basic residual block (2 conv layers with skip connection)."""
    
    def __init__(self, in_ch: int, out_ch: int, downsample: bool = False):
        self.downsample = downsample
        self.W1 = np.random.randn(in_ch, out_ch) * 0.01
        self.W2 = np.random.randn(out_ch, out_ch) * 0.01
        # Projection shortcut if dimensions change
        self.W_proj = np.random.randn(in_ch, out_ch) * 0.01 if in_ch != out_ch or downsample else None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
    
        out = x @ self.W1
        out = relu(out)
    
        out = out @ self.W2
    
        if self.W_proj is not None:
            shortcut = x @ self.W_proj
        else:
            shortcut = x
    
        out = out + shortcut
        out = relu(out)
    
        return out

class ResNet18:
    
    def __init__(self, num_classes: int = 10):
        self.conv1 = np.random.randn(3, 64) * 0.01

        self.layer1 = [
            BasicBlock(64, 64),
            BasicBlock(64, 64)
        ]

        self.layer2 = [
            BasicBlock(64, 128, downsample=True),
            BasicBlock(128, 128)
        ]

        self.layer3 = [
            BasicBlock(128, 256, downsample=True),
            BasicBlock(256, 256)
        ]

        self.layer4 = [
            BasicBlock(256, 512, downsample=True),
            BasicBlock(512, 512)
        ]

        self.fc = np.random.randn(512, num_classes) * 0.01
    
    def forward(self, x: np.ndarray) -> np.ndarray:

        # Initial conv
        x = x @ self.conv1
        x = relu(x)
    
        # Residual stages
        for block in self.layer1:
            x = block.forward(x)
    
        for block in self.layer2:
            x = block.forward(x)
    
        for block in self.layer3:
            x = block.forward(x)
    
        for block in self.layer4:
            x = block.forward(x)
    
        # Global average pooling (keep batch dimension)
        if x.ndim > 2:
            x = np.mean(x, axis=tuple(range(1, x.ndim)))
    
        # Fully connected
        logits = x @ self.fc
    
        return logits