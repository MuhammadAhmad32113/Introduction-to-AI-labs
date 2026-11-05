"""Lab 07 - CNN Layers, built from scratch with NumPy.
Pipeline from 'CNN Architecture.pptx':
Input -> Conv + ReLU -> Pooling -> Flatten -> Fully Connected -> Softmax
Run:  python lab07_cnn_layers.py
"""
import numpy as np

def conv2d(image, kernel):
    """Valid convolution (stride 1, no padding)."""
    kh, kw = kernel.shape
    oh, ow = image.shape[0] - kh + 1, image.shape[1] - kw + 1
    out = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
            out[i, j] = np.sum(image[i:i + kh, j:j + kw] * kernel)
    return out

def relu(x):
    return np.maximum(0, x)

def max_pool(x, size=2):
    h, w = x.shape[0] // size, x.shape[1] // size
    out = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            out[i, j] = x[i*size:(i+1)*size, j*size:(j+1)*size].max()
    return out

def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()

if __name__ == "__main__":
    np.random.seed(0)
    image = np.array([[3,1,0,2,1,2],
                      [0,1,3,0,4,1],
                      [0,1,2,1,3,2],
                      [0,1,0,2,2,1],
                      [1,0,3,1,0,2],
                      [2,1,0,1,3,0]], dtype=float)
    print("1. Input image", image.shape, "\n", image)

    kernel = np.array([[1,0,-1],[1,0,-1],[1,0,-1]], dtype=float)
    conv = conv2d(image, kernel)
    print("\n2a. Convolution output", conv.shape, "\n", conv)
    act = relu(conv)
    print("2b. After ReLU\n", act)

    pooled = max_pool(act, 2)
    print("\n3. After 2x2 max pooling", pooled.shape, "\n", pooled)

    flat = pooled.flatten()
    print("\n4. Flattened vector", flat)

    W = np.random.randn(3, flat.size) * 0.5
    logits = W @ flat
    print("\n5. Fully connected output (logits)", logits)

    probs = softmax(logits)
    print("\n6. Softmax probabilities", probs, "| sum =", probs.sum())
    print("Predicted class:", int(np.argmax(probs)))
