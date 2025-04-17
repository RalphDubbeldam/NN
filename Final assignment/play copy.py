import numpy as np
import matplotlib.pyplot as plt

# Set seed for reproducibility
np.random.seed(42)

# Sample size
N = 10000

# 1. Normal distribution samples (take absolute value to keep positive)
normal_samples = np.abs(np.random.normal(loc=0, scale=1, size=N))
normal_log_sum = np.sum(np.log(normal_samples))

# 2. Image pixel-like samples (uniformly random in [1, 255])
image_samples = np.random.uniform(low=1, high=255, size=N)
image_log_sum = np.sum(np.log(image_samples))

# Print results
print(f"Sum of log(x) for normal samples: {normal_log_sum:.2f}")
print(f"Sum of log(x) for image-like samples: {image_log_sum:.2f}")

# Optional: plot histograms of log(x)
plt.figure(figsize=(12, 5))
plt.hist(np.log(normal_samples), bins=100, alpha=0.6, label='log(Normal samples)', color='blue')
plt.hist(np.log(image_samples), bins=100, alpha=0.6, label='log(Image samples)', color='orange')
plt.axvline(np.mean(np.log(normal_samples)), color='blue', linestyle='dashed')
plt.axvline(np.mean(np.log(image_samples)), color='orange', linestyle='dashed')
plt.title('Histogram of log(x) for two sample sets')
plt.xlabel('log(x)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.show()
