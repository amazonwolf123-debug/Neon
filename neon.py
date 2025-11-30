import cv2
import numpy as np
from matplotlib import pyplot as plt

# load
image_path = 'joeyselfie.jpg'
original = cv2.imread(image_path)

if original is None:
    raise FileNotFoundError("Image not found. Check path to joeyselfie.jpg")

# convert
gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

# noise red
blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)

# canny ed red
edges = cv2.Canny(blurred, 40, 140, apertureSize=3, L2gradient=True)

# 3-channel ed for glow
edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

# === MULTI-LAYER NEON GLOW PARAMETERS ===
glow_layers = 6                # number - layer
initial_blur = 7               
blur_growth = 4                
initial_alpha = 0.35           # how strong the first glow layer is
alpha_decay = 0.6              # decrease factor per layer
neon_bgr = np.array([255, 50, 255]) 

final = original.copy()

# === MULTI-LAYER GLOW LOOP ===
for i in range(glow_layers):
    # Expand blur size each pass for larger halo
    k = initial_blur + blur_growth * i
    k = max(3, k | 1)  # force odd kernel

    # Blur edges
    glow = cv2.GaussianBlur(edges_color, (k, k), 0)

    # Tint the glow w neon
    tinted_glow = cv2.multiply(glow, neon_bgr / 255.0)

    # Reduce alpha
    alpha = initial_alpha * (alpha_decay ** i)

    # Blend into final!!!
    final = cv2.addWeighted(final, 1.0, tinted_glow.astype(np.uint8), alpha, 0)

# Final edge highlight
sharp_glow = cv2.bitwise_and(neon_bgr, neon_bgr, mask=edges)
final = cv2.addWeighted(final, 0.9, sharp_glow, 0.4, 0)

# === DISPLAY ===
plt.figure(figsize=(14, 6))

plt.subplot(131)
plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
plt.title('Original')
plt.axis('off')

plt.subplot(132)
plt.imshow(edges, cmap='gray')
plt.title('Edges')
plt.axis('off')

plt.subplot(133)
plt.imshow(cv2.cvtColor(final, cv2.COLOR_BGR2RGB))
plt.title('Neon Multi-Glow')
plt.axis('off')

plt.show()

# Save
cv2.imwrite('joey_neon_multiglow.jpg', final)