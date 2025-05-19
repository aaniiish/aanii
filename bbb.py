import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load grayscale image
image_path = 'C:/Users/Administrator/Desktop/annie/OIP.jpg'  # Replace with your image path
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Apply adaptive threshold to detect varied intensities
thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 11, 2)

# Optional: use edge detection (Canny)
edges = cv2.Canny(img, 50, 150)

# Combine both for enhanced effect
combined = cv2.bitwise_or(thresh, edges)

# Find contours (simulated abnormal regions)
contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Convert original to color to draw bounding boxes
output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# Draw contours and bounding boxes
for cnt in contours:
    if cv2.contourArea(cnt) > 100:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Show result
plt.imshow(output)
plt.title('Abnormality Detection (Simulated)')
plt.axis('off')
plt.show()


