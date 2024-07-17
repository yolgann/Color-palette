# Color-palette
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def extract_palette(image_path, num_colors):
    # Read the image
    image = cv2.imread(image_path)
    # Convert the image to RGB format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Reshape the image to be a list of pixels
    pixels = image.reshape(-1, 3)
    
    # Perform k-means clustering to find the most dominant colors
    kmeans = KMeans(n_clusters=num_colors)
    kmeans.fit(pixels)
    
    # Get the cluster centers (the dominant colors)
    colors = kmeans.cluster_centers_
    # Convert colors to integers
    colors = colors.round(0).astype(int)
    
    # Display the colors as a palette
    plt.figure(figsize=(8, 2))
    plt.subplot(1, 2, 1)
    plt.imshow([colors])
    plt.axis('off')
    
    # Display the original image
    plt.subplot(1, 2, 2)
    plt.imshow(image)
    plt.axis('off')
    
    plt.show()
    
    return colors

# Example usage
image_path = 'path_to_your_image.jpg'
num_colors = 5
palette = extract_palette(image_path, num_colors)
print("Extracted Colors:", palette)
