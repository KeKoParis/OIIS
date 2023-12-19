import numpy as np
from PIL import Image

def median_filter(image, filter_size):
    width, height = image.shape
    filtered_image = np.zeros((width, height))

    pad = filter_size // 2
    padded_image = np.pad(image, ((pad, pad), (pad, pad)), mode='edge')

    for i in range(width):
        for j in range(height):
            neighbors = padded_image[i:i+filter_size, j:j+filter_size]
            median = np.median(neighbors)
            filtered_image[i, j] = median

    return filtered_image.astype(np.uint8)


def histogram_equalization(image):
    width, height = image.shape
    histogram = np.bincount(image.flatten(), minlength=256)
    cumulative_histogram = np.cumsum(histogram)
    cumulative_histogram_normalized = (cumulative_histogram * 255) / cumulative_histogram[-1]
    equalized_image = cumulative_histogram_normalized[image]
    return equalized_image.astype(np.uint8)


def edge_detection(image):
    width, height = image.shape
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    gradient_x = np.zeros((width, height))
    gradient_y = np.zeros((width, height))

    pad = 1
    padded_image = np.pad(image, ((pad, pad), (pad, pad)), mode='edge')

    for i in range(width):
        for j in range(height):
            neighbors = padded_image[i:i+3, j:j+3]
            gradient_x[i, j] = np.sum(neighbors * sobel_x)
            gradient_y[i, j] = np.sum(neighbors * sobel_y)

    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_magnitude *= 255.0 / np.max(gradient_magnitude)

    return gradient_magnitude.astype(np.uint8)


# Пример использования функций
image_path = 'pic.jpeg'  # Путь к изображению
image = np.array(Image.open(image_path).convert('L'))  # Загрузка изображения и преобразование в оттенки серого

# Медианная фильтрация
filtered_image = median_filter(image, filter_size=3)

# Гистограммное выравнивание
equalized_image = histogram_equalization(filtered_image)

# Выделение границ
edge_image = edge_detection(equalized_image)

# Сохранение результатов
filtered_image_path = 'filtered_image.png'
equalized_image_path = 'equalized_image.png'
edge_image_path = 'edge_image.png'
Image.fromarray(filtered_image).save(filtered_image_path)
Image.fromarray(equalized_image).save(equalized_image_path)
Image.fromarray(edge_image).save(edge_image_path)