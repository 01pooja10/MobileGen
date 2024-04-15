import cv2
import numpy as np
import time

class ImageFilter:
    def __init__(self, filter_type='gaussian', weight=[0.25, 0.25, 0.25, 0.25]):
        self.filter_type = filter_type
        self.w = weight
        
    def gaussian_filtering(self, image, kernel_size=5):
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    def median_filtering(self, image, kernel_size=5):
        return cv2.medianBlur(image, kernel_size)

    def bilateral_filtering(self, image, d=9, sigma_color=75, sigma_space=75):
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

    def nlm_filtering(self, image, h=10, search_window=21, template_window=7):
        return cv2.fastNlMeansDenoisingColored(image, None, h, h, search_window, template_window)

    def weighted_combination(self, image):
        weights = {
            'gaussian': self.w[0],
            'median': self.w[1],
            'bilateral': self.w[2],
            'nlm': self.w[3],
        }

        result = np.zeros_like(image, dtype=np.float32)
        
        for func_name, weight in weights.items():
            filtered_image = getattr(self, func_name + '_filtering')(image)
            result += filtered_image.astype(np.float32) * weight
        
        return result.astype(np.uint8)

    def apply_filter(self, image, **kwargs):
        start_time = time.time()
        
        if self.filter_type == 'gaussian':
            filtered_image = self.gaussian_filtering(image, **kwargs)
        elif self.filter_type == 'median':
            filtered_image = self.median_filtering(image, **kwargs)
        elif self.filter_type == 'bilateral':
            filtered_image = self.bilateral_filtering(image, **kwargs)
        elif self.filter_type == 'nlm':
            filtered_image = self.nlm_filtering(image, **kwargs)
        elif self.filter_type == 'weighted':
            filtered_image = self.weighted_combination(image, **kwargs)
        else:
            raise ValueError("Invalid filter type. Choose from 'gaussian', 'median', 'bilateral', 'nlm', or 'weighted'.")
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        return filtered_image, execution_time


# if __name__=='__main__':
#     # Create an instance of ImageFilter
#     filter_instance = ImageFilter(filter_type='weighted', weight = [0, 0, 0.5, 0.5])

#     # Load the image
#     image = cv2.imread('/content/drive/MyDrive/VisualComputing/frog.jpg', cv2.IMREAD_COLOR)

#     # Apply the selected filter
#     filtered_image, execution_time = filter_instance.apply_filter(image)

#     plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

#     print("Time taken:", execution_time)