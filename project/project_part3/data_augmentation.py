import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from PIL import Image

# Funções de processamento de imagem
def read_image(url):
    return io.imread(url)

def filter_log(image):
    c = 255 / np.log(1 + np.max(image))
    log_image = c * (np.log(image + 1))
    log_image = np.array(log_image, dtype=np.uint8)
    return log_image
    

def filter_pot(image, gamma=2.2):
    gamma_corrected = np.array(255 * (image / 255) ** gamma, dtype='uint8')
    return gamma_corrected

def filter_cnv(image):
    flt_img = cv2.blur(image, (9, 9))
    return flt_img

def show_image(image):
    plt.imshow(image)
    plt.axis('off')
    plt.show()

# Lista de frutas
fruits = ["apple", "avocado", "banana", "guava", "lemon", "mango", "melon", "nectarine", "orange", "passion_fruit"]
base_directory = '/content/digital-image-processing/project/project_part1/fruits/'  # Diretório base para as imagens

# Loop através de cada fruta
for fruit in fruits:
    image_directory = os.path.join(base_directory, fruit)
    output_directory = os.path.join(base_directory, f'augmentation_{fruit}')
    
    # Verifica se o diretório de saída existe, se não, cria
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # Carregar as imagens
    my_images = os.listdir(image_directory)
    for image_name in my_images:
        if image_name.endswith('.png'):
            image_path = os.path.join(image_directory, image_name)
            image = read_image(image_path)
            
            # Aplicar filtros
            image_log = filter_log(image)
            image_pot = filter_pot(image)
            image_cnv = filter_cnv(image)
            
            # Salvar as imagens processadas
            log_image_path = os.path.join(output_directory, f'log_{image_name}')
            pot_image_path = os.path.join(output_directory, f'pot_{image_name}')
            cnv_image_path = os.path.join(output_directory, f'cnv_{image_name}')
            
            io.imsave(log_image_path, image_log)
            io.imsave(pot_image_path, image_pot)
            io.imsave(cnv_image_path, image_cnv)

print("Data augmentation completed for all fruits.")

