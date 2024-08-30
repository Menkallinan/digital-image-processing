import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from skimage import io
from PIL import Image
import numpy as np

# Configuração do gerador de aumento de dados
datagen = ImageDataGenerator(
    rotation_range=45,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Lista de frutas
fruits = ["apple", "avocado", "banana", "guava", "lemon", "mango", "melon", "nectarine", "orange", "passion_fruit"]
base_directory = '/content/digital-image-processing/project/project_part1/fruits/'

# Tamanho para redimensionamento das imagens
SIZE = 1000

# Número de novas imagens aumentadas desejadas (aumenta em 3x)
augmentation_factor = 3

# Loop através de cada fruta
for fruit in fruits:
    dataset = []
    image_directory = os.path.join(base_directory, fruit)
    output_directory = os.path.join(base_directory, f'augmentation_{fruit}')
    
    # Criar o diretório de saída se não existir
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # Carregar as imagens
    my_images = os.listdir(image_directory)
    for image_name in my_images:
        if '.' in image_name and image_name.split('.')[-1] == 'png':
            image_path = os.path.join(image_directory, image_name)
            image = io.imread(image_path)
            image = Image.fromarray(image, 'RGB')
            image = image.resize((SIZE, SIZE))
            dataset.append(np.array(image))
    
    x = np.array(dataset)
    
    # Número desejado de novas imagens aumentadas
    num_desired_augmented_images = len(x) * augmentation_factor
    
    i = 0
    # Aplicar data augmentation e salvar as imagens
    for batch in datagen.flow(x, batch_size=1, save_to_dir=output_directory, save_prefix='aug', save_format='png'):
        i += 1
        if i >= num_desired_augmented_images:
            break

print("Data augmentation completed for all fruits.")
