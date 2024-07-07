import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

img = Image.open("luta.jpg")
img = np.asarray(img,dtype=np.float32)/255

# visualizando a imagem
plt.figure(figsize=(5,5))
plt.title("Imagem Original",weight="bold",size=12)
plt.xticks([])
plt.yticks([])
plt.imshow(img,aspect="auto")
plt.axis("off")
plt.show()

# imagem camada de vermelho
plt.figure(figsize=(5,5))
plt.title("Vermelho",weight="bold",size=12,color="red")
plt.xticks([])
plt.yticks([])
red = img[:,:,0]
plt.imshow(red)
plt.show()

# imagem camada de verde
plt.figure(figsize=(5,5))
plt.title("Verde",weight="bold",size=12,color="green")
plt.xticks([])
plt.yticks([])
green = img[:,:,1]
plt.imshow(green)
plt.show()

# imagem camada de azul
plt.figure(figsize=(5,5))
plt.title("Azul",weight="bold",size=12,color="blue")
plt.xticks([])
plt.yticks([])
blue = img[:,:,2]
plt.imshow(blue)
plt.show()
