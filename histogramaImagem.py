import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

img = np.array(Image.open("paris.jpg"))  # M = np.asarray(M,dtype=float32)/255


# visualizando a imagem
plt.figure(figsize=(5,5))
plt.title("Imagem Original",weight="bold",size=12)
plt.xticks([])
plt.yticks([])
plt.imshow(img,aspect="auto")
plt.axis("off")


# imagem camada de azul
plt.figure(figsize=(5,5))
plt.title("Azul",weight="bold",size=12,color="blue")
plt.xticks([])
plt.yticks([])
blue = img[:,:,2]
plt.imshow(blue)

plt.figure(figsize=(5,5))
plt.hist(blue.ravel(),bins=256,color="blue")
plt.title("Azul",weight="bold",size=12,color="blue")
plt.show()
