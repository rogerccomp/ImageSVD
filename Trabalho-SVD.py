import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

##Abre a imagem e retorna 3 matrizes, cada uma correspondente a um canal (R, G and B channels)
def openImage(imagePath):
    imOrig = Image.open(imagePath)
    im = np.array(imOrig)
    aRed = im[:, :, 0]
    aGreen = im[:, :, 1]
    aBlue = im[:, :, 2]
    print("Imagem Original : {}".format(im.shape))

    return [aRed, aGreen, aBlue, imOrig]


## Compressão da matriz de um canal específico
def compressSingleChannel(channelDataMatrix, singularValuesLimit):
    uChannel, sChannel, vhChannel = np.linalg.svd(channelDataMatrix)
    aChannelCompressed = np.zeros((channelDataMatrix.shape[0], channelDataMatrix.shape[1]))
    k = singularValuesLimit

    leftSide = np.matmul(uChannel[:, 0:k], np.diag(sChannel)[0:k, 0:k])
    aChannelCompressedInner = np.matmul(leftSide, vhChannel[0:k, :])
    aChannelCompressed = aChannelCompressedInner.astype('uint8')
    return aChannelCompressed


## Programa Principal:
print('*** \033[33mCompressão de Imagem via Decomposição SVD\033[m ***')
aRed, aGreen, aBlue, originalImage = openImage("luta.jpg")

## Largura e Altura da Imagem:
imageWidth = 428
imageHeight = 428

## Numero de valores singulares para reconstruir a imagem comprimida
singularValuesLimit = 20

aRedCompressed = compressSingleChannel(aRed, singularValuesLimit)
aGreenCompressed = compressSingleChannel(aGreen, singularValuesLimit)
aBlueCompressed = compressSingleChannel(aBlue, singularValuesLimit)

imr = Image.fromarray(aRedCompressed, mode=None)
img = Image.fromarray(aGreenCompressed, mode=None)
imb = Image.fromarray(aBlueCompressed, mode=None)

newImage = Image.merge("RGB", (imr, img, imb))

## Plotando imagem original e NovaImagem
plt.figure("Figura 1 - Compressão de Imagem por SVD (Original)",facecolor="gray")
plt.get_current_fig_manager().window.wm_iconbitmap("logo-uerj.ico")
plt.imshow(originalImage)
plt.title("Imagem Original",fontweight="bold")
plt.xticks([])
plt.yticks([])


plt.figure("Figura 2 - Imagem Comprimida",facecolor="gray")
plt.get_current_fig_manager().window.wm_iconbitmap("logo-uerj.ico")
plt.imshow(newImage)
plt.title(f"Compressão com k = {singularValuesLimit}",fontweight="bold")
plt.xticks([])
plt.yticks([])
plt.box(False)

plt.show()


## Calculo do raio de compressão
mr = imageHeight
mc = imageWidth

originalSize = mr * mc * 3
compressedSize = singularValuesLimit * (1 + mr + mc) * 3

print('\033[35mTamanho Original \033[m:')
print(originalSize)

print('\033[35mTamanho Compressão \033[m:')
print(compressedSize)

print("\033[36mRazão tamanho compressão / tamanho original \033[m:")
razao = compressedSize * 1.0 / originalSize
print(razao)

print('Tamanho da Imagem de Compressão é ' + str(round(razao * 100, 2)) + '%  da imagem original ')
print("\033[32mProcesso de Compressão de Imagem Concluído com Sucesso!!\033[m")
