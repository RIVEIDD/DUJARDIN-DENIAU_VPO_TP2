# pylint: disable=no-member

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


def part_2():
    img_name: str = "peppers-256.png"
    src = os.path.join(os.getcwd(), "imagesDeTest", img_name)# permet de garder la même nomenclature de chemin entre les différents os

    new_wh: int = 64 #nouvelle largeur d'image
    old_wh: int = 256 #ancienne largeur d'image
    divisor: int = old_wh // new_wh #ratio entre ancienne et nouvelle image (ici 4)

    img = cv2.imread(src, cv2.IMREAD_GRAYSCALE) #permet de lire l'image
    new_img = numpy.zeros([new_wh, new_wh, 1], dtype=numpy.uint8) #crée une nouvelle image vide de taille 64x64 en nuance de gris

    for i in range(0, old_wh, divisor): #itère de 4 en 4 sur les pixels de l'image source
        for j in range(0, old_wh, divisor):
            new_img[i // divisor, j // divisor] = img[i, j] #on divise la valeur de l'itération par 4 pour avoir des valeurs entre 0 et 63

    cv2.imshow(img_name, img) #affiche image source
    cv2.imshow("New " + img_name, new_img) #affiche nouvelle image
    cv2.waitKey(0) #en attente d'une touche
    cv2.destroyAllWindows() #ferme les images


def part_3():
    img_name: str = "peppers-512.png"
    src = os.path.join(os.getcwd(), "imagesDeTest", img_name)

    img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
    #tableau de matrice des différents filtres : 
    filters = [
        numpy.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16,
        numpy.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9,
        numpy.array([[1, -3, 1], [-3, 9, -3], [1, -3, 1]]),
        numpy.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
        numpy.array([[0, -1, -1], [1, 0, -1], [1, 1, 0]]),
        numpy.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]),
    ]
    #on itérère sur l'ensemble du tableau de matrices
    for i, filter_test in enumerate(filters):
        dst = numpy.zeros([img.shape[0], img.shape[0], 1], dtype=numpy.uint8) #on définit comme destination une image vide
        cv2.filter2D(img, -1, filter_test, dst) #on applique le filtre (filtre n°i) sur l'image "img" (ici peppers-512.png) et on enregistre le résultat dans l'image vide (dst)
        cv2.imshow(f"Filter {i} {img_name}", dst) #on affiche l'image filtrée
    cv2.imshow(img_name, img) #on affiche l'image source

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    



# Charger l'image en niveaux de gris et la binariser
image = cv2.imread('peppers-512.png', cv2.IMREAD_GRAYSCALE)
_, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

# Définir l'élément structurant (carré 3x3)
kernel_size = 3
kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

def erosion_manual(img, kernel):
    """Implémentation manuelle de l'érosion."""
    rows, cols = img.shape
    krows, kcols = kernel.shape
    pad = krows // 2
    padded_img = np.pad(img, pad, mode='constant', constant_values=0)
    result = np.zeros_like(img)

    for i in range(rows):
        for j in range(cols):
            # Vérifier si le kernel s'applique entièrement
            region = padded_img[i:i + krows, j:j + kcols]
            if np.array_equal(region & kernel, kernel):  # Tous les pixels de la région sont blancs
                result[i, j] = 255
            else:
                result[i, j] = 0
    return result

def dilation_manual(img, kernel):
    """Implémentation manuelle de la dilatation."""
    rows, cols = img.shape
    krows, kcols = kernel.shape
    pad = krows // 2
    padded_img = np.pad(img, pad, mode='constant', constant_values=0)
    result = np.zeros_like(img)

    for i in range(rows):
        for j in range(cols):
            # Vérifier si au moins un pixel de la région est blanc
            region = padded_img[i:i + krows, j:j + kcols]
            if np.any(region & kernel):  # Au moins un pixel de la région est blanc
                result[i, j] = 255
            else:
                result[i, j] = 0
    return result

# Appliquer l'érosion et la dilatation manuelles
eroded_image = erosion_manual(binary_image, kernel)
dilated_image = dilation_manual(binary_image, kernel)

# Afficher les résultats
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title("Image binaire")
plt.imshow(binary_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Érosion manuelle")
plt.imshow(eroded_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Dilatation manuelle")
plt.imshow(dilated_image, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()


def part_4():
    img_name: str = "peppers-512.png"
    src = os.path.join(os.getcwd(), "imagesDeTest", img_name)

    img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
    #itération sur les pixels :
    new_img = numpy.zeros([512, 512, 1], dtype=numpy.uint8)
    for i in range(0, 512, 1): #itère de 4 en 4 sur les pixels de l'image source
        for j in range(0, 512, 1):
            if  img[i , j ] >= 126 :
                new_img[i , j ]= 255
            else :
                new_img[i , j ] = 0
            

    
    cv2.imshow(img_name, img) #affiche image source
    cv2.imshow("New " + img_name, new_img) #affiche nouvelle image
    cv2.waitKey(0) #en attente d'une touche
    cv2.destroyAllWindows() #ferme les images


if __name__ == "__main__":
    part_4()    