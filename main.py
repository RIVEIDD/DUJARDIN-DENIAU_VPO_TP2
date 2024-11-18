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
    new_img = np.zeros([new_wh, new_wh, 1], dtype=np.uint8) #crée une nouvelle image vide de taille 64x64 en nuance de gris

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
        np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16,
        np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9,
        np.array([[1, -3, 1], [-3, 9, -3], [1, -3, 1]]),
        np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
        np.array([[0, -1, -1], [1, 0, -1], [1, 1, 0]]),
        np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]),
    ]
    #on itérère sur l'ensemble du tableau de matrices
    for i, filter_test in enumerate(filters):
        dst = np.zeros([img.shape[0], img.shape[0], 1], dtype=np.uint8) #on définit comme destination une image vide
        cv2.filter2D(img, -1, filter_test, dst) #on applique le filtre (filtre n°i) sur l'image "img" (ici peppers-512.png) et on enregistre le résultat dans l'image vide (dst)
        cv2.imshow(f"Filter {i} {img_name}", dst) #on affiche l'image filtrée
    cv2.imshow(img_name, img) #on affiche l'image source

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    







def erosion_manual(img, kernel):
    """Implémentation manuelle de l'érosion."""
    # Convertir l'image en niveaux de gris si nécessaire
    if len(img.shape) == 3:  # Image en couleur (hauteur, largeur, canaux)
        img = np.mean(img, axis=2).astype(np.uint8)  # Conversion en niveaux de gris
    # Dimensions de l'image
    rows, cols = img.shape
    # Dimensions du noyau
    krows, kcols = kernel.shape
    # pad = krows // 2
    # padded_img = np.pad(img, pad, mode='constant', constant_values=0)
    # Padding nécessaire
    pad_row = krows // 2
    pad_col = kcols // 2
    
    # Ajouter du padding autour de l'image
    padded_img = np.pad(img, ((pad_row, pad_row), (pad_col, pad_col)), mode='constant', constant_values=0)
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
    # Convertir l'image en niveaux de gris si nécessaire
    if len(img.shape) == 3:  # Image en couleur (hauteur, largeur, canaux)
        img = np.mean(img, axis=2).astype(np.uint8)  # Conversion en niveaux de gris
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


def part_4():
    img_name: str = "peppers-512.png"
    src = os.path.join(os.getcwd(), "imagesDeTest", img_name)

    img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
    #itération sur les pixels :
    new_img = np.zeros([512, 512, 1], dtype=np.uint8)
    for i in range(0, 512, 1): #itère de 4 en 4 sur les pixels de l'image source
        for j in range(0, 512, 1):
            if  img[i , j ] >= 126 :
                new_img[i , j ]= 255
            else :
                new_img[i , j ] = 0
    
    # Définir l'élément structurant (carré 3x3)
    kernel_size = 3
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    # Appliquer l'érosion et la dilatation manuelles
    eroded_image = erosion_manual(new_img, kernel)
    dilated_image = dilation_manual(new_img, kernel)   
    # Afficher les résultats avec OpenCV
    cv2.imshow("Image binaire", new_img)
    cv2.imshow("Érosion manuelle", eroded_image)
    cv2.imshow("Dilatation manuelle", dilated_image)

    # Attendre que l'utilisateur appuie sur une touche pour fermer les fenêtres
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def part_5():
    img_name: str = "peppers-512.png"
    src = os.path.join(os.getcwd(), "imagesDeTest", img_name)
    img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)

    dft = np.fft.fft2(img) #réalisation de la transformée de fourrier
    dft_shift = np.fft.fftshift(dft) #décalage de la transformée de fourrier pour une meilleure visibilité 
    magnitude_spectrum = 20 * np.log(np.abs(dft_shift)) #calcul de la magnitude
    phase_spectrum = np.angle(dft_shift) #calcul de la phase

    plt.subplot(231), plt.imshow(img, cmap="gray")
    plt.title(img_name), plt.xticks([]), plt.yticks([])
    plt.subplot(232), plt.imshow(magnitude_spectrum, cmap="gray")
    plt.title("Magnitude"), plt.xticks([]), plt.yticks([])
    plt.subplot(233), plt.imshow(phase_spectrum, cmap="gray")
    plt.title("Phase"), plt.xticks([]), plt.yticks([])

    im_center = tuple(np.array(img.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(im_center, 90, 1.0)

    rotated_img = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR) #image retournée de 90 degrés 
    r_dft = np.fft.fft2(rotated_img)
    r_dft_shift = np.fft.fftshift(r_dft)
    r_magnitude_spectrum = 20 * np.log(np.abs(r_dft_shift))
    r_phase_spectrum = np.angle(r_dft_shift)

    plt.subplot(234), plt.imshow(rotated_img, cmap="gray")
    plt.title(f"Rotation 90° {img_name}"), plt.xticks([]), plt.yticks([])
    plt.subplot(235), plt.imshow(r_magnitude_spectrum, cmap="gray")
    plt.title("Magnitude"), plt.xticks([]), plt.yticks([])
    plt.subplot(236), plt.imshow(r_phase_spectrum, cmap="gray")
    plt.title("Phase"), plt.xticks([]), plt.yticks([])

    plt.show()

if __name__ == "__main__":
    part_5()    