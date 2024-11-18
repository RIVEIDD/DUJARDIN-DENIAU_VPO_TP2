# pylint: disable=no-member

import cv2
import os
import numpy


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