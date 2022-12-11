import os
import cv2
from PIL import Image, ImageGrab


def redimention(dataset):
    # Diretórios com as imagens
    # paths = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    #          'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    paths = ['A']

    # Quantidade de imagens em cada diretório
    # sizes = [7469, 4526, 11833, 5341, 5785, 10622, 2964, 3673, 13994, 4388, 2850, 5886, 10487,
    #          9588, 29139, 9744, 3018, 5882, 24272, 11396, 14604, 5433, 5501, 3203, 5541, 2075]

    sizes = [3]

    i = -1
    for character in paths:
        i += 1
        images = os.listdir(dataset + '/' + character)
        for j in range(sizes[i]):
            path = dataset + '/' + character + '/' + images[j]
            print(path, j)
            image = Image.open(path)
            img_resized = image.resize((30, 30))
            # img_resized.save('./results/' + character + '/' + images[j])
            img_resized.save('./testando/' + character + '/' + images[j])


def pixels(dataset):
    # Diretório com as imagens redimensionadas
    paths = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
             'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

    # paths = ['A']

    # Quantidade de imagens em cada diretório
    sizes = [7469, 4526, 11833, 5341, 5785, 10622, 2964, 3673, 13994, 4388, 2850, 5886, 10487,
             9588, 29139, 9744, 3018, 5882, 24272, 11396, 14604, 5433, 5501, 3203, 5541, 2075]

    # sizes = [10]

    # Definição das variáveis de contagem de pixel
    sumBlack = 0
    sumWhite = 0

    i = -1
    output = open('results.txt', 'w')
    output.write("Classe    Pixels Brancos    Pixels Pretos\n")
    for character in paths:
        i += 1
        images = os.listdir(dataset + '/' + character)
        for j in range(sizes[i]):
            file = dataset + '/' + character + '/' + images[j]
            image = Image.open(file, 'r')
            width, height = image.size
            pixels = list(image.getdata())
            image_pixels = width * height
            for pos in range(image_pixels):
                if (pixels[pos] == 1):
                    sumBlack += 1
                else:
                    sumWhite += 1

            characterSpace = character + '             '
            output.write(characterSpace)
            sumWhite = str(sumWhite) + '                 '
            output.write(str(sumWhite))
            output.write(str(sumBlack))
            output.write('\n')
            sumBlack = 0
            sumWhite = 0
