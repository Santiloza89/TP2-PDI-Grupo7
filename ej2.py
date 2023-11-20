import cv2
import matplotlib.pyplot as plt
import numpy as np

#Funcion para ver imagenes
def imshow(img, new_fig=True, title=None, color_img=False, blocking=False, colorbar=True, ticks=False):
    if new_fig:
        plt.figure()
    if color_img:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    if not ticks:
        plt.xticks([]), plt.yticks([])
    if colorbar:
        plt.colorbar()
    if new_fig:
        plt.show(block=blocking)


#Lista con los nombres de nuestras imagenes
image_files = ['img01.png', 'img02.png', 'img03.png', 'img04.png', 'img05.png', 'img06.png', 'img07.png', 'img08.png', 'img09.png', 'img10.png', 'img11.png', 'img12.png']

#--------------------
# DETECCION PATENTES
#--------------------

# Lista para almacenar las patentes de cada imagen
todas_las_patentes = []

# Iteramos sobre cada imagen
for image_file in image_files:
    # Lee la imagen
    img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)

    # CANNY
    f_blur = cv2.GaussianBlur(img, ksize=(5, 5), sigmaX=1)
    gcan = cv2.Canny(f_blur, threshold1=0.2*255, threshold2=1*255)

    # Erosionamos para quitar líneas no deseadas
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    erosion = cv2.erode(gcan, kernel, iterations=1)
    
    # Dilatamos
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 5))
    Fd = cv2.dilate(erosion, kernel, iterations=1)
    

    # Clausura
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 2))
    closing = cv2.morphologyEx(Fd, cv2.MORPH_CLOSE, kernel)

    #Componentes conectadas
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closing, connectivity=8)

    canvas = img.copy()

    area_umbral_min = 800
    area_umbral_max = 2630

    aspect_ratio_min = 1.80
    aspect_ratio_max = 3.10

    patentes = []

    # Filtramos
    for label in range(1, num_labels):
        if (
            area_umbral_min < stats[label, cv2.CC_STAT_AREA] < area_umbral_max
            and aspect_ratio_min < stats[label, cv2.CC_STAT_WIDTH] / stats[label, cv2.CC_STAT_HEIGHT] < aspect_ratio_max
        ):
            x, y, w, h = stats[label, cv2.CC_STAT_LEFT], stats[label, cv2.CC_STAT_TOP], stats[label, cv2.CC_STAT_WIDTH], stats[label, cv2.CC_STAT_HEIGHT]
            cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 0), 2)
            patente = img[y:y+h, x:x+w]
            patentes.append(patente)

    todas_las_patentes.append(patentes)



# Índices a eliminar: patentes[1][0], patentes[2][1], patentes[8][0], patentes[10][0]
indices_a_eliminar = [(1, 0), (2, 1), (8, 0), (10, 0)]

todas_las_patentes = [
    [patente for j, patente in enumerate(patentes) if (i, j) not in indices_a_eliminar]
    for i, patentes in enumerate(todas_las_patentes)]



#---------------------
# DETECCION CARACTERES
#---------------------


# Iteramos sobre todas las patentes 
for i, patente in enumerate(todas_las_patentes):
    for j, patente in enumerate(patente):


        # Aplicamos 'Black Hat' para resaltar regiones oscuras en la imagen
        se = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 40))
        g3 = cv2.morphologyEx(patente, kernel=se, op=cv2.MORPH_BLACKHAT)


        # Binarizamos la imagen 
        _, pat_binarizada = cv2.threshold(g3, 70, 255, cv2.THRESH_BINARY_INV)

        #Hacemos una copia de la imagen original
        canvas = patente.copy()
        canvas = cv2.cvtColor(patente, cv2.COLOR_GRAY2BGR)

        #Hacemos componentes conectadas
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(pat_binarizada, connectivity=8)

         # Definimos umbrales
        area_umbral_min = 15
        area_umbral_max = 300

        aspect_ratio_min = 0.4
        aspect_ratio_max = 1.2

        # Iteramos sobre cada componente conectada y filtramos
        for label in range(1, num_labels):
            if (
                area_umbral_min < stats[label, cv2.CC_STAT_AREA] < area_umbral_max
                and aspect_ratio_min < stats[label, cv2.CC_STAT_WIDTH] / stats[label, cv2.CC_STAT_HEIGHT] < aspect_ratio_max
            ):
                # Obtenemos las coordenadas del cuadro delimitador de la componente conectada
                x, y, w, h = stats[label, cv2.CC_STAT_LEFT], stats[label, cv2.CC_STAT_TOP], stats[label, cv2.CC_STAT_WIDTH], stats[label, cv2.CC_STAT_HEIGHT]

                # Dibujamos un cuadro alrededor de la componente conectada 
                cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 255), 2)

                #Imprimos informacion
                #print(stats[label, cv2.CC_STAT_WIDTH], stats[label, cv2.CC_STAT_HEIGHT], "area: ", stats[label, cv2.CC_STAT_AREA])
                
        

        plt.imshow(canvas)
        plt.show()
        
        
        