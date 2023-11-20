import cv2
import numpy as np
import matplotlib.pyplot as plt

#Funcion para mostrar la imagen
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


#Vemos la imagen en escala de grises
f = cv2.imread('monedas.jpg', cv2.IMREAD_GRAYSCALE)



#Filtro pasa bajo
w1 = np.ones((2,2))/(2*2)
img1 = cv2.filter2D(f,-1,w1)


#Canny
f_blur = cv2.GaussianBlur(img1, ksize=(7, 7), sigmaX=1)
gcan = cv2.Canny(f_blur, threshold1=0.12*255, threshold2=0.62*255)



#Dilatamos
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (36,36))
Fd = cv2.dilate(gcan, kernel, iterations=1)




#Reconstrucción Morgológica
def imreconstruct(marker, mask, kernel=None):
    if kernel==None:
        kernel = np.ones((3,3), np.uint8)
    while True:
        expanded = cv2.dilate(marker, kernel)                               # Dilatacion
        expanded_intersection = cv2.bitwise_and(src1=expanded, src2=mask)   # Interseccion
        if (marker == expanded_intersection).all():                         # Finalizacion?
            break                                                           #
        marker = expanded_intersection
    return expanded_intersection


#Rellenamos
def imfillhole(img):
    # img: Imagen binaria de entrada. Valores permitidos: 0 (False), 255 (True).
    mask = np.zeros_like(img)                                                   # Genero mascara para...
    mask = cv2.copyMakeBorder(mask[1:-1,1:-1], 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=int(255)) # ... seleccionar los bordes.
    marker = cv2.bitwise_not(img, mask=mask)                # El marcador lo defino como el complemento de los bordes.
    img_c = cv2.bitwise_not(img)                            # La mascara la defino como el complemento de la imagen.
    img_r = imreconstruct(marker=marker, mask=img_c)        # Calculo la reconstrucción R_{f^c}(f_m)
    img_fh = cv2.bitwise_not(img_r)                         # La imagen con sus huecos rellenos es igual al complemento de la reconstruccion.
    return img_fh


#Rellenamos huecos de nuestra imagen
Fd = imfillhole(Fd)




#Apertura
B = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
Aop = cv2.morphologyEx(Fd, cv2.MORPH_OPEN, B)



#Componentes Conectadas
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(Aop)

labels_color = np.uint8(255/(num_labels-1)*labels)                  # Llevo el rango de valores a [0 255] para diferenciar mejor los colores
                                         
im_color = cv2.applyColorMap(labels_color, cv2.COLORMAP_JET)
im_color = cv2.cvtColor(im_color, cv2.COLOR_BGR2RGB)                # El mapa de color que se aplica está en BGR --> convierto a RGB

# --- Defino parametros para la clasificación -------------------------------------------
aux = np.zeros_like(labels)
labeled_image = cv2.merge([aux, aux, aux])

moneda50, moneda1, moneda10 = 0,0,0

#Lista donde iremos almacenando los diccionarios con datos de las monedas y dados
lista_objetos = []

# --- Clasificación ---------------------------------------------------------------------
# Clasifico en base al factor de forma
for i in range(1, num_labels):

    #Creamos diccionario para almacenar el objeto y sus datos
    objeto_dicc = {}

    # --- Selecciono el objeto actual -----------------------------------------
    obj = (labels == i).astype(np.uint8)

    # --- Calculo Rho ---------------------------------------------------------
    ext_contours, _ = cv2.findContours(obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = cv2.contourArea(ext_contours[0])
    perimeter = cv2.arcLength(ext_contours[0], True)
    rho = 4 * np.pi * area/(perimeter**2)

    fp = area / perimeter**2

    area = stats[i, cv2.CC_STAT_AREA] // 1000

    # Guardamos el objeto y su centroide
    objeto_dicc['labels'] = obj
    objeto_dicc['centroid'] = centroids[i]

    # --- Clasificamos -----------------------------------------------------------
    if 1/fp > 10 and 1/fp < 15:
      objeto_dicc['clase'] = 'moneda'
      if (area > 110):
        labeled_image[obj == 1, 0] = 255    # moneda de 50 (Rojo)
        moneda50 += 1
        objeto_dicc['valor'] = '0.5'
      elif (area < 110 and area > 100 ):
        labeled_image[obj == 1, 1] = 255    # moneda de 1 (Verde)
        moneda1 += 1
        objeto_dicc['valor'] = '1'
      else:
        labeled_image[obj == 1, 2] = 255    # moneda de 10 (azul)
        moneda10 += 1
        objeto_dicc['valor'] = '0.1'
    else:
        labeled_image[obj == 1, 1] = 50    # dados (verde oscuro)
        objeto_dicc['clase'] = 'dado'
    lista_objetos.append(objeto_dicc)


#--------------------
# Identificar dados
#--------------------

# Hacer una máscara con solo los dados
dados = [dicc for dicc in lista_objetos if dicc['clase'] == 'dado']

#placeholder
labeled_image = np.zeros_like(labels)

for dado in dados:
  numero_dado = 0
  # Decidimos usar Canny para identificar cada punto. Umbralizar también es una opción
  dado_canny = dado['labels'] * gcan
  # Hacemos componentes conectadas para separar cada punto del dado
  num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dado_canny)
  # Clasifico en base al factor de forma
  for i in range(1, num_labels):
      
      # --- Selecciono el objeto actual -----------------------------------------
      obj = (labels == i).astype(np.uint8)

       # --- Calculo Rho ---------------------------------------------------------
      ext_contours, _ = cv2.findContours(obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      area = cv2.contourArea(ext_contours[0])
      perimeter = cv2.arcLength(ext_contours[0], True)
      rho = 4 * np.pi * area/(perimeter**2)

      fp = area / perimeter**2

      # Removemos el caso donde fp = 0, es decir, una componente conectada de un solo punto
      if fp == 0:
        continue
      
      # Identificamos círculos y nos quedamos con aquellos más grandes (filtramos según área)
      if 1/fp > 10 and 1/fp < 15 and stats[i][4] > 80:
        labeled_image[obj == 1] = 255
        # Contamos los puntos
        numero_dado += 1
  dado['valor'] = str(numero_dado)


# Graficamos los dados con su número y las monedas con su tipo
aux = np.zeros_like(labels)
labeled_image = cv2.merge([aux, aux, aux])

# Parámetros para el texto
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 4
font_thickness = 15
font_color = (0, 0, 0)  
delta = 70

for objeto in lista_objetos:
  obj = (objeto['labels']).astype(np.uint8)
  if objeto['clase'] == 'moneda':
    if objeto['valor'] == '0.5':
      labeled_image[obj == 1, 0] = 255    # moneda de 50
      cv2.putText(labeled_image, objeto['valor'], (int(objeto['centroid'][0])-delta,int(objeto['centroid'][1])+delta), font, font_scale, font_color, font_thickness)
    elif objeto['valor'] == '1':
      labeled_image[obj == 1, 1] = 255    # moneda de 1
      cv2.putText(labeled_image, objeto['valor'], (int(objeto['centroid'][0])-delta,int(objeto['centroid'][1])+delta), font, font_scale, font_color, font_thickness)
    elif objeto['valor'] == '0.1':
      labeled_image[obj == 1, 2] = 255    # moneda de 10
      cv2.putText(labeled_image, objeto['valor'], (int(objeto['centroid'][0])-delta,int(objeto['centroid'][1])+delta), font, font_scale, font_color, font_thickness)
  if objeto['clase'] == 'dado':
    labeled_image[obj == 1, 1] = 50    # dado
    cv2.putText(labeled_image, objeto['valor'], (int(objeto['centroid'][0])-delta,int(objeto['centroid'][1])+delta), font, font_scale, font_color, font_thickness)

plt.imshow(labeled_image)
plt.show()
