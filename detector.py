import cv2

# Cargar la imagen de entrada y la imagen objetivo
img = cv2.imread('exercisis/img.png')
objetivo = cv2.imread('objectes/muro.png')
# Convertir las imágenes a escala de grises
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
objetivo_gray = cv2.cvtColor(objetivo, cv2.COLOR_BGR2GRAY)

# Inicializar el detector ORB y el objeto BFMatcher de OpenCV
orb = cv2.ORB_create()
bf = cv2.BFMatcher()

# Detectar y calcular características en la imagen objetivo y la imagen de entrada
kp_objetivo, des_objetivo = orb.detectAndCompute(objetivo_gray, None)
kp_img, des_img = orb.detectAndCompute(img_gray, None)

# Emparejar las características utilizando el objeto BFMatcher de OpenCV
matches = bf.match(des_objetivo, des_img)
matches = sorted(matches, key=lambda x: x.distance)

# Dibujar las mejores coincidencias en la imagen de entrada
img_matches = cv2.drawMatches(objetivo_gray, kp_objetivo, img_gray, kp_img, matches[:50], img_gray, flags=2)

# Mostrar la imagen de entrada con la imagen objetivo resaltada
cv2.imshow('Image', img_matches)
cv2.waitKey()