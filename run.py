# Cargar OpenCV
import os

try:
    import cv2
except ImportError:
    os.system('pip install opencv-python')

try:
    import sys
except ImportError:
    os.system('pip install os-sys')
    
try:
    import numpy as np
except ImportError:
    os.system('pip install numpy')
# Sirve para operar con cualquier dato numérico
try:
    import math
except ImportError:
    os.system("pip install python-math")

try:
    from pathfinding.core.grid import Grid
except ImportError:
    os.system("pip install pathfinding")
    
    
matrix = np.array([[1] * 20] * 20, dtype=object)
matrix_inicio = np.array([[1] * 20] * 20, dtype=object)
posh = 478
posx = 477
a = 0
r = 0
karel = ()
pos_karel_inicio = []
exit = ()
trays = []
min_distancia = 100
pos_min_beeper = ()
dire = 0
sum_matr = []
lista_pasos = []
cant_beepers = 0
custom_costs = {
    0: 1,   # costo para celdas vacías
    1: float('inf'),  # costo para celdas obstáculo
    2: 0,
}
# 1: suelo
# 2: Beepers
# 3: Muros
# 4: Player
# 5: Exit
# 6: Trays


class Punto:
    def __init__(self, x, y):
        self.x = x
        self.y = y


# Leer las imágenes que vamos a comparar
# Imagen sobre la que vamos a detectar si existe otra imagen
print("Exercisis: ")
os.system("dir exercisis")
print(" ")
path = input("Nombre de la imagen del ejercicio('solo escribe el nombre y su extension'): ")
img = cv2.imread("exercisis/"+path)
# Imagen que comprobamos si existe en la imagen Todo

# Orientación
print("Arriba = 0")
print("Derecha = 90")
print("Abajo = 180")
print("Izquierda = 270")

dire = int(input("En que dirección mira el personaje?('Numeros'): "))
di = ""

def elegirImg(dire):
    global di
    global sum_matr
    if dire == 0:
        sum_matr = [-1, 0]
        di = "Arriba"
        return ("Karel_up.png")
        
    if dire == 90:
        sum_matr = [0, 1]
        di = "Derecha"
        return ("Karel_right.png")
        
    if dire == 180:
        sum_matr = [-1, 0]
        di = "Abajo"
        return ("Karel_down.png")
        
    if dire == 270:
        di = "Izquierda"
        sum_matr = [0, -1]
        return ("Karel_left.png")
        


path = elegirImg(dire)
karel = cv2.imread('karel_pos/'+str(path))
beeper = cv2.imread("objectes/beeper.png")
exit = cv2.imread("objectes/exit.png")
muro = cv2.imread("objectes/muro.png")
tray = cv2.imread("objectes/tray.png")



#reescalar (tamaño obj 22x22)
img_rgb = cv2.resize(img, (477, 478), interpolation=cv2.INTER_CUBIC)
tray = cv2.resize(tray, (22,22), interpolation=cv2.INTER_CUBIC)

# Tamaño de la imagen 1.jpg
w, h = karel.shape[:-1]

levelWidth = 477
levelHeight = 478
# Función que sirve para detectar si una imagen está contenida en otra
res = cv2.matchTemplate(img_rgb, karel, cv2.TM_CCOEFF_NORMED)

# Umbral admitido
threshold = .7

# Si está dentro del umbral, crear un cuadrado sobre la imagen contenida en la imagen Todo
loc = np.where(res >= threshold)
for pt in zip(*loc[::-1]):  # Cambiar columnas y filas
    a = pt[0]*20/posx
    r = pt[1]*20/posh
    karel = (round(r), round(a))
    pos_karel_inicio = (round(r), round(a))
    matrix[round(r)][round(a)] = 4
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 0), 1)

# Función que sirve para detectar si una imagen está contenida en otra
w, h = beeper.shape[:-1]
res = cv2.matchTemplate(img_rgb, beeper, cv2.TM_CCOEFF_NORMED)

# Umbral admitido
threshold = .7

# Si está dentro del umbral, crear un cuadrado sobre la imagen contenida en la imagen Todo
loc = np.where(res >= threshold)
for pt in zip(*loc[::-1]):  # Cambiar columnas y filas
    a = pt[0]*20/posx
    r = pt[1]*20/posh
    matrix[round(r)][round(a)] = 2
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 0), 1)

# Función que sirve para detectar si una imagen está contenida en otra
w, h = exit.shape[:-1]
res = cv2.matchTemplate(img_rgb, exit, cv2.TM_CCOEFF_NORMED)

# Umbral admitido
threshold = .7

# Si está dentro del umbral, crear un cuadrado sobre la imagen contenida en la imagen Todo
loc = np.where(res >= threshold)
for pt in zip(*loc[::-1]):  # Cambiar columnas y filas
    a = pt[0]*20/posx
    r = pt[1]*20/posh
    matrix[round(r)][round(a)] = 5
    exit = [round(r), round(a)]
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 0), 1)

# Función que sirve para detectar si una imagen está contenida en otra
w, h = muro.shape[:-1]
res = cv2.matchTemplate(img_rgb, muro, cv2.TM_CCOEFF_NORMED)

# Umbral admitido
threshold = .7

# Si está dentro del umbral, crear un cuadrado sobre la imagen contenida en la imagen Todo
loc = np.where(res >= threshold)
a = 0
c = 0

for pt in zip(*loc[::-1]):  # Cambiar columnas y filas
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 0), 1)
    a = pt[0]*20/posx
    r = pt[1]*20/posh
    matrix[round(r)][round(a)] = 3
    
    
    
# Función que sirve para detectar si una imagen está contenida en otra
w, h = muro.shape[:-1]
res = cv2.matchTemplate(img_rgb, tray, cv2.TM_CCOEFF_NORMED)

# Umbral admitido
threshold = .8

# Si está dentro del umbral, crear un cuadrado sobre la imagen contenida en la imagen Todo
loc = np.where(res >= threshold)
a = 0
c = 0

for pt in zip(*loc[::-1]):  # Cambiar columnas y filas
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 0), 1)
    a = pt[0]*20/posx
    r = pt[1]*20/posh
    matrix[round(r)][round(a)] = 6

fil = int(input("Fila donde se encuentra el exit(0-19): "))
col = int(input("Columna donde se encuentra el exit(0-19): "))
exit = (fil,col)
matrix[fil][col] = 5 


matrix_inicio = matrix
def imprimir():
    global matrix

    for i in range(len(matrix)):
        for a in range(len(matrix[i])):
            if matrix[i][a] == 1:
                sys.stdout.write("  ")
            else:
                sys.stdout.write(" "+str(matrix[i][a]))
        sys.stdout.write("\n")
        
        
def imprimir_inicio():
    global matrix_inicio
    for i in range(len(matrix_inicio)):
        for a in range(len(matrix_inicio[i])):
            
            if matrix_inicio[i][a] == 1:
                sys.stdout.write("  ")
            else:
                sys.stdout.write(" "+str(matrix_inicio[i][a]))
        sys.stdout.write("\n")
imprimir_inicio()

# pathfinding
print(karel)
grid = Grid(matrix=matrix)

avoid_values = [3]


def cost_func(node):
    # Si el valor en la matriz está en los valores a evitar, devuelve un valor alto
    if node.value in avoid_values:
        return 999999
    else:
        return 1


def generarBeepers():
    global beepers
    global cant_beepers
    beepers = []
    cant_beepers = 0
    for i in range(len(matrix)):
        for a in range(len(matrix[i])):
            if matrix[i][a] == 2:
                beepers.append([i, a])
                cant_beepers += 1
    return(beepers)


def generarTrays():
    global trays
    trays = []
    for i in range(len(matrix)):
        for a in range(len(matrix[i])):
            if matrix[i][a] == 6:
                trays.append([i, a])
    return(trays)


beepers = generarBeepers()
trays = generarTrays()

start = (karel[1], karel[0])


def getNextMoves(x, y):
    return {
        'left':  [x-1, y],
        'right': [x+1, y],
        'up':  [x, y-1],
        'down':  [x, y+1]
    }.values()


def getShortestPath(level, startCoordinate, endCoordinate):
    global pasos
    pasos = []
    searchPaths = [[startCoordinate]]
    visitedCoordinates = [startCoordinate]

    while searchPaths != []:
        currentPath = searchPaths.pop(0)
        currentCoordinate = currentPath[-1]

        currentX, currentY = currentCoordinate

        if currentCoordinate == endCoordinate:
            return currentPath

        for nextCoordinate in getNextMoves(currentX, currentY):
            nextX, nextY = nextCoordinate

            if nextX < 0 or nextX >= levelWidth:
                continue

            if nextY < 0 or nextY >= levelHeight:
                continue

            if nextCoordinate in visitedCoordinates:
                continue
            try:
                if level[nextY][nextX] == ' ':
                    continue
            except IndexError:
                pass

            searchPaths.append(currentPath + [nextCoordinate])
            visitedCoordinates += [nextCoordinate]
            

            





def calcular_distancia(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)


def calc_en(beepers, start):
    global min_distancia
    global pos_min_beeper
    for element in beepers:
        punto = Punto(element[0], element[1])
        k = Punto(start[1], start[0])
        valor = calcular_distancia(k, punto)
        if valor < min_distancia:
            min_distancia = valor
        pos_min_beeper = (element[0], element[1])
    return(pos_min_beeper)

imprimir()
cont = True







mov = []
beepers = generarBeepers()
#calc pasos beepers
continuar = True
while cant_beepers > 0 and continuar == True:
    pos_karel = []
    for i in range(len(matrix)):
        for a in range(len(matrix)):
            if matrix[i][a] == 4:
                pos_karel = [i,a]
    beepers = generarBeepers()
    pos_min_beeper = calc_en(beepers, pos_karel)
    
    pasos = getShortestPath(matrix, pos_karel, list(pos_min_beeper))
    pasos.pop(0)
    for element in pasos:
        lista_pasos.append(list(element))
    
    print("Pos beeper mas cercano: ", pos_min_beeper)
    print("Cantidad de beepers: ",cant_beepers)
    print("Pos Karel: ", karel)
    print("Pasos a seguir ", pasos)
    if cant_beepers == 0:
        continuar = False
    for i in range(len(matrix)):
        for a in range(len(matrix)):
            if matrix[i][a] == 4:
                pos_karel = [i,a]  
    matrix[karel[0], karel[1]] = 1
    karel = pos_min_beeper
    matrix[karel[0], karel[1]] = 4
    for i in range(len(matrix)):
        for a in range(len(matrix)):
            if matrix[i][a] == 4:
                pos_karel = (i,a) 

pasos_exit = getShortestPath(matrix, list(pos_karel), list(exit))
pasos_exit.pop(0)

    
imprimir()

def rotar(di,ang):
    if di == 0 and ang < 0:
        di = 270
        return(di)
    elif di == 270 and ang > 0:
        di = 0
        return(di)
    else:
        di += ang
        return(di)
    
    

# función para mover el objeto
def calc_direcciones(lista_pasos, orientacion):
    
    
    direcciones = []
    dire = orientacion
    di = dire
    for i in range(len(lista_pasos)):
        
        movimiento = []
        try:
            #arriba                
            if dire == 0:
                if lista_pasos[i+1][0] < lista_pasos[i][0]:
                    movimiento.append("Front")
                if lista_pasos[i+1][0] > lista_pasos[i][0]:
                    movimiento.append("Left")
                    di = rotar(di,-90)
                    movimiento.append("Left")
                    movimiento.append("Front")
                    di = rotar(di,-90)
                if lista_pasos[i+1][1] < lista_pasos[i][1]:
                    movimiento.append("Left")
                    movimiento.append("Front")
                    di = rotar(di,-90)
                if lista_pasos[i+1][1] > lista_pasos[i][1]:
                    movimiento.append("Right")
                    movimiento.append("Front")
                    
                    di = rotar(di,90)
            
            #derecha
            if dire == 90:
                if lista_pasos[i+1][0] < lista_pasos[i][0]:
                    movimiento.append("Left")
                    movimiento.append("Front")
                    di = rotar(di,-90)
                if lista_pasos[i+1][0] > lista_pasos[i][0]:
                    movimiento.append("Right")
                    movimiento.append("Front")
                    di = rotar(di,90)
                if lista_pasos[i+1][1] < lista_pasos[i][1]:
                    movimiento.append("Left")
                    di = rotar(di,-90)
                    movimiento.append("Left")
                    movimiento.append("Front")
                    di = rotar(di,-90)
                if lista_pasos[i+1][1] > lista_pasos[i][1]:
                    movimiento.append("Front")
                    
            
            #abajo
            if dire == 180:
                if lista_pasos[i+1][0] < lista_pasos[i][0]:
                    movimiento.append("Left")
                    di = rotar(di,-90)
                    movimiento.append("Left")
                    movimiento.append("Front")
                    di = rotar(di,-90)
                if lista_pasos[i+1][0] > lista_pasos[i][0]:
                    movimiento.append("Front")
                if lista_pasos[i+1][1] < lista_pasos[i][1]:
                    movimiento.append("Right")
                    movimiento.append("Front")
                    di = rotar(di,90)
                if lista_pasos[i+1][1] > lista_pasos[i][1]:
                    movimiento.append("Left")
                    movimiento.append("Front")
                    di = rotar(di,-90)

            #izquierda              
            if dire == 270:
                if lista_pasos[i+1][0] < lista_pasos[i][0]:
                    movimiento.append("Right")
                    movimiento.append("Front")
                    di = rotar(di,90)
                if lista_pasos[i+1][0] > lista_pasos[i][0]:
                    movimiento.append("Left")
                    movimiento.append("Front")
                    di = rotar(di,-90)
                if lista_pasos[i+1][1] < lista_pasos[i][1]:
                    movimiento.append("Front")
                if lista_pasos[i+1][1] > lista_pasos[i][1]:
                    movimiento.append("Left")
                    di = rotar(di,-90)
                    movimiento.append("Left")
                    movimiento.append("Front")
                    di = rotar(di,-90)
                    
            print(lista_pasos[i]," ",movimiento," ",di," ",dire)
            for element in movimiento:
                direcciones.append(element)
            dire = di
        except IndexError:
            pass
        
        
        #cambiar los elif por if
    return(direcciones)




pos_karel_exit = karel

imprimir()
print("Pos beeper mas cercano: ", pos_min_beeper)
print("Pos Karel: ", karel)
print("pasos a seguir ", pasos)
lista_pasos.insert(0,list(pos_karel_inicio))

giros = calc_direcciones(lista_pasos,dire)
pasos_exit.insert(0,list(pos_karel_exit))
camino_exit = calc_direcciones(pasos_exit,dire)


filename = "resultado.txt"
with open(filename, 'w') as f:
    f.write("ss")
def pasarACodigo(lista_pasos):
    with open(filename, 'a') as f:
        for element in lista_pasos:
            
            if element == "Front":
                f.write("if(beepersPresent()){"+"\n")
                f.write("pickBeeper();"+"\n")
                f.write("}"+"\n")
                f.write("move();"+"\n")
            if element == "Left":
                f.write("turnLeft();"+"\n")
            if element == "Right":
                f.write("turnRight();"+"\n")

print(" ")
print(" ")
pasarACodigo(giros)
pasarACodigo(camino_exit)
with open(filename, 'a') as f:
    f.write("exit();")
# Guardar el resultado
imprimir()
cv2.imshow('resultado',img_rgb)
cv2.waitKey(0)
