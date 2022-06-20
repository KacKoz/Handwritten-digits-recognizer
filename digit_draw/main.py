import subprocess
import pygame

pygame.init()

screen = pygame.display.set_mode([560, 560])

running = True

squares = []
for i in range(28):
    squares.append([])
    for j in range(28):
        squares[i].append(0)


def draw(x, y, color):
    x = (x // 20) * 20
    y = (y // 20) * 20
    squares[y//20][x//20] = 255-color[0]
    if color == (0,0,0):
        for i in range(-1, 2, 1):
            for j in range(-1, 2, 1):
                if j == 0 and i == 0:
                    continue

                if (x//20)+i >= 0 and (x//20)+i < 28 and (y//20)+j >=0 and (y//20)+j < 28 and squares[(y//20)+j][(x//20)+i] == 0:
                    draw(x+(i*20), y+(j*20), (130,130,130))

    pygame.draw.rect(screen, color, (x, y, 20, 20))

screen.fill((255, 255, 255))

is_pressed = False
color = (0,0,0)
while running:
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            is_pressed = True
            (x, y) = pygame.mouse.get_pos()
            color = (0,0,0) if event.button == 1 else (255,255,255)
            draw(x, y, color)
        if event.type == pygame.MOUSEBUTTONUP:
            is_pressed = False
        if is_pressed and event.type == pygame.MOUSEMOTION:
            (x, y) = pygame.mouse.get_pos()
            draw(x, y, color)
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                with open("plik", "w") as file:
                    for i in squares:
                        for v in i:
                            file.write(str(v))
                            file.write("\n")
                cmd = "./digits_recognizer plik ../data/test.csv"
                ret = subprocess.call(cmd, shell=True)

            if event.key == pygame.K_SPACE:
                screen.fill((255, 255, 255))
                for i in squares:
                    for j in range(len(i)):
                        i[j] = 0


    pygame.display.flip()