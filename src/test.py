import random

stay = 0
changed = 0
for _ in range(100000):
    lista = [True, False, False]
    random.shuffle(lista)
    idx = random.randrange(3)
    choice = lista.pop(idx)

    if lista[0] == lista[1]:
        i = random.randrange(2)
        ch = lista.pop(i)

    elif lista[0]:
        ch = lista[0]
    elif lista[1]:
        ch = lista[1]

    if choice:
        stay += 1
    if ch:
        changed += 1




print(f'True: {stay}\nFalse: {changed}')
