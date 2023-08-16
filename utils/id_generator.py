import random
import string


def serial_id_generator(prefix: str = '', length = 10, args: list[tuple[str, float]] = [], numeric = True):
    chars = list(string.digits)
    if numeric==False:
        chars += list(string.ascii_letters)
    id = prefix
    for i,t in enumerate(args):
        if i==0:
            id += '_'
        id +=  t[0]+str(t[1]) + '_'
    for _ in range(length):
        index = random.randint(0,len(chars)-1)
        id += chars[index]
    return id