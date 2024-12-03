import pickle

objects = []
with (open("gamin archive/2024-12-03_08_34_59.pickle", "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break

for element in objects:
    print(element)