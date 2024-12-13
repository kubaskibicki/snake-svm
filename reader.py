import pickle
from model import game_state_to_data_sample


objects = []
with (open("gamin archive/kanyewest.pickle", "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break

print(len(objects[0]["data"]))

