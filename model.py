import pickle


from snake import Direction
import numpy as np

def game_state_to_data_sample(game_state: dict, block_size: int, bounds: tuple):
    snake_body = game_state["snake_body"]
    snake_head = snake_body[-1]
    print(snake_head)
    food = game_state["food"]
    direction = game_state["snake_direction"]
    output = [
        # Obstacles
        snake_head[1] + block_size == bounds[1], # obstacle below head,
        snake_head[0] + block_size == bounds[0],  # bottom right corner
        snake_head[1] == 0,  # obstacle N
        snake_head[0] == 0 and snake_head[1] == 0,  # top left corner
        # Food in same axis
        # Food in same x
        snake_head[0] == food[0],
        # Going left
        direction.value == Direction.LEFT.value,
        # Going right
        direction.value == Direction.RIGHT.value,
        # Going up
        direction.value == Direction.UP.value,
        # Going down
        direction.value == Direction.DOWN.value,
    ]
    return output


def prepare_samples(path):
    args, outs = [], []
    objects = []
    with (open(path, "rb")) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break

    for element in objects:
        bounds = element["bounds"]
        block_size = element["block_size"]
        i = 0
        for game_state in element["data"]:
            if i % 100 == 0:
                print(i)
            args.append(game_state_to_data_sample(game_state[0], block_size, bounds))
            outs.append(game_state[1].value)
            i += 1

    return np.array(args), np.array(outs)


class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.iterations = iterations
        self.w = None
        self.b = None

    def train(self, x, y):
        n_samples, n_features = x.shape
        self.w = np.zeros(n_features, float)
        self.b = 0
        for _ in range(self.iterations):
            for idx, x_i in enumerate(x):
                condition = y[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w - np.dot(x_i, y[idx]))
                    self.b -= self.learning_rate * y[idx]


    def predict(self, x):
        return np.dot(x, self.w) - self.b


class OneVsRestSVM:
    """
    Creates as many binary SVM's as there are classes (unique outputs) (4 classes in snake game)
    """
    def __init__(self, learning_rate=0.001, lambda_param=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.iterations = iterations
        self.w = None
        self.b = None
        self.all_models = {}


    def train(self, x, y):
        classes = np.unique(y)
        for single_class in classes:
            new_y = np.where(y == single_class, 1, -1)
            model = SVM(self.learning_rate, self.lambda_param, self.iterations)
            model.train(x, new_y)
            self.all_models[single_class] = model


    def predict(self, x):
        best_value = float("-inf")
        best_direction = None
        for value, model in self.all_models.items():
            result = model.predict(x)
            if result > best_value:
                best_value = result
                best_direction = value
        return best_direction

    def save(self, filename):
        with open(f"models/{filename}.pickle", 'wb') as f:
            pickle.dump(self.all_models, f)

    def load(self, filename):
        with open(f"models/{filename}.pickle", 'rb') as f:
            self.all_models = pickle.load(f)


def main():
    pass

if __name__ == "__main__":
    svm = OneVsRestSVM(0.01, 0.01, 3000)
    a, b = prepare_samples("gamin archive/circles.pickle")
    svm.train(a, b)
    svm.save("circ")

