from collections import defaultdict

from metaflow import FlowSpec, step, conda_base

from config import *


@conda_base(libraries={'tensorflow': '2.4'}, python='3.8')
class MetricLearningFlow(FlowSpec):
    @step
    def start(self):
        import numpy as np
        from tensorflow.keras.datasets import cifar10

        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        self.x_train, self.y_train = x_train.astype("float32") / 255.0, np.squeeze(y_train)
        self.x_test, self.y_test = x_test.astype("float32") / 255.0, np.squeeze(y_test)

        self.class_idx_to_train_idxs = defaultdict(list)
        for y_train_idx, y in enumerate(self.y_train):
            self.class_idx_to_train_idxs[y].append(y_train_idx)

        self.class_idx_to_test_idxs = defaultdict(list)
        for y_test_idx, y in enumerate(self.y_test):
            self.class_idx_to_test_idxs[y].append(y_test_idx)

        self.next(self.train)

    @step
    def train(self):
        from dataset import AnchorPositivePairs
        from tensorflow import keras
        from model import create_model

        dataset = AnchorPositivePairs(1, self.x_train, self.class_idx_to_train_idxs)
        model = create_model()

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        )

        history = model.fit(dataset, epochs=1)
        self.loss = history.history["loss"]

        self.model = model.to_json()
        self.weights = model.get_weights()

        self.next(self.get_examples)

    @step
    def get_examples(self):
        import numpy as np
        from model import create_model

        model = create_model()
        model.set_weights(self.weights)

        embeddings = model.predict(self.x_test)
        gram_matrix = np.einsum("ae,be->ab", embeddings, embeddings)

        self.near_neighbours = np.argsort(gram_matrix.T)[:, -(near_neighbours_per_example + 1):]

        examples = np.empty(
            (
                num_collage_examples,
                near_neighbours_per_example + 1,
                height_width,
                height_width,
                3,
            ),
            dtype=np.float32,
        )
        for row_idx in range(num_collage_examples):
            examples[row_idx, 0] = self.x_test[row_idx]
            anchor_near_neighbours = reversed(self.near_neighbours[row_idx][:-1])
            for col_idx, nn_idx in enumerate(anchor_near_neighbours):
                examples[row_idx, col_idx + 1] = self.x_test[nn_idx]

        self.examples = examples
        self.next(self.end)

    @step
    def end(self):
        import numpy as np
        confusion_matrix = np.zeros((num_classes, num_classes))

        for class_idx in range(num_classes):
            example_idxs = self.class_idx_to_test_idxs[class_idx][:10]
            for y_test_idx in example_idxs:
                for nn_idx in self.near_neighbours[y_test_idx][:-1]:
                    nn_class_idx = self.y_test[nn_idx]
                    confusion_matrix[class_idx, nn_class_idx] += 1

        self.labels = [
            "Airplane",
            "Automobile",
            "Bird",
            "Cat",
            "Deer",
            "Dog",
            "Frog",
            "Horse",
            "Ship",
            "Truck",
        ]

        print(np.diag(confusion_matrix))


if __name__ == '__main__':
    MetricLearningFlow()
