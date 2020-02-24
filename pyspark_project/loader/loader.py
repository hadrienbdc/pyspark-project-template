from sklearn.datasets import fetch_20newsgroups


class DataLoader:
    def __init__(self, categories):
        self.categories = categories

    def load_data(self):
        twenty_train = fetch_20newsgroups(subset='train', categories=self.categories,
                                          shuffle=True, random_state=42)
        twenty_test = fetch_20newsgroups(subset='test', categories=self.categories,
                                         shuffle=True, random_state=42)

        return twenty_train, twenty_test
