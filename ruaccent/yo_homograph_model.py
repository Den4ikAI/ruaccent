from .token_classifier import TokenClassifier


class YoHomographModel(TokenClassifier):
    """Predicts whether an е/ё homograph should be written with ё."""

    def predict_yo_homographs(self, text):
        return self.predict(text)
