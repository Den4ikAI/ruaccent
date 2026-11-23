from .token_classifier import TokenClassifier


class StressUsagePredictorModel(TokenClassifier):
    """Predicts which words of a sentence should get a stress mark."""

    def predict_stress_usage(self, text):
        return self.predict(text)
