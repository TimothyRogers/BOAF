class NoPredictException(Exception):
    """Exception when Clustering Cannot Predict
    """
    
    def __str__(self):
        return "This clustering approach has no predict method"