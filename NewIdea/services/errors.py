class KnowledgeError(Exception):
    """A rejected knowledge operation, with an HTTP status for the API adapter."""
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status
