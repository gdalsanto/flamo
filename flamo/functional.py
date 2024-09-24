def skew_matrix(X):
    r"""
    Generate a skew symmetric matrix from a given matrix X.
    """
    A = X.triu(1)
    return A - A.transpose(-1, -2)
