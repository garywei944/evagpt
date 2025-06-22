import numpy as np


def low_rank(A, k):
    """Truncated SVD rank-k approximation of A."""
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    Uk = U[:, :k]
    sk = s[:k]
    Vtk = Vt[:k, :]
    return Uk @ (np.diag(sk) @ Vtk)


def hodlr(W, k, levels):
    """
    HODLR-style approximation:
      - Recursively split W into 4 blocks.
      - Approximate off-diagonals to rank k.
      - Recurse on the two diagonal blocks down to 'levels' deep.
    """
    n = W.shape[0]
    if levels == 0 or n <= 2 * k:
        return W.copy()  # no further splitting
    m = n // 2
    A11 = W[:m, :m]
    A12 = W[:m, m:]
    A21 = W[m:, :m]
    A22 = W[m:, m:]

    # low-rank approx of off-diagonals
    A12k = low_rank(A12, k)
    A21k = low_rank(A21, k)

    # recurse on diagonals
    A11h = hodlr(A11, k, levels - 1)
    A22h = hodlr(A22, k, levels - 1)

    # reassemble
    top = np.hstack([A11h, A12k])
    bottom = np.hstack([A21k, A22h])
    return np.vstack([top, bottom])


def main():
    np.random.seed(0)

    n = 256  # matrix size
    k = 32  # per-block rank
    levels = 3  # recursion depth

    # random test data
    W = np.random.randn(n, n)
    x = np.random.randn(n)

    # global rank-k approx
    W_glob = low_rank(W, k)

    # HODLR-style approx
    W_hodl = hodlr(W, k, levels)

    # print number of elements in each matrix
    print(f"Global SVD rank-{k}:    {W_glob.size} elements")
    print(f"HODLR  levels={levels}: {W_hodl.size} elements")

    # compute errors on Wx
    err_glob = np.linalg.norm(W @ x - W_glob @ x)
    err_hodl = np.linalg.norm(W @ x - W_hodl @ x)

    print(f"Global  SVD rank-{k}:    ‖Wx - W_glob x‖₂ = {err_glob:.6f}")
    print(f"HODLR  levels={levels}: ‖Wx - W_hodl x‖₂ = {err_hodl:.6f}")


if __name__ == "__main__":
    main()
