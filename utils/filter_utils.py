"""Functions for converting between polynomial (monomial), Chebyshev, and Bernstein basis coefficients."""

import torch
from math import comb


def _cheb_basis_matrix(K):
    """Build (K+1)x(K+1) matrix M where row k holds monomial coefficients of T_k(x).

    Uses the recurrence: T_0=1, T_1=x, T_k = 2x*T_{k-1} - T_{k-2}.
    """
    M = torch.zeros(K + 1, K + 1)
    M[0, 0] = 1.0
    if K >= 1:
        M[1, 1] = 1.0
    for k in range(2, K + 1):
        M[k, 1:] += 2 * M[k - 1, :-1]
        M[k] -= M[k - 2]
    return M


def chev2poly(chev_coeffs):
    """Convert Chebyshev coefficients to polynomial (monomial) coefficients.

    Args:
        chev_coeffs: Tensor of shape (..., K+1)
    Returns:
        Tensor of shape (..., K+1) with monomial coefficients.
    """
    K = chev_coeffs.size(-1) - 1
    M = _cheb_basis_matrix(K).to(chev_coeffs)
    return chev_coeffs @ M


def poly2chev(poly_coeffs):
    """Convert polynomial (monomial) coefficients to Chebyshev coefficients.

    Args:
        poly_coeffs: Tensor of shape (..., K+1)
    Returns:
        Tensor of shape (..., K+1) with Chebyshev coefficients.
    """
    K = poly_coeffs.size(-1) - 1
    M = _cheb_basis_matrix(K).to(poly_coeffs)
    return poly_coeffs @ torch.linalg.inv(M)


def _bern_basis_matrix(K):
    """Build (K+1)x(K+1) matrix M where row k holds monomial coefficients of B_{k,K}(x).

    B_{k,n}(x) = C(n,k) * x^k * (1-x)^(n-k).
    Coefficient of x^m: C(n,k) * C(n-k, m-k) * (-1)^(m-k)  for k <= m <= n.
    """
    M = torch.zeros(K + 1, K + 1)
    for k in range(K + 1):
        for m in range(k, K + 1):
            M[k, m] = comb(K, k) * comb(K - k, m - k) * ((-1) ** (m - k))
    return M


def bern2poly(bern_coeffs):
    """Convert Bernstein coefficients to polynomial (monomial) coefficients.

    Args:
        bern_coeffs: Tensor of shape (..., K+1)
    Returns:
        Tensor of shape (..., K+1) with monomial coefficients.
    """
    K = bern_coeffs.size(-1) - 1
    M = _bern_basis_matrix(K).to(bern_coeffs)
    return bern_coeffs @ M


def poly2bern(poly_coeffs):
    """Convert polynomial (monomial) coefficients to Bernstein coefficients.

    Args:
        poly_coeffs: Tensor of shape (..., K+1)
    Returns:
        Tensor of shape (..., K+1) with Bernstein coefficients.
    """
    K = poly_coeffs.size(-1) - 1
    M = _bern_basis_matrix(K).to(poly_coeffs)
    return poly_coeffs @ torch.linalg.inv(M)

def replace_laplacian(poly_coeffs):
    """Given a polynomial function f(x), return the coefficients of f(1-x) in the polynomial basis."""

    K = poly_coeffs.size(-1) - 1
    new_coeffs = torch.zeros_like(poly_coeffs)
    for m in range(K + 1):
        for k in range(m, K + 1):
            new_coeffs[m] += poly_coeffs[k] * comb(k, m) * ((-1) ** (m))
    return new_coeffs



def write_out(coeffs, basis="p"):
    """p: polynomial, c: Chebyshev, b: Bernstein"""

    if basis.lower() == "c":
        coeffs = chev2poly(coeffs)
    elif basis.lower() == "b":
        coeffs = bern2poly(coeffs)

    """Print the polynomial in human-readable form."""
    terms = []
    for i, coeff in enumerate(coeffs):
        if abs(coeff) < 1e-8:
            continue
        if i == 0:
            terms.append(f"{coeff:.4f}")
        elif i == 1:
            terms.append(f"{coeff:.4f}*x")
        else:
            terms.append(f"{coeff:.4f}*x^{i}")
    poly_str = " + ".join(terms) if terms else "0"
    return f"P(x) = {poly_str}"

if __name__ == "__main__":
    # ---- Known conversions ----
    # T_0 = 1               =>  poly [1]         <-> chev [1]
    # T_1 = x               =>  poly [0,1]       <-> chev [0,1]
    # x^2 = (T_0 + T_2)/2   =>  poly [0,0,1]     <-> chev [0.5, 0, 0.5]
    # x^3 = (3T_1 + T_3)/4  =>  poly [0,0,0,1]   <-> chev [0, 0.75, 0, 0.25]
    # T_2 = 2x^2 - 1        =>  chev [0,0,1]     <-> poly [-1, 0, 2]
    # 1+2x+3x^2             =>  poly [1,2,3]     <-> chev [2.5, 2, 1.5]

    tests = [
        ("1 = T_0",
         torch.tensor([1.0]),
         torch.tensor([1.0])),
        ("x = T_1",
         torch.tensor([0.0, 1.0]),
         torch.tensor([0.0, 1.0])),
        ("x^2 = (T_0+T_2)/2",
         torch.tensor([0.0, 0.0, 1.0]),
         torch.tensor([0.5, 0.0, 0.5])),
        ("x^3 = (3T_1+T_3)/4",
         torch.tensor([0.0, 0.0, 0.0, 1.0]),
         torch.tensor([0.0, 0.75, 0.0, 0.25])),
        ("T_2 = 2x^2 - 1",
         torch.tensor([-1.0, 0.0, 2.0]),
         torch.tensor([0.0, 0.0, 1.0])),
        ("1 + 2x + 3x^2",
         torch.tensor([1.0, 2.0, 3.0]),
         torch.tensor([2.5, 2.0, 1.5])),
    ]

    print("=== poly2chev / chev2poly tests ===\n")
    all_pass = True
    for name, poly, chev in tests:
        got_chev = poly2chev(poly)
        got_poly = chev2poly(chev)
        ok_p2c = torch.allclose(got_chev, chev, atol=1e-6)
        ok_c2p = torch.allclose(got_poly, poly, atol=1e-6)
        status = "PASS" if (ok_p2c and ok_c2p) else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"[{status}] {name}")
        if not ok_p2c:
            print(f"  poly2chev: expected {chev.tolist()}, got {got_chev.tolist()}")
        if not ok_c2p:
            print(f"  chev2poly: expected {poly.tolist()}, got {got_poly.tolist()}")

    # Round-trip test with random coefficients
    rand_poly = torch.randn(5)
    roundtrip = chev2poly(poly2chev(rand_poly))
    ok_rt = torch.allclose(roundtrip, rand_poly, atol=1e-5)
    if not ok_rt:
        all_pass = False
    print(f"\n[{'PASS' if ok_rt else 'FAIL'}] Round-trip (random degree-4 poly)")

    # Batched test
    batch_poly = torch.randn(3, 4, 5)
    batch_rt = chev2poly(poly2chev(batch_poly))
    ok_batch = torch.allclose(batch_rt, batch_poly, atol=1e-5)
    if not ok_batch:
        all_pass = False
    print(f"[{'PASS' if ok_batch else 'FAIL'}] Batched round-trip (3x4 batch, degree 4)")

    # ---- Bernstein conversions (defined on [0,1]) ----
    # B_{0,2} = (1-x)^2 = 1-2x+x^2,  B_{1,2} = 2x(1-x) = 2x-2x^2,  B_{2,2} = x^2
    # bern [1,0,0] = (1-x)^2          -> poly [1, -2, 1]
    # bern [0,0,1] = x^2              -> poly [0, 0, 1]
    # bern [1,1,1] = 1 (constant)     -> poly [1, 0, 0]
    # bern [0,1,0] = 2x(1-x)          -> poly [0, 2, -2]
    # x = B_{0,1}*0 + B_{1,1}*1       -> poly [0, 1] <-> bern [0, 1]

    bern_tests = [
        ("bern [1,0,0] = (1-x)^2",
         torch.tensor([1.0, -2.0, 1.0]),
         torch.tensor([1.0, 0.0, 0.0])),
        ("bern [0,0,1] = x^2",
         torch.tensor([0.0, 0.0, 1.0]),
         torch.tensor([0.0, 0.0, 1.0])),
        ("bern [1,1,1] = 1",
         torch.tensor([1.0, 0.0, 0.0]),
         torch.tensor([1.0, 1.0, 1.0])),
        ("bern [0,1,0] = 2x(1-x)",
         torch.tensor([0.0, 2.0, -2.0]),
         torch.tensor([0.0, 1.0, 0.0])),
        ("x (degree 1)",
         torch.tensor([0.0, 1.0]),
         torch.tensor([0.0, 1.0])),
    ]

    print("\n=== poly2bern / bern2poly tests ===\n")
    for name, poly, bern in bern_tests:
        got_bern = poly2bern(poly)
        got_poly = bern2poly(bern)
        ok_p2b = torch.allclose(got_bern, bern, atol=1e-6)
        ok_b2p = torch.allclose(got_poly, poly, atol=1e-6)
        status = "PASS" if (ok_p2b and ok_b2p) else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"[{status}] {name}")
        if not ok_p2b:
            print(f"  poly2bern: expected {bern.tolist()}, got {got_bern.tolist()}")
        if not ok_b2p:
            print(f"  bern2poly: expected {poly.tolist()}, got {got_poly.tolist()}")

    # Bernstein round-trip
    rand_poly = torch.randn(5)
    roundtrip = bern2poly(poly2bern(rand_poly))
    ok_rt_b = torch.allclose(roundtrip, rand_poly, atol=1e-5)
    if not ok_rt_b:
        all_pass = False
    print(f"\n[{'PASS' if ok_rt_b else 'FAIL'}] Bernstein round-trip (random degree-4 poly)")

    # Bernstein batched
    batch_poly = torch.randn(3, 4, 5)
    batch_rt = bern2poly(poly2bern(batch_poly))
    ok_batch_b = torch.allclose(batch_rt, batch_poly, atol=1e-5)
    if not ok_batch_b:
        all_pass = False
    print(f"[{'PASS' if ok_batch_b else 'FAIL'}] Bernstein batched round-trip (3x4 batch, degree 4)")

    print(f"\n{'All tests passed!' if all_pass else 'Some tests FAILED.'}")
