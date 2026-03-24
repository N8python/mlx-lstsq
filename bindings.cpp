#include <nanobind/nanobind.h>
#include <nanobind/stl/variant.h>

#include "mps_cholesky/mps_cholesky.h"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_ext, m) {
  m.doc() = "Native MLX extensions backed by Apple MPS.";

  m.def(
      "mps_syrk_gram_rhs",
      [](const mlx::core::array& a, const mlx::core::array& b) {
        auto outputs = mlx_mps_ext::mps_syrk_gram_rhs(a, b);
        return nb::make_tuple(outputs.at(0), outputs.at(1));
      },
      "a"_a,
      "b"_a,
      R"(
        Compute the normal-equation front end on GPU with row-blocked MPS
        matmuls on a private command queue.

        Args:
            a (array): Design matrix with shape ``(m, n)``.
            b (array): Right-hand side with shape ``(m,)``.

        Returns:
            tuple[array, array]: ``(gram_upper, rhs)`` where
            ``rhs = a.T @ b`` and only the row-major upper triangle of
            ``gram_upper`` is guaranteed to be populated.
      )");

  m.def(
      "mps_cholesky_factor",
      [](const mlx::core::array& gram) {
        return mlx_mps_ext::mps_cholesky_factor(gram);
      },
      "gram"_a,
      R"(
        Compute an upper-triangular Cholesky factor ``U`` on GPU such that
        ``gram = U.T @ U``.

        Args:
            gram (array): Symmetric positive-definite matrix with shape ``(n, n)``.

        Returns:
            array: Upper-triangular Cholesky factor with shape ``(n, n)``.
      )");

  m.def(
      "cpu_triangular_solve",
      [](const mlx::core::array& a,
         const mlx::core::array& b,
         bool upper,
         bool transpose,
         bool unitriangular) {
        return mlx_mps_ext::cpu_triangular_solve(a, b, upper, transpose, unitriangular);
      },
      "a"_a,
      "b"_a,
      "upper"_a,
      "transpose"_a = false,
      "unitriangular"_a = false,
      R"(
        Solve a triangular system on CPU using a direct BLAS triangular solve.

        Args:
            a (array): Triangular matrix with shape ``(n, n)``.
            b (array): Right-hand side with shape ``(n,)`` or ``(n, k)``.
            upper (bool): Whether ``a`` is upper triangular.
            transpose (bool, optional): Whether to solve with ``a.T``. Default: ``False``.
            unitriangular (bool, optional): Whether to assume unit diagonal. Default: ``False``.

        Returns:
            array: Solution with the same shape as ``b``.
      )");

  m.def(
      "mps_cholesky_solve",
      [](const mlx::core::array& gram, const mlx::core::array& rhs) {
        return mlx_mps_ext::mps_cholesky_solve(gram, rhs);
      },
      "gram"_a,
      "rhs"_a,
      R"(
        Solve ``gram @ x = rhs`` using Apple MPS Cholesky factorization on GPU.

        Args:
            gram (array): Symmetric positive-definite matrix with shape ``(n, n)``.
            rhs (array): Right-hand side with shape ``(n,)`` or ``(n, k)``.

        Returns:
            array: Solution with the same shape as ``rhs``.
      )");
}
