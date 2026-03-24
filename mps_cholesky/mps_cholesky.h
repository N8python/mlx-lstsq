#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace mx = mlx::core;

namespace mlx_mps_ext {

std::vector<mx::array>
mps_syrk_gram_rhs(const mx::array& a, const mx::array& b, mx::StreamOrDevice s = {});
mx::array
mps_cholesky_factor(const mx::array& gram, mx::StreamOrDevice s = {});
mx::array
mps_cholesky_solve(const mx::array& gram, const mx::array& rhs, mx::StreamOrDevice s = {});
mx::array cpu_triangular_solve(
    const mx::array& a,
    const mx::array& b,
    bool upper,
    bool transpose = false,
    bool unitriangular = false,
    mx::StreamOrDevice s = {});

class MPSCholeskyFactor : public mx::UnaryPrimitive {
 public:
  explicit MPSCholeskyFactor(mx::Stream stream) : mx::UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<mx::array>& inputs, mx::array& out) override;
  void eval_gpu(const std::vector<mx::array>& inputs, mx::array& out) override;

  const char* name() const override {
    return "MPSCholeskyFactor";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    return dynamic_cast<const MPSCholeskyFactor*>(&other) != nullptr;
  }

  std::vector<mx::Shape> output_shapes(const std::vector<mx::array>& inputs) override {
    return {inputs.at(0).shape()};
  }
};

class MPSCholeskySolve : public mx::UnaryPrimitive {
 public:
  explicit MPSCholeskySolve(mx::Stream stream) : mx::UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<mx::array>& inputs, mx::array& out) override;
  void eval_gpu(const std::vector<mx::array>& inputs, mx::array& out) override;

  const char* name() const override {
    return "MPSCholeskySolve";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    return dynamic_cast<const MPSCholeskySolve*>(&other) != nullptr;
  }

  std::vector<mx::Shape> output_shapes(const std::vector<mx::array>& inputs) override {
    return {inputs.at(1).shape()};
  }
};

class CPUTriangularSolve : public mx::UnaryPrimitive {
 public:
  CPUTriangularSolve(
      mx::Stream stream,
      bool upper,
      bool transpose,
      bool unitriangular)
      : mx::UnaryPrimitive(stream),
        upper_(upper),
        transpose_(transpose),
        unitriangular_(unitriangular) {}

  void eval_cpu(const std::vector<mx::array>& inputs, mx::array& out) override;
  void eval_gpu(const std::vector<mx::array>& inputs, mx::array& out) override;

  const char* name() const override {
    return "CPUTriangularSolve";
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    auto* p = dynamic_cast<const CPUTriangularSolve*>(&other);
    return p != nullptr && upper_ == p->upper_ && transpose_ == p->transpose_ &&
        unitriangular_ == p->unitriangular_;
  }

  std::vector<mx::Shape> output_shapes(const std::vector<mx::array>& inputs) override {
    return {inputs.at(1).shape()};
  }

  bool upper() const {
    return upper_;
  }

  bool transpose() const {
    return transpose_;
  }

  bool unitriangular() const {
    return unitriangular_;
  }

 private:
  bool upper_;
  bool transpose_;
  bool unitriangular_;
};

} // namespace mlx_mps_ext
