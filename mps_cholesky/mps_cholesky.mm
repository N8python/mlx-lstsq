#import <Foundation/Foundation.h>
#import <Accelerate/Accelerate.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "mlx/backend/cpu/encoder.h"
#include "mlx/backend/metal/device.h"
#include "mlx/transforms.h"

#include "mps_cholesky/mps_cholesky.h"

namespace mlx_mps_ext {

namespace {

id<MTLDevice> bridge_device(MTL::Device* device) {
  return (__bridge id<MTLDevice>)static_cast<void*>(device);
}

id<MTLCommandBuffer> bridge_command_buffer(MTL::CommandBuffer* command_buffer) {
  return (__bridge id<MTLCommandBuffer>)static_cast<void*>(command_buffer);
}

id<MTLComputePipelineState> bridge_compute_pipeline(MTL::ComputePipelineState* pipeline) {
  return (__bridge id<MTLComputePipelineState>)static_cast<void*>(pipeline);
}

id<MTLBuffer> bridge_buffer(const mx::array& arr) {
  return (__bridge id<MTLBuffer>)arr.buffer().ptr();
}

MPSDataType to_mps_dtype(const mx::Dtype& dtype) {
  if (dtype == mx::float32) {
    return MPSDataTypeFloat32;
  }
  throw std::runtime_error("MPSCholeskySolve only supports float32 inputs.");
}

NSUInteger rows_for_rhs(const mx::array& rhs) {
  return static_cast<NSUInteger>(rhs.shape(0));
}

NSUInteger cols_for_rhs(const mx::array& rhs) {
  return rhs.ndim() == 1 ? 1 : static_cast<NSUInteger>(rhs.shape(1));
}

NSUInteger row_bytes(const mx::array& arr) {
  if (arr.ndim() == 1) {
    return static_cast<NSUInteger>(arr.itemsize());
  }
  return static_cast<NSUInteger>(arr.strides(0) * arr.itemsize());
}

NSUInteger offset_bytes(const mx::array& arr) {
  return static_cast<NSUInteger>(arr.offset());
}

MPSMatrix* make_matrix_view(
    const mx::array& arr,
    NSUInteger rows,
    NSUInteger cols,
    NSUInteger row_bytes_override,
    NSUInteger offset_bytes_override,
    MPSDataType data_type);

MPSMatrix* make_matrix_buffer_view(
    id<MTLBuffer> buffer,
    NSUInteger rows,
    NSUInteger cols,
    NSUInteger row_bytes_override,
    NSUInteger offset_bytes_override,
    MPSDataType data_type);

MPSMatrix* make_matrix(
    const mx::array& arr,
    NSUInteger rows,
    NSUInteger cols,
    MPSDataType data_type) {
  return make_matrix_view(
      arr,
      rows,
      cols,
      row_bytes(arr),
      offset_bytes(arr),
      data_type);
}

MPSMatrix* make_matrix_view(
    const mx::array& arr,
    NSUInteger rows,
    NSUInteger cols,
    NSUInteger row_bytes_override,
    NSUInteger offset_bytes_override,
    MPSDataType data_type) {
  auto* descriptor = [MPSMatrixDescriptor
      matrixDescriptorWithRows:rows
                       columns:cols
                      rowBytes:row_bytes_override
                      dataType:data_type];
  return [[MPSMatrix alloc]
      initWithBuffer:bridge_buffer(arr)
              offset:offset_bytes_override
          descriptor:descriptor];
}

MPSMatrix* make_matrix_buffer_view(
    id<MTLBuffer> buffer,
    NSUInteger rows,
    NSUInteger cols,
    NSUInteger row_bytes_override,
    NSUInteger offset_bytes_override,
    MPSDataType data_type) {
  auto* descriptor = [MPSMatrixDescriptor
      matrixDescriptorWithRows:rows
                       columns:cols
                      rowBytes:row_bytes_override
                      dataType:data_type];
  return [[MPSMatrix alloc]
      initWithBuffer:buffer
              offset:offset_bytes_override
          descriptor:descriptor];
}

void check_gram(const mx::array& gram) {
  if (gram.ndim() != 2) {
    throw std::invalid_argument("gram must have shape (n, n).");
  }
  if (gram.shape(0) != gram.shape(1)) {
    throw std::invalid_argument("gram must be square.");
  }
}

void check_inputs(const mx::array& gram, const mx::array& rhs) {
  check_gram(gram);
  if (rhs.ndim() != 1 && rhs.ndim() != 2) {
    throw std::invalid_argument("rhs must have shape (n,) or (n, k).");
  }
  if (rhs.shape(0) != gram.shape(0)) {
    throw std::invalid_argument("rhs leading dimension must match gram.");
  }
}

void check_design_rhs_inputs(const mx::array& a, const mx::array& b) {
  if (a.ndim() != 2) {
    throw std::invalid_argument("a must have shape (m, n).");
  }
  if (b.ndim() != 1) {
    throw std::invalid_argument("b must have shape (m,).");
  }
  if (b.shape(0) != a.shape(0)) {
    throw std::invalid_argument("b length must match a.shape(0).");
  }
}

void check_triangular_inputs(const mx::array& a, const mx::array& b) {
  if (a.ndim() != 2) {
    throw std::invalid_argument("a must have shape (n, n).");
  }
  if (a.shape(0) != a.shape(1)) {
    throw std::invalid_argument("a must be square.");
  }
  if (b.ndim() != 1 && b.ndim() != 2) {
    throw std::invalid_argument("b must have shape (n,) or (n, k).");
  }
  if (b.shape(0) != a.shape(0)) {
    throw std::invalid_argument("b leading dimension must match a.");
  }
  if (a.dtype() != mx::float32 && a.dtype() != mx::float64) {
    throw std::invalid_argument("cpu_triangular_solve only supports float32 or float64 inputs.");
  }
  if (a.dtype() != b.dtype()) {
    throw std::invalid_argument("a and b must have the same dtype.");
  }
}

template <typename T>
void cblas_triangular_solve(
    CBLAS_ORDER order,
    CBLAS_UPLO uplo,
    CBLAS_TRANSPOSE trans,
    CBLAS_DIAG diag,
    int n,
    int nrhs,
    const T* a,
    int lda,
    T* b,
    int ldb);

template <>
void cblas_triangular_solve<float>(
    CBLAS_ORDER order,
    CBLAS_UPLO uplo,
    CBLAS_TRANSPOSE trans,
    CBLAS_DIAG diag,
    int n,
    int nrhs,
    const float* a,
    int lda,
    float* b,
    int ldb) {
  cblas_strsm(
      order,
      CblasLeft,
      uplo,
      trans,
      diag,
      n,
      nrhs,
      1.0f,
      a,
      lda,
      b,
      ldb);
}

template <>
void cblas_triangular_solve<double>(
    CBLAS_ORDER order,
    CBLAS_UPLO uplo,
    CBLAS_TRANSPOSE trans,
    CBLAS_DIAG diag,
    int n,
    int nrhs,
    const double* a,
    int lda,
    double* b,
    int ldb) {
  cblas_dtrsm(
      order,
      CblasLeft,
      uplo,
      trans,
      diag,
      n,
      nrhs,
      1.0,
      a,
      lda,
      b,
      ldb);
}

uint64_t kernel_key(NSUInteger order, NSUInteger nrhs = 0, NSUInteger mode = 0) {
  return (static_cast<uint64_t>(order) << 32) |
      (static_cast<uint64_t>(nrhs) << 8) | static_cast<uint64_t>(mode);
}

MPSMatrixSolveTriangular* get_triangular_kernel(
    id<MTLDevice> device,
    NSUInteger order,
    NSUInteger nrhs,
    BOOL upper,
    BOOL transpose) {
  static std::mutex mtx;
  static std::unordered_map<uint64_t, MPSMatrixSolveTriangular*> cache;
  std::lock_guard<std::mutex> lock(mtx);
  auto key = kernel_key(
      order,
      nrhs,
      (upper ? 1 : 0) | ((transpose ? 1 : 0) << 1));
  auto it = cache.find(key);
  if (it == cache.end()) {
    auto* kernel = [[MPSMatrixSolveTriangular alloc]
        initWithDevice:device
                 right:NO
                 upper:upper
             transpose:transpose
                  unit:NO
                 order:order
numberOfRightHandSides:nrhs
                 alpha:1.0];
    it = cache.emplace(key, kernel).first;
  }
  return it->second;
}

uint64_t matmul_key(NSUInteger lhs_cols, NSUInteger rhs_cols, NSUInteger k) {
  return (static_cast<uint64_t>(lhs_cols) << 43) |
      (static_cast<uint64_t>(rhs_cols) << 22) | (static_cast<uint64_t>(k) << 1);
}

MPSMatrixMultiplication* get_matmul_kernel(
    id<MTLDevice> device,
    NSUInteger result_rows,
    NSUInteger result_cols,
    NSUInteger interior_cols,
    BOOL transpose_left,
    BOOL transpose_right,
    BOOL accumulate) {
  static std::mutex mtx;
  static std::unordered_map<uint64_t, MPSMatrixMultiplication*> cache;
  std::lock_guard<std::mutex> lock(mtx);
  auto key = matmul_key(result_rows, result_cols, interior_cols) |
      (static_cast<uint64_t>(transpose_left ? 1 : 0) << 63) |
      (static_cast<uint64_t>(transpose_right ? 1 : 0) << 62) |
      (static_cast<uint64_t>(accumulate ? 1 : 0) << 61);
  auto it = cache.find(key);
  if (it == cache.end()) {
    auto* kernel = [[MPSMatrixMultiplication alloc]
        initWithDevice:device
         transposeLeft:transpose_left
        transposeRight:transpose_right
            resultRows:result_rows
         resultColumns:result_cols
       interiorColumns:interior_cols
                 alpha:1.0
                  beta:(accumulate ? 1.0 : 0.0)];
    it = cache.emplace(key, kernel).first;
  }
  return it->second;
}

std::string extension_binary_dir() {
  Dl_info info;
  if (dladdr(reinterpret_cast<const void*>(&extension_binary_dir), &info) == 0 ||
      info.dli_fname == nullptr) {
    throw std::runtime_error("Failed to resolve extension binary path.");
  }
  return std::filesystem::path(info.dli_fname).parent_path().string();
}

MTL::Library* get_extension_library(mx::metal::Device& metal_device) {
  return metal_device.get_library("mlx_ext", extension_binary_dir());
}

void encode_copy_contiguous_f32(
    mx::metal::Device& metal_device,
    mx::Stream stream,
    const mx::array& src,
    mx::array& dst) {
  auto* lib = get_extension_library(metal_device);
  auto* kernel = metal_device.get_kernel("copy_contiguous_f32", lib);
  auto& encoder = metal_device.get_command_encoder(stream.index);
  encoder.set_compute_pipeline_state(kernel);
  encoder.set_input_array(src, 0);
  encoder.set_output_array(dst, 1);
  auto n = static_cast<uint32_t>(src.size());
  encoder.set_bytes(n, 2);
  auto tgp =
      static_cast<NS::UInteger>(std::min<size_t>(kernel->maxTotalThreadsPerThreadgroup(), 256));
  encoder.dispatch_threads(
      MTL::Size(n, 1, 1),
      MTL::Size(tgp, 1, 1));
}

void encode_zero_i32(
    mx::metal::Device& metal_device,
    mx::Stream stream,
    mx::array& dst) {
  auto* lib = get_extension_library(metal_device);
  auto* kernel = metal_device.get_kernel("zero_i32", lib);
  auto& encoder = metal_device.get_command_encoder(stream.index);
  encoder.set_compute_pipeline_state(kernel);
  encoder.set_output_array(dst, 0);
  encoder.dispatch_threads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
}

void encode_zero_lower_triangle_f32(
    mx::metal::Device& metal_device,
    mx::Stream stream,
    mx::array& dst) {
  auto* lib = get_extension_library(metal_device);
  auto* kernel = metal_device.get_kernel("zero_lower_triangle_f32", lib);
  auto& encoder = metal_device.get_command_encoder(stream.index);
  encoder.set_compute_pipeline_state(kernel);
  encoder.set_output_array(dst, 0);
  const auto rows = static_cast<uint32_t>(dst.shape(0));
  const auto cols = static_cast<uint32_t>(dst.shape(1));
  encoder.set_bytes(rows, 1);
  encoder.set_bytes(cols, 2);
  encoder.dispatch_threads(MTL::Size(cols, rows, 1), MTL::Size(16, 16, 1));
}

id<MTLCommandQueue> get_private_command_queue(id<MTLDevice> device) {
  static std::mutex mtx;
  static std::unordered_map<void*, id<MTLCommandQueue>> cache;
  std::lock_guard<std::mutex> lock(mtx);
  auto key = (__bridge void*)device;
  auto it = cache.find(key);
  if (it == cache.end()) {
    auto queue = [device newCommandQueue];
    it = cache.emplace(key, queue).first;
  }
  return it->second;
}

NSUInteger getenv_uint(const char* name, NSUInteger fallback) {
  if (const char* value = std::getenv(name)) {
    char* end = nullptr;
    const auto parsed = std::strtoull(value, &end, 10);
    if (end != value && *end == '\0' && parsed > 0) {
      return static_cast<NSUInteger>(parsed);
    }
  }
  return fallback;
}

NSUInteger default_block_cols(NSUInteger n) {
  if (n >= 4096) {
    return std::min<NSUInteger>(1024, n);
  }
  if (n >= 512) {
    return std::min<NSUInteger>(384, n);
  }
  return n;
}

NSUInteger default_block_rows(NSUInteger n) {
  return n >= 4096 ? static_cast<NSUInteger>(8192) : static_cast<NSUInteger>(65536);
}

void encode_blocked_gram_rhs_private_mps(
    const mx::array& a,
    const mx::array& b,
    mx::array& gram,
    mx::array& rhs,
    mx::Stream stream) {
  const auto data_type = to_mps_dtype(a.dtype());
  const auto m = static_cast<NSUInteger>(a.shape(0));
  const auto n = static_cast<NSUInteger>(a.shape(1));
  const auto block_rows = getenv_uint("MLX_LSTSQ_GRAM_BLOCK_ROWS", default_block_rows(n));
  const auto block_cols = getenv_uint("MLX_LSTSQ_GRAM_BLOCK_COLS", default_block_cols(n));
  const auto itemsize = static_cast<NSUInteger>(a.itemsize());

  auto& metal_device = mx::metal::device(stream.device);
  auto device = bridge_device(metal_device.mtl_device());
  auto* lib = get_extension_library(metal_device);
  auto copy_block_pipeline =
      bridge_compute_pipeline(metal_device.get_kernel("copy_block_f32", lib));
  auto queue = get_private_command_queue(device);
  auto command_buffer = [queue commandBuffer];

  const auto a_row_bytes = row_bytes(a);
  const auto num_blocks = (n + block_cols - 1) / block_cols;
  for (NSUInteger i = 0; i < num_blocks; ++i) {
    const auto i0 = i * block_cols;
    const auto bi = std::min(block_cols, n - i0);
    auto* rhs_i = make_matrix_view(
        rhs,
        bi,
        1,
        static_cast<NSUInteger>(rhs.itemsize()),
        offset_bytes(rhs) + i0 * static_cast<NSUInteger>(rhs.itemsize()),
        data_type);

    std::vector<id<MTLBuffer>> scratch_buffers;
    std::vector<MPSMatrix*> scratch_matrices;
    std::vector<NSUInteger> block_widths;
    scratch_buffers.reserve(num_blocks - i);
    scratch_matrices.reserve(num_blocks - i);
    block_widths.reserve(num_blocks - i);

    for (NSUInteger j = i; j < num_blocks; ++j) {
      const auto bj = std::min(block_cols, n - j * block_cols);
      auto scratch = [device newBufferWithLength:bi * bj * itemsize
                                         options:MTLResourceStorageModeShared];
      scratch_buffers.push_back(scratch);
      scratch_matrices.push_back(
          make_matrix_buffer_view(scratch, bi, bj, bj * itemsize, 0, data_type));
      block_widths.push_back(bj);
    }

    for (NSUInteger row0 = 0; row0 < m; row0 += block_rows) {
      const auto bm = std::min(block_rows, m - row0);
      const auto a_i_offset =
          offset_bytes(a) + row0 * a_row_bytes + i0 * itemsize;
      const auto b_offset =
          offset_bytes(b) + row0 * static_cast<NSUInteger>(b.itemsize());
      MPSMatrix* a_i = make_matrix_view(a, bm, bi, a_row_bytes, a_i_offset, data_type);
      MPSMatrix* b_block =
          make_matrix_view(b, bm, 1, static_cast<NSUInteger>(b.itemsize()), b_offset, data_type);
      const BOOL accumulate = row0 != 0;

      auto* rhs_kernel = get_matmul_kernel(device, bi, 1, bm, YES, NO, accumulate);
      [rhs_kernel encodeToCommandBuffer:command_buffer
                             leftMatrix:a_i
                            rightMatrix:b_block
                           resultMatrix:rhs_i];

      for (NSUInteger j = i; j < num_blocks; ++j) {
        const auto local_j = j - i;
        const auto j0 = j * block_cols;
        const auto bj = block_widths[local_j];
        const auto a_j_offset =
            offset_bytes(a) + row0 * a_row_bytes + j0 * itemsize;
        MPSMatrix* a_j = make_matrix_view(a, bm, bj, a_row_bytes, a_j_offset, data_type);
        auto* gram_kernel =
            get_matmul_kernel(device, bi, bj, bm, YES, NO, accumulate);
        [gram_kernel encodeToCommandBuffer:command_buffer
                                leftMatrix:a_i
                               rightMatrix:a_j
                              resultMatrix:scratch_matrices[local_j]];
      }
    }

    auto copy_encoder = [command_buffer computeCommandEncoder];
    [copy_encoder setComputePipelineState:copy_block_pipeline];
    for (NSUInteger j = i; j < num_blocks; ++j) {
      const auto local_j = j - i;
      const auto j0 = j * block_cols;
      const auto bj = block_widths[local_j];
      const auto rows_u32 = static_cast<uint32_t>(bi);
      const auto cols_u32 = static_cast<uint32_t>(bj);
      const auto src_cols_u32 = static_cast<uint32_t>(bj);
      const auto dst_cols_u32 = static_cast<uint32_t>(gram.shape(1));
      const auto dst_row_offset_u32 = static_cast<uint32_t>(i0);
      const auto dst_col_offset_u32 = static_cast<uint32_t>(j0);
      [copy_encoder setBuffer:scratch_buffers[local_j] offset:0 atIndex:0];
      [copy_encoder setBuffer:bridge_buffer(gram) offset:offset_bytes(gram) atIndex:1];
      [copy_encoder setBytes:&rows_u32 length:sizeof(rows_u32) atIndex:2];
      [copy_encoder setBytes:&cols_u32 length:sizeof(cols_u32) atIndex:3];
      [copy_encoder setBytes:&src_cols_u32 length:sizeof(src_cols_u32) atIndex:4];
      [copy_encoder setBytes:&dst_cols_u32 length:sizeof(dst_cols_u32) atIndex:5];
      [copy_encoder setBytes:&dst_row_offset_u32 length:sizeof(dst_row_offset_u32) atIndex:6];
      [copy_encoder setBytes:&dst_col_offset_u32 length:sizeof(dst_col_offset_u32) atIndex:7];
      [copy_encoder dispatchThreads:MTLSizeMake(cols_u32, rows_u32, 1)
              threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
    }
    [copy_encoder endEncoding];
  }

  [command_buffer commit];
  [command_buffer waitUntilCompleted];
}

void encode_blocked_cholesky_lower_transpose(
    mx::metal::Device& metal_device,
    mx::Stream stream,
    mx::array& factor,
    mx::array& info) {
  auto* lib = get_extension_library(metal_device);
  auto* factor_diag = metal_device.get_kernel("factorDiagonalBlockL", lib);
  auto* apply_trsm = metal_device.get_kernel("applyTRSML", lib);
  auto* apply_syrk = metal_device.get_kernel("applySYRKL", lib);

  auto& encoder = metal_device.get_command_encoder(stream.index);
  const auto n = static_cast<uint32_t>(factor.shape(0));
  const auto nb = std::min<uint32_t>(32, n);
  const auto num_blocks = (n + nb - 1) / nb;
  const auto threadgroup = MTL::Size(32, 8, 1);

  for (uint32_t k = 0; k < num_blocks; ++k) {
    encoder.set_compute_pipeline_state(factor_diag);
    encoder.set_output_array(factor, 0);
    encoder.set_output_array(info, 1);
    encoder.set_bytes(n, 2);
    encoder.set_bytes(nb, 3);
    encoder.set_bytes(k, 4);
    encoder.dispatch_threadgroups(MTL::Size(1, 1, 1), threadgroup);

    if (k + 1 >= num_blocks) {
      continue;
    }

    const auto n_blocks_j = num_blocks - (k + 1);
    encoder.set_compute_pipeline_state(apply_trsm);
    encoder.set_output_array(factor, 0);
    encoder.set_bytes(n, 2);
    encoder.set_bytes(nb, 3);
    encoder.set_bytes(k, 4);
    encoder.dispatch_threadgroups(
        MTL::Size(1, n_blocks_j, 1),
        threadgroup);

    const auto n_pairs = n_blocks_j * (n_blocks_j + 1) / 2;
    encoder.set_compute_pipeline_state(apply_syrk);
    encoder.set_output_array(factor, 0);
    encoder.set_bytes(n, 2);
    encoder.set_bytes(nb, 3);
    encoder.set_bytes(k, 4);
    encoder.dispatch_threadgroups(
        MTL::Size(1, n_pairs, 1),
        threadgroup);
  }
}

} // namespace

std::vector<mx::array>
mps_syrk_gram_rhs(const mx::array& a, const mx::array& b, mx::StreamOrDevice s) {
  auto stream = mx::to_stream(s);
  check_design_rhs_inputs(a, b);

  if (stream.device.type != mx::Device::gpu) {
    throw std::invalid_argument("mps_syrk_gram_rhs requires a GPU stream/device.");
  }

  auto a_prepared = mx::contiguous(mx::astype(a, mx::float32, stream), false, stream);
  auto b_prepared = mx::contiguous(mx::astype(b, mx::float32, stream), false, stream);
  const auto n = a_prepared.shape(1);
  auto gram = mx::zeros({n, n}, mx::float32, stream);
  auto rhs = mx::zeros({n}, mx::float32, stream);
  mx::eval(a_prepared, b_prepared, gram, rhs);
  mx::synchronize(stream);
  encode_blocked_gram_rhs_private_mps(a_prepared, b_prepared, gram, rhs, stream);
  auto& metal_device = mx::metal::device(stream.device);
  encode_zero_lower_triangle_f32(metal_device, stream, gram);
  metal_device.end_encoding(stream.index);
  mx::synchronize(stream);
  return {gram, rhs};
}

mx::array
cpu_triangular_solve(
    const mx::array& a,
    const mx::array& b,
    bool upper,
    bool transpose,
    bool unitriangular,
    mx::StreamOrDevice s) {
  auto stream = mx::to_stream(s, mx::Device(mx::Device::cpu));
  check_triangular_inputs(a, b);

  if (stream.device.type != mx::Device::cpu) {
    throw std::invalid_argument("cpu_triangular_solve requires a CPU stream/device.");
  }
  if (!a.flags().row_contiguous) {
    throw std::invalid_argument("cpu_triangular_solve requires a row-contiguous a.");
  }
  if (!b.flags().row_contiguous) {
    throw std::invalid_argument("cpu_triangular_solve requires a row-contiguous b.");
  }

  return mx::array(
      b.shape(),
      b.dtype(),
      std::make_shared<CPUTriangularSolve>(stream, upper, transpose, unitriangular),
      {a, b});
}

void CPUTriangularSolve::eval_cpu(const std::vector<mx::array>& inputs, mx::array& out) {
  const auto& a = inputs.at(0);
  const auto& b = inputs.at(1);

  check_triangular_inputs(a, b);
  if (!a.flags().row_contiguous) {
    throw std::runtime_error("cpu_triangular_solve requires row-contiguous a.");
  }
  if (!b.flags().row_contiguous) {
    throw std::runtime_error("cpu_triangular_solve requires row-contiguous b.");
  }

  out.set_data(mx::allocator::malloc(out.nbytes()));

  auto& encoder = mx::cpu::get_command_encoder(stream());
  encoder.set_input_array(a);
  encoder.set_input_array(b);
  encoder.set_output_array(out);

  const int n = static_cast<int>(a.shape(0));
  const int nrhs = b.ndim() == 1 ? 1 : static_cast<int>(b.shape(1));
  const int lda = static_cast<int>(a.shape(1));
  const auto diag = unitriangular() ? CblasUnit : CblasNonUnit;
  const bool use_col_major_fast_path = (nrhs == 1);

  // Reinterpret the row-major triangular factor as column-major transposed data.
  // For nrhs == 1 this avoids packing A/B into Fortran scratch buffers and reaches
  // the same fast column-major BLAS path Torch prefers on CPU.
  const auto order = use_col_major_fast_path ? CblasColMajor : CblasRowMajor;
  const auto uplo = use_col_major_fast_path
      ? (upper() ? CblasLower : CblasUpper)
      : (upper() ? CblasUpper : CblasLower);
  const auto trans = use_col_major_fast_path
      ? (transpose() ? CblasNoTrans : CblasTrans)
      : (transpose() ? CblasTrans : CblasNoTrans);
  const int ldb = use_col_major_fast_path ? n : nrhs;

  if (a.dtype() == mx::float32) {
    const float* a_ptr = a.data<float>();
    const float* b_ptr = b.data<float>();
    float* out_ptr = out.data<float>();
    const size_t nbytes = out.nbytes();
    encoder.dispatch([=]() {
      std::memcpy(out_ptr, b_ptr, nbytes);
      cblas_triangular_solve<float>(order, uplo, trans, diag, n, nrhs, a_ptr, lda, out_ptr, ldb);
    });
  } else if (a.dtype() == mx::float64) {
    const double* a_ptr = a.data<double>();
    const double* b_ptr = b.data<double>();
    double* out_ptr = out.data<double>();
    const size_t nbytes = out.nbytes();
    encoder.dispatch([=]() {
      std::memcpy(out_ptr, b_ptr, nbytes);
      cblas_triangular_solve<double>(order, uplo, trans, diag, n, nrhs, a_ptr, lda, out_ptr, ldb);
    });
  } else {
    throw std::runtime_error("cpu_triangular_solve only supports float32 or float64.");
  }
}

void CPUTriangularSolve::eval_gpu(const std::vector<mx::array>&, mx::array&) {
  throw std::runtime_error("CPUTriangularSolve has no GPU implementation.");
}

mx::array
mps_cholesky_factor(const mx::array& gram, mx::StreamOrDevice s) {
  auto stream = mx::to_stream(s);
  check_gram(gram);

  if (stream.device.type != mx::Device::gpu) {
    throw std::invalid_argument("mps_cholesky_factor requires a GPU stream/device.");
  }

  auto gram_prepared = mx::contiguous(mx::astype(gram, mx::float32, stream), false, stream);

  return mx::array(
      gram_prepared.shape(),
      gram_prepared.dtype(),
      std::make_shared<MPSCholeskyFactor>(stream),
      {gram_prepared});
}

void MPSCholeskyFactor::eval_cpu(const std::vector<mx::array>&, mx::array&) {
  throw std::runtime_error("MPSCholeskyFactor has no CPU implementation.");
}

void MPSCholeskyFactor::eval_gpu(const std::vector<mx::array>& inputs, mx::array& out) {
  @autoreleasepool {
    const auto& gram = inputs.at(0);

    check_gram(gram);
    if (stream().device.type != mx::Device::gpu) {
      throw std::runtime_error("MPSCholeskyFactor requires a GPU stream.");
    }
    if (!gram.flags().row_contiguous) {
      throw std::runtime_error("gram must be row-contiguous.");
    }

    out.set_data(mx::allocator::malloc(out.nbytes()));
    auto info = mx::array(mx::allocator::malloc(sizeof(int32_t)), {1}, mx::int32);

    auto& metal_device = mx::metal::device(stream().device);
    metal_device.add_temporary(info, stream().index);

    encode_copy_contiguous_f32(metal_device, stream(), gram, out);
    encode_zero_i32(metal_device, stream(), info);
    encode_blocked_cholesky_lower_transpose(metal_device, stream(), out, info);
  }
}

mx::array
mps_cholesky_solve(const mx::array& gram, const mx::array& rhs, mx::StreamOrDevice s) {
  auto stream = mx::to_stream(s);
  check_inputs(gram, rhs);

  if (stream.device.type != mx::Device::gpu) {
    throw std::invalid_argument("mps_cholesky_solve requires a GPU stream/device.");
  }

  auto gram_prepared = mx::contiguous(mx::astype(gram, mx::float32, stream), false, stream);
  auto rhs_prepared = mx::contiguous(mx::astype(rhs, mx::float32, stream), false, stream);

  return mx::array(
      rhs_prepared.shape(),
      rhs_prepared.dtype(),
      std::make_shared<MPSCholeskySolve>(stream),
      {gram_prepared, rhs_prepared});
}

void MPSCholeskySolve::eval_cpu(const std::vector<mx::array>&, mx::array&) {
  throw std::runtime_error("MPSCholeskySolve has no CPU implementation.");
}

void MPSCholeskySolve::eval_gpu(const std::vector<mx::array>& inputs, mx::array& out) {
  @autoreleasepool {
    const auto& gram = inputs.at(0);
    const auto& rhs = inputs.at(1);

    check_inputs(gram, rhs);
    if (stream().device.type != mx::Device::gpu) {
      throw std::runtime_error("MPSCholeskySolve requires a GPU stream.");
    }
    if (!gram.flags().row_contiguous) {
      throw std::runtime_error("gram must be row-contiguous.");
    }
    if (rhs.ndim() == 2 && !rhs.flags().row_contiguous) {
      throw std::runtime_error("rhs must be row-contiguous.");
    }

    out.set_data(mx::allocator::malloc(out.nbytes()));
    auto factor = mx::array(mx::allocator::malloc(gram.nbytes()), gram.shape(), gram.dtype());
    auto y = mx::array(mx::allocator::malloc(out.nbytes()), out.shape(), out.dtype());
    auto info = mx::array(mx::allocator::malloc(sizeof(int32_t)), {1}, mx::int32);

    auto& metal_device = mx::metal::device(stream().device);
    metal_device.add_temporary(factor, stream().index);
    metal_device.add_temporary(y, stream().index);
    metal_device.add_temporary(info, stream().index);

    encode_copy_contiguous_f32(metal_device, stream(), gram, factor);
    encode_zero_i32(metal_device, stream(), info);
    encode_blocked_cholesky_lower_transpose(metal_device, stream(), factor, info);

    metal_device.end_encoding(stream().index);

    auto* command_buffer_cpp = metal_device.get_command_buffer(stream().index);
    auto command_buffer = bridge_command_buffer(command_buffer_cpp);
    auto device = bridge_device(metal_device.mtl_device());
    auto data_type = to_mps_dtype(gram.dtype());

    const auto n = static_cast<NSUInteger>(gram.shape(0));
    const auto nrhs = cols_for_rhs(rhs);

    MPSMatrix* factor_matrix = make_matrix(factor, n, n, data_type);
    MPSMatrix* rhs_matrix = make_matrix(rhs, rows_for_rhs(rhs), nrhs, data_type);
    MPSMatrix* y_matrix = make_matrix(y, rows_for_rhs(rhs), nrhs, data_type);
    MPSMatrix* out_matrix = make_matrix(out, rows_for_rhs(rhs), nrhs, data_type);

    auto* lower_solve = get_triangular_kernel(device, n, nrhs, YES, YES);
    [lower_solve
        encodeToCommandBuffer:command_buffer
                  sourceMatrix:factor_matrix
           rightHandSideMatrix:rhs_matrix
                solutionMatrix:y_matrix];

    auto* upper_solve = get_triangular_kernel(device, n, nrhs, YES, NO);
    [upper_solve
        encodeToCommandBuffer:command_buffer
                  sourceMatrix:factor_matrix
           rightHandSideMatrix:y_matrix
                solutionMatrix:out_matrix];
  }
}

} // namespace mlx_mps_ext
