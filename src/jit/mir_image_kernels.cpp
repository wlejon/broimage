#include "broimage/jit/mir_image_kernels.h"

#include <brass/gpu/vulkan_driver.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/target/target.hpp>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace broimage::jit {

namespace {

constexpr uint32_t kGpuBlockSizeX = 64;

const char* default_kernel_name(MirKernelKind kind, MirExecTarget target) {
    const bool gpu = (target == MirExecTarget::Gpu);
    switch (kind) {
        case MirKernelKind::AffineClampF32:
            return gpu ? "mir_affine_clamp_f32_gpu" : "mir_affine_clamp_f32_cpu";
        case MirKernelKind::RgbToGrayF32:
            return gpu ? "mir_rgb_to_gray_f32_gpu" : "mir_rgb_to_gray_f32_cpu";
        case MirKernelKind::ChwNormalizeF32:
            return gpu ? "mir_chw_normalize_f32_gpu" : "mir_chw_normalize_f32_cpu";
        case MirKernelKind::SeparableBlurHF32:
            return gpu ? "mir_separable_blur_h_f32_gpu" : "mir_separable_blur_h_f32_cpu";
        case MirKernelKind::SeparableBlurVF32:
            return gpu ? "mir_separable_blur_v_f32_gpu" : "mir_separable_blur_v_f32_cpu";
        case MirKernelKind::BilinearResizeF32:
            return gpu ? "mir_bilinear_resize_f32_gpu" : "mir_bilinear_resize_f32_cpu";
    }
    return "mir_image_kernel";
}

brass::Value* emit_gpu_global_tid_x(brass::Builder& b) {
    brass::Value* ctaid = b.build_call("gpu_ctaid_x", brass::Type::i32());
    brass::Value* ntid = b.build_call("gpu_ntid_x", brass::Type::i32());
    brass::Value* tid = b.build_call("gpu_tid_x", brass::Type::i32());
    return b.build_add(b.build_mul(ctaid, ntid), tid);
}

brass::Value* emit_clamp_i32(brass::codegen::KernelBuilder& kb,
                             brass::Value* v,
                             brass::Value* lo,
                             brass::Value* hi) {
    brass::Builder& b = kb.builder();
    brass::Value* lt = b.build_slt(v, lo);
    brass::Value* t = b.build_select(lt, lo, v);
    brass::Value* gt = b.build_sgt(t, hi);
    return b.build_select(gt, hi, t);
}

brass::Value* emit_clamp_f32(brass::codegen::KernelBuilder& kb,
                             brass::Value* v,
                             brass::Value* lo,
                             brass::Value* hi) {
    brass::Builder& b = kb.builder();
    return b.build_fmin_f32(b.build_fmax_f32(v, lo), hi);
}

// ---------------------------------------------------------------------------
// Single-source per-element / per-pixel MIR body emitters
// ---------------------------------------------------------------------------

void emit_affine_clamp_elem(
    brass::codegen::KernelBuilder& kb,
    brass::Value* src,
    brass::Value* dst,
    brass::Value* idx,
    brass::Value* alpha,
    brass::Value* beta,
    brass::Value* lo,
    brass::Value* hi
) {
    brass::Value* v = kb.load_f32_indexed(src, idx, 4, 0);
    brass::Value* aff = kb.fma(v, alpha, beta);
    brass::Value* clamped = emit_clamp_f32(kb, aff, lo, hi);
    kb.store_f32_indexed(dst, idx, clamped, 4, 0);
}

void emit_rgb_to_gray_pixel(
    brass::codegen::KernelBuilder& kb,
    brass::Value* src_rgb,
    brass::Value* dst_gray,
    brass::Value* pix
) {
    brass::Builder& b = kb.builder();
    brass::Value* c1 = kb.const_i32(1);
    brass::Value* c2 = kb.const_i32(2);
    brass::Value* c3 = kb.const_i32(3);

    brass::Value* base = b.build_mul(pix, c3);
    brass::Value* r = kb.load_f32_indexed(src_rgb, base, 4, 0);
    brass::Value* g = kb.load_f32_indexed(src_rgb, b.build_add(base, c1), 4, 0);
    brass::Value* bl = kb.load_f32_indexed(src_rgb, b.build_add(base, c2), 4, 0);

    brass::Value* wr = kb.const_f32(0.299f);
    brass::Value* wg = kb.const_f32(0.587f);
    brass::Value* wb = kb.const_f32(0.114f);

    brass::Value* acc = kb.fma(r, wr, b.build_mul(g, wg));
    brass::Value* gray = kb.fma(bl, wb, acc);
    kb.store_f32_indexed(dst_gray, pix, gray, 4, 0);
}

void emit_chw_normalize_elem(
    brass::codegen::KernelBuilder& kb,
    brass::Value* src,
    brass::Value* dst,
    brass::Value* elem_idx,
    brass::Value* mean,
    brass::Value* inv_std
) {
    brass::Builder& b = kb.builder();
    brass::Value* v = kb.load_f32_indexed(src, elem_idx, 4, 0);
    brass::Value* diff = b.build_sub(v, mean);
    brass::Value* norm = b.build_mul(diff, inv_std);
    kb.store_f32_indexed(dst, elem_idx, norm, 4, 0);
}

void emit_separable_blur_pixel(
    brass::codegen::KernelBuilder& kb,
    bool horizontal,
    brass::Value* src,
    brass::Value* dst,
    brass::Value* weights,
    brass::Value* width,
    brass::Value* height,
    brass::Value* radius,
    brass::Value* x,
    brass::Value* y
) {
    brass::Builder& b = kb.builder();
    brass::Value* c0 = kb.const_i32(0);
    brass::Value* c1 = kb.const_i32(1);
    brass::Value* f0 = kb.const_f32(0.0f);

    brass::Value* neg_r = b.build_sub(c0, radius);
    brass::Value* r_plus_1 = b.build_add(radius, c1);
    brass::Value* axis_max = b.build_sub(horizontal ? width : height, c1);

    brass::Value* sum = kb.for_range_reduce(neg_r, r_plus_1, c1, f0, [&](brass::Value* k, brass::Value* acc) {
        brass::Value* coord = b.build_add(horizontal ? x : y, k);
        brass::Value* clamped = emit_clamp_i32(kb, coord, c0, axis_max);
        brass::Value* sx = horizontal ? clamped : x;
        brass::Value* sy = horizontal ? y : clamped;
        brass::Value* src_idx = b.build_add(b.build_mul(sy, width), sx);
        brass::Value* sample = kb.load_f32_indexed(src, src_idx, 4, 0);
        brass::Value* w = kb.load_f32_indexed(weights, b.build_add(k, radius), 4, 0);
        return b.build_add(acc, b.build_mul(sample, w));
    });

    brass::Value* dst_idx = b.build_add(b.build_mul(y, width), x);
    kb.store_f32_indexed(dst, dst_idx, sum, 4, 0);
}

void emit_bilinear_resize_pixel(
    brass::codegen::KernelBuilder& kb,
    brass::Value* src,
    brass::Value* dst,
    brass::Value* src_w,
    brass::Value* src_h,
    brass::Value* dst_w,
    brass::Value* xs,
    brass::Value* ys,
    brass::Value* sw_minus_1,
    brass::Value* sh_minus_1,
    brass::Value* x,
    brass::Value* y
) {
    brass::Builder& b = kb.builder();
    brass::Value* c0 = kb.const_i32(0);
    brass::Value* c1 = kb.const_i32(1);
    brass::Value* f0 = kb.const_f32(0.0f);
    brass::Value* f1 = kb.const_f32(1.0f);
    brass::Value* fhalf = kb.const_f32(0.5f);

    brass::Value* xf = b.build_sitofp_f32_i32(x);
    brass::Value* yf = b.build_sitofp_f32_i32(y);

    brass::Value* fx = b.build_sub(b.build_mul(b.build_add(xf, fhalf), xs), fhalf);
    brass::Value* fy = b.build_sub(b.build_mul(b.build_add(yf, fhalf), ys), fhalf);

    brass::Value* floor_x = b.build_floor_f32(fx);
    brass::Value* floor_y = b.build_floor_f32(fy);

    brass::Value* ix0 = b.build_fptosi_i32_f32(floor_x);
    brass::Value* iy0 = b.build_fptosi_i32_f32(floor_y);
    brass::Value* ix1 = b.build_add(ix0, c1);
    brass::Value* iy1 = b.build_add(iy0, c1);

    brass::Value* x0 = emit_clamp_i32(kb, ix0, c0, sw_minus_1);
    brass::Value* x1 = emit_clamp_i32(kb, ix1, c0, sw_minus_1);
    brass::Value* y0 = emit_clamp_i32(kb, iy0, c0, sh_minus_1);
    brass::Value* y1 = emit_clamp_i32(kb, iy1, c0, sh_minus_1);

    brass::Value* tx = emit_clamp_f32(kb, b.build_sub(fx, floor_x), f0, f1);
    brass::Value* ty = emit_clamp_f32(kb, b.build_sub(fy, floor_y), f0, f1);
    tx = b.build_select(b.build_slt(ix0, c0), f0, tx);
    ty = b.build_select(b.build_slt(iy0, c0), f0, ty);
    tx = b.build_select(b.build_sge(ix1, src_w), f1, tx);
    ty = b.build_select(b.build_sge(iy1, src_h), f1, ty);

    brass::Value* y0_row = b.build_mul(y0, src_w);
    brass::Value* y1_row = b.build_mul(y1, src_w);

    brass::Value* c00 = kb.load_f32_indexed(src, b.build_add(y0_row, x0), 4, 0);
    brass::Value* c10 = kb.load_f32_indexed(src, b.build_add(y0_row, x1), 4, 0);
    brass::Value* c01 = kb.load_f32_indexed(src, b.build_add(y1_row, x0), 4, 0);
    brass::Value* c11 = kb.load_f32_indexed(src, b.build_add(y1_row, x1), 4, 0);

    brass::Value* omtx = b.build_sub(f1, tx);
    brass::Value* omty = b.build_sub(f1, ty);
    brass::Value* top = kb.fma(c10, tx, b.build_mul(c00, omtx));
    brass::Value* bot = kb.fma(c11, tx, b.build_mul(c01, omtx));
    brass::Value* res = kb.fma(bot, ty, b.build_mul(top, omty));

    brass::Value* dst_idx = b.build_add(b.build_mul(y, dst_w), x);
    kb.store_f32_indexed(dst, dst_idx, res, 4, 0);
}

// ---------------------------------------------------------------------------
// Kernel builders per MirKernelKind
// ---------------------------------------------------------------------------

brass::Function* build_affine_clamp_kernel(brass::Module& mod, MirExecTarget target, const char* name) {
    using namespace brass;
    Function* fn = mod.create_function(
        name, Type::void_type(),
        {Type::ptr(), Type::ptr(), Type::i32(), Type::f32(), Type::f32(), Type::f32(), Type::f32()}
    );
    codegen::KernelBuilder kb(mod, fn);
    Builder& b = kb.builder();
    BasicBlock* entry = b.append_block("entry");
    b.position_at_end(entry);

    Value* src = b.add_block_param(entry, Type::ptr());
    Value* dst = b.add_block_param(entry, Type::ptr());
    Value* total = b.add_block_param(entry, Type::i32());
    Value* alpha = b.add_block_param(entry, Type::f32());
    Value* beta = b.add_block_param(entry, Type::f32());
    Value* lo = b.add_block_param(entry, Type::f32());
    Value* hi = b.add_block_param(entry, Type::f32());

    if (target == MirExecTarget::CpuSimd) {
        kb.for_range(kb.const_i32(0), total, kb.const_i32(1), [&](Value* idx) {
            emit_affine_clamp_elem(kb, src, dst, idx, alpha, beta, lo, hi);
        });
    } else {
        Value* gid = emit_gpu_global_tid_x(b);
        kb.if_then(b.build_slt(gid, total), [&] {
            emit_affine_clamp_elem(kb, src, dst, gid, alpha, beta, lo, hi);
        });
    }
    b.build_ret_void();
    fn->rebuild_cfg_predecessors();
    return fn;
}

brass::Function* build_rgb_to_gray_kernel(brass::Module& mod, MirExecTarget target, const char* name) {
    using namespace brass;
    Function* fn = mod.create_function(
        name, Type::void_type(),
        {Type::ptr(), Type::ptr(), Type::i32(), Type::i32()}
    );
    codegen::KernelBuilder kb(mod, fn);
    Builder& b = kb.builder();
    BasicBlock* entry = b.append_block("entry");
    b.position_at_end(entry);

    Value* src = b.add_block_param(entry, Type::ptr());
    Value* dst = b.add_block_param(entry, Type::ptr());
    Value* width = b.add_block_param(entry, Type::i32());
    Value* height = b.add_block_param(entry, Type::i32());
    Value* total = b.build_mul(width, height);

    if (target == MirExecTarget::CpuSimd) {
        kb.for_range(kb.const_i32(0), total, kb.const_i32(1), [&](Value* pix) {
            emit_rgb_to_gray_pixel(kb, src, dst, pix);
        });
    } else {
        Value* gid = emit_gpu_global_tid_x(b);
        kb.if_then(b.build_slt(gid, total), [&] {
            emit_rgb_to_gray_pixel(kb, src, dst, gid);
        });
    }
    b.build_ret_void();
    fn->rebuild_cfg_predecessors();
    return fn;
}

brass::Function* build_chw_normalize_kernel(brass::Module& mod, MirExecTarget target, const char* name) {
    using namespace brass;
    Function* fn = mod.create_function(
        name, Type::void_type(),
        {Type::ptr(), Type::ptr(), Type::i32(), Type::i32(),
         Type::f32(), Type::f32(), Type::f32(),
         Type::f32(), Type::f32(), Type::f32()}
    );
    codegen::KernelBuilder kb(mod, fn);
    Builder& b = kb.builder();
    BasicBlock* entry = b.append_block("entry");
    b.position_at_end(entry);

    Value* src = b.add_block_param(entry, Type::ptr());
    Value* dst = b.add_block_param(entry, Type::ptr());
    Value* width = b.add_block_param(entry, Type::i32());
    Value* height = b.add_block_param(entry, Type::i32());
    Value* means[3] = {
        b.add_block_param(entry, Type::f32()),
        b.add_block_param(entry, Type::f32()),
        b.add_block_param(entry, Type::f32()),
    };
    Value* inv_stds[3] = {
        b.add_block_param(entry, Type::f32()),
        b.add_block_param(entry, Type::f32()),
        b.add_block_param(entry, Type::f32()),
    };

    Value* plane = b.build_mul(width, height);
    Value* c0 = kb.const_i32(0);
    Value* c1 = kb.const_i32(1);

    if (target == MirExecTarget::CpuSimd) {
        // Emit one contiguous 1D loop per plane so CPU SIMD auto-vectorizes each channel cleanly.
        for (int c = 0; c < 3; ++c) {
            Value* plane_off = (c == 0) ? c0 : b.build_mul(kb.const_i32(c), plane);
            kb.for_range(c0, plane, c1, [&](Value* pix) {
                Value* idx = (c == 0) ? pix : b.build_add(plane_off, pix);
                emit_chw_normalize_elem(kb, src, dst, idx, means[c], inv_stds[c]);
            });
        }
    } else {
        Value* gid = emit_gpu_global_tid_x(b);
        kb.if_then(b.build_slt(gid, plane), [&] {
            for (int c = 0; c < 3; ++c) {
                Value* idx = (c == 0) ? gid : b.build_add(b.build_mul(kb.const_i32(c), plane), gid);
                emit_chw_normalize_elem(kb, src, dst, idx, means[c], inv_stds[c]);
            }
        });
    }
    b.build_ret_void();
    fn->rebuild_cfg_predecessors();
    return fn;
}

brass::Function* build_separable_blur_kernel(
    brass::Module& mod,
    MirExecTarget target,
    bool horizontal,
    const char* name
) {
    using namespace brass;
    Function* fn = mod.create_function(
        name, Type::void_type(),
        {Type::ptr(), Type::ptr(), Type::ptr(), Type::i32(), Type::i32(), Type::i32()}
    );
    codegen::KernelBuilder kb(mod, fn);
    Builder& b = kb.builder();
    BasicBlock* entry = b.append_block("entry");
    b.position_at_end(entry);

    Value* src = b.add_block_param(entry, Type::ptr());
    Value* dst = b.add_block_param(entry, Type::ptr());
    Value* weights = b.add_block_param(entry, Type::ptr());
    Value* width = b.add_block_param(entry, Type::i32());
    Value* height = b.add_block_param(entry, Type::i32());
    Value* radius = b.add_block_param(entry, Type::i32());

    Value* c0 = kb.const_i32(0);
    Value* c1 = kb.const_i32(1);

    if (target == MirExecTarget::CpuSimd) {
        kb.for_range(c0, height, c1, [&](Value* y) {
            kb.for_range(c0, width, c1, [&](Value* x) {
                emit_separable_blur_pixel(kb, horizontal, src, dst, weights, width, height, radius, x, y);
            });
        });
    } else {
        Value* total = b.build_mul(width, height);
        Value* gid = emit_gpu_global_tid_x(b);
        kb.if_then(b.build_slt(gid, total), [&] {
            Value* y = b.build_sdiv(gid, width);
            Value* x = b.build_sub(gid, b.build_mul(y, width));
            emit_separable_blur_pixel(kb, horizontal, src, dst, weights, width, height, radius, x, y);
        });
    }
    b.build_ret_void();
    fn->rebuild_cfg_predecessors();
    return fn;
}

brass::Function* build_bilinear_resize_kernel(brass::Module& mod, MirExecTarget target, const char* name) {
    using namespace brass;
    Function* fn = mod.create_function(
        name, Type::void_type(),
        {Type::ptr(), Type::ptr(), Type::i32(), Type::i32(), Type::i32(), Type::i32()}
    );
    codegen::KernelBuilder kb(mod, fn);
    Builder& b = kb.builder();
    BasicBlock* entry = b.append_block("entry");
    b.position_at_end(entry);

    Value* src = b.add_block_param(entry, Type::ptr());
    Value* dst = b.add_block_param(entry, Type::ptr());
    Value* src_w = b.add_block_param(entry, Type::i32());
    Value* src_h = b.add_block_param(entry, Type::i32());
    Value* dst_w = b.add_block_param(entry, Type::i32());
    Value* dst_h = b.add_block_param(entry, Type::i32());

    Value* c0 = kb.const_i32(0);
    Value* c1 = kb.const_i32(1);
    Value* xs = b.build_sdiv(b.build_sitofp_f32_i32(src_w), b.build_sitofp_f32_i32(dst_w));
    Value* ys = b.build_sdiv(b.build_sitofp_f32_i32(src_h), b.build_sitofp_f32_i32(dst_h));
    Value* sw_m1 = b.build_sub(src_w, c1);
    Value* sh_m1 = b.build_sub(src_h, c1);

    if (target == MirExecTarget::CpuSimd) {
        kb.for_range(c0, dst_h, c1, [&](Value* y) {
            kb.for_range(c0, dst_w, c1, [&](Value* x) {
                emit_bilinear_resize_pixel(kb, src, dst, src_w, src_h, dst_w, xs, ys, sw_m1, sh_m1, x, y);
            });
        });
    } else {
        Value* total = b.build_mul(dst_w, dst_h);
        Value* gid = emit_gpu_global_tid_x(b);
        kb.if_then(b.build_slt(gid, total), [&] {
            Value* y = b.build_sdiv(gid, dst_w);
            Value* x = b.build_sub(gid, b.build_mul(y, dst_w));
            emit_bilinear_resize_pixel(kb, src, dst, src_w, src_h, dst_w, xs, ys, sw_m1, sh_m1, x, y);
        });
    }
    b.build_ret_void();
    fn->rebuild_cfg_predecessors();
    return fn;
}

brass::codegen::KernelOptions make_cpu_simd_options() {
    brass::codegen::KernelOptions opts;
    opts.opt_level = 2;
    opts.enable_optimizations = true;
    opts.enable_sroa = true;
    opts.enable_bce = true;
    opts.enable_smt_guard_elim = true;
    opts.enable_superopt = true;
    opts.enable_avx2 = true;
    opts.enable_fma = true;
    opts.enable_fp_reassociation = true;
    opts.enable_vectorize = true;
    opts.enable_slp = true;
    opts.enable_unroll = true;
    opts.unroll_factor = 4;
    opts.enable_parallel = false;
    opts.enable_ftz_daz = true;
    return opts;
}

bool launch_vulkan_1d(
    const brass::target::SpirvKernel& kernel,
    uint32_t total_items,
    const std::vector<brass::gpu::VulkanArg>& args
) {
    if (total_items == 0) return true;
    std::string err;
    brass::gpu::VulkanModule mod = brass::gpu::VulkanModule::load(kernel, &err);
    if (!mod.valid()) return false;
    brass::gpu::VulkanDispatch dispatch;
    dispatch.block[0] = kGpuBlockSizeX;
    dispatch.block[1] = 1;
    dispatch.block[2] = 1;
    dispatch.grid[0] = (total_items + kGpuBlockSizeX - 1u) / kGpuBlockSizeX;
    dispatch.grid[1] = 1;
    dispatch.grid[2] = 1;
    return mod.launch(dispatch, args, &err);
}

} // namespace

brass::Function* build_mir_image_kernel(
    brass::Module& mod,
    MirKernelKind kind,
    MirExecTarget target,
    const char* fn_name
) {
    const char* name = (fn_name && fn_name[0] != '\0') ? fn_name : default_kernel_name(kind, target);
    switch (kind) {
        case MirKernelKind::AffineClampF32:
            return build_affine_clamp_kernel(mod, target, name);
        case MirKernelKind::RgbToGrayF32:
            return build_rgb_to_gray_kernel(mod, target, name);
        case MirKernelKind::ChwNormalizeF32:
            return build_chw_normalize_kernel(mod, target, name);
        case MirKernelKind::SeparableBlurHF32:
            return build_separable_blur_kernel(mod, target, true, name);
        case MirKernelKind::SeparableBlurVF32:
            return build_separable_blur_kernel(mod, target, false, name);
        case MirKernelKind::BilinearResizeF32:
            return build_bilinear_resize_kernel(mod, target, name);
    }
    return nullptr;
}

bool compile_mir_kernel_gpu_targets(
    const brass::Function& fn,
    CompiledGpuArtifacts& out,
    uint32_t local_size_x
) {
    out = {};
    brass::DiagnosticReporter diag;
    out.portability_ok = brass::verify_gpu_portability(fn, {}, &diag);
    out.portability_diagnostics = diag.format_all();
    if (!out.portability_ok) {
        return false;
    }

    brass::target::SpirvOptions spv_opts;
    spv_opts.stage = brass::target::SpirvStage::Compute;
    spv_opts.local_size_x = local_size_x;
    out.spirv = brass::target::SpirvTarget::compile(fn, spv_opts);

    out.ptx = brass::target::PtxTarget::emit_function(fn);

    brass::target::MslOptions msl_opts;
    msl_opts.workgroup_x = local_size_x;
    out.msl = brass::target::MslTarget::compile(fn, msl_opts);

    return !out.spirv.words.empty() && !out.ptx.empty() && !out.msl.source.empty();
}

MirImageKernelCompiler& MirImageKernelCompiler::instance() {
    static MirImageKernelCompiler s_instance;
    return s_instance;
}

bool MirImageKernelCompiler::is_cpu_supported() {
    return brass::Target::host().is_x64() || brass::Target::host().is_aarch64();
}

bool MirImageKernelCompiler::is_gpu_vulkan_available() {
    return brass::gpu::vulkan_available();
}

void MirImageKernelCompiler::clear_cache() {
    std::lock_guard<std::mutex> lock(mutex_);
    cpu_cache_.clear();
    spirv_cache_.clear();
}

size_t MirImageKernelCompiler::cpu_cache_size() {
    std::lock_guard<std::mutex> lock(mutex_);
    return cpu_cache_.size();
}

size_t MirImageKernelCompiler::gpu_cache_size() {
    std::lock_guard<std::mutex> lock(mutex_);
    return spirv_cache_.size();
}

template <typename FnPtr>
FnPtr MirImageKernelCompiler::get_or_compile_cpu(MirKernelKind kind) {
    if (!is_cpu_supported()) return nullptr;
    const uint8_t key = static_cast<uint8_t>(kind);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = cpu_cache_.find(key);
        if (it != cpu_cache_.end() && it->second.is_valid()) {
            return it->second.as<FnPtr>();
        }
    }

    const char* name = default_kernel_name(kind, MirExecTarget::CpuSimd);
    brass::Module mod(name);
    brass::Function* fn = build_mir_image_kernel(mod, kind, MirExecTarget::CpuSimd, name);
    if (!fn) return nullptr;

    brass::codegen::KernelJit jit(make_cpu_simd_options());
    brass::codegen::KernelFunction kfn = jit.compile(mod, name);
    if (!kfn.is_valid()) return nullptr;

    FnPtr ptr = kfn.as<FnPtr>();
    std::lock_guard<std::mutex> lock(mutex_);
    cpu_cache_[key] = std::move(kfn);
    return ptr;
}

const brass::target::SpirvKernel* MirImageKernelCompiler::get_or_compile_spirv(MirKernelKind kind) {
    const uint8_t key = static_cast<uint8_t>(kind);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = spirv_cache_.find(key);
        if (it != spirv_cache_.end()) {
            return &it->second;
        }
    }

    const char* name = default_kernel_name(kind, MirExecTarget::Gpu);
    brass::Module mod(name);
    brass::Function* fn = build_mir_image_kernel(mod, kind, MirExecTarget::Gpu, name);
    if (!fn) return nullptr;

    brass::DiagnosticReporter diag;
    if (!brass::verify_gpu_portability(*fn, {}, &diag)) {
        return nullptr;
    }

    brass::target::SpirvOptions spv_opts;
    spv_opts.stage = brass::target::SpirvStage::Compute;
    spv_opts.local_size_x = kGpuBlockSizeX;
    brass::target::SpirvKernel spv = brass::target::SpirvTarget::compile(*fn, spv_opts);
    if (spv.words.empty()) return nullptr;

    std::lock_guard<std::mutex> lock(mutex_);
    auto [it, inserted] = spirv_cache_.emplace(key, std::move(spv));
    (void)inserted;
    return &it->second;
}

bool MirImageKernelCompiler::execute_affine_clamp_f32_cpu(
    const float* src, float* dst, int width, int height, int channels,
    float alpha, float beta, float clamp_lo, float clamp_hi
) {
    if (!src || !dst || width <= 0 || height <= 0 || channels <= 0) return false;
    auto fn = get_or_compile_cpu<MirAffineClampF32Fn>(MirKernelKind::AffineClampF32);
    if (!fn) return false;
    const int32_t total = static_cast<int32_t>(width) * height * channels;
    fn(src, dst, total, alpha, beta, clamp_lo, clamp_hi);
    return true;
}

bool MirImageKernelCompiler::execute_rgb_to_gray_f32_cpu(
    const float* src_rgb, float* dst_gray, int width, int height
) {
    if (!src_rgb || !dst_gray || width <= 0 || height <= 0) return false;
    auto fn = get_or_compile_cpu<MirRgbToGrayF32Fn>(MirKernelKind::RgbToGrayF32);
    if (!fn) return false;
    fn(src_rgb, dst_gray, width, height);
    return true;
}

bool MirImageKernelCompiler::execute_chw_normalize_f32_cpu(
    const float* src_chw, float* dst_chw, int width, int height,
    const float mean[3], const float std_dev[3]
) {
    if (!src_chw || !dst_chw || width <= 0 || height <= 0) return false;
    const float def_m[3] = {0.0f, 0.0f, 0.0f};
    const float def_s[3] = {1.0f, 1.0f, 1.0f};
    const float* m = mean ? mean : def_m;
    const float* s = std_dev ? std_dev : def_s;
    const float inv_s[3] = {
        s[0] != 0.0f ? (1.0f / s[0]) : 0.0f,
        s[1] != 0.0f ? (1.0f / s[1]) : 0.0f,
        s[2] != 0.0f ? (1.0f / s[2]) : 0.0f,
    };
    auto fn = get_or_compile_cpu<MirChwNormalizeF32Fn>(MirKernelKind::ChwNormalizeF32);
    if (!fn) return false;
    fn(src_chw, dst_chw, width, height, m[0], m[1], m[2], inv_s[0], inv_s[1], inv_s[2]);
    return true;
}

bool MirImageKernelCompiler::execute_separable_blur_1d_f32_cpu(
    const float* src, float* dst, const float* weights,
    int width, int height, int radius, bool horizontal
) {
    if (!src || !dst || !weights || width <= 0 || height <= 0 || radius < 0) return false;
    const MirKernelKind kind = horizontal ? MirKernelKind::SeparableBlurHF32
                                          : MirKernelKind::SeparableBlurVF32;
    auto fn = get_or_compile_cpu<MirSeparableBlur1DF32Fn>(kind);
    if (!fn) return false;
    fn(src, dst, weights, width, height, radius);
    return true;
}

bool MirImageKernelCompiler::execute_separable_blur_f32_cpu(
    const float* src, float* dst, const float* weights,
    int width, int height, int radius
) {
    if (!src || !dst || !weights || width <= 0 || height <= 0 || radius < 0) return false;
    std::vector<float> tmp(static_cast<size_t>(width) * height);
    return execute_separable_blur_1d_f32_cpu(src, tmp.data(), weights, width, height, radius, true) &&
           execute_separable_blur_1d_f32_cpu(tmp.data(), dst, weights, width, height, radius, false);
}

bool MirImageKernelCompiler::execute_bilinear_resize_f32_cpu(
    const float* src, int src_w, int src_h,
    float* dst, int dst_w, int dst_h
) {
    if (!src || !dst || src_w <= 0 || src_h <= 0 || dst_w <= 0 || dst_h <= 0) return false;
    auto fn = get_or_compile_cpu<MirBilinearResizeF32Fn>(MirKernelKind::BilinearResizeF32);
    if (!fn) return false;
    fn(src, dst, src_w, src_h, dst_w, dst_h);
    return true;
}

bool MirImageKernelCompiler::execute_affine_clamp_f32_gpu(
    const float* src, float* dst, int width, int height, int channels,
    float alpha, float beta, float clamp_lo, float clamp_hi
) {
    if (!src || !dst || width <= 0 || height <= 0 || channels <= 0) return false;
    if (!is_gpu_vulkan_available()) return false;
    const auto* spv = get_or_compile_spirv(MirKernelKind::AffineClampF32);
    if (!spv) return false;

    const size_t total = static_cast<size_t>(width) * height * channels;
    const size_t bytes = total * sizeof(float);
    brass::gpu::VulkanBuffer d_src = brass::gpu::VulkanBuffer::alloc(bytes);
    brass::gpu::VulkanBuffer d_dst = brass::gpu::VulkanBuffer::alloc(bytes);
    if (!d_src.valid() || !d_dst.valid() || !d_src.upload(src, bytes)) return false;

    if (!launch_vulkan_1d(
            *spv, static_cast<uint32_t>(total),
            {d_src, d_dst, static_cast<int32_t>(total), alpha, beta, clamp_lo, clamp_hi})) {
        return false;
    }
    return d_dst.download(dst, bytes);
}

bool MirImageKernelCompiler::execute_rgb_to_gray_f32_gpu(
    const float* src_rgb, float* dst_gray, int width, int height
) {
    if (!src_rgb || !dst_gray || width <= 0 || height <= 0) return false;
    if (!is_gpu_vulkan_available()) return false;
    const auto* spv = get_or_compile_spirv(MirKernelKind::RgbToGrayF32);
    if (!spv) return false;

    const size_t total = static_cast<size_t>(width) * height;
    const size_t src_bytes = total * 3 * sizeof(float);
    const size_t dst_bytes = total * sizeof(float);
    brass::gpu::VulkanBuffer d_src = brass::gpu::VulkanBuffer::alloc(src_bytes);
    brass::gpu::VulkanBuffer d_dst = brass::gpu::VulkanBuffer::alloc(dst_bytes);
    if (!d_src.valid() || !d_dst.valid() || !d_src.upload(src_rgb, src_bytes)) return false;

    if (!launch_vulkan_1d(
            *spv, static_cast<uint32_t>(total),
            {d_src, d_dst, static_cast<int32_t>(width), static_cast<int32_t>(height)})) {
        return false;
    }
    return d_dst.download(dst_gray, dst_bytes);
}

bool MirImageKernelCompiler::execute_chw_normalize_f32_gpu(
    const float* src_chw, float* dst_chw, int width, int height,
    const float mean[3], const float std_dev[3]
) {
    if (!src_chw || !dst_chw || width <= 0 || height <= 0) return false;
    if (!is_gpu_vulkan_available()) return false;
    const auto* spv = get_or_compile_spirv(MirKernelKind::ChwNormalizeF32);
    if (!spv) return false;

    const float def_m[3] = {0.0f, 0.0f, 0.0f};
    const float def_s[3] = {1.0f, 1.0f, 1.0f};
    const float* m = mean ? mean : def_m;
    const float* s = std_dev ? std_dev : def_s;
    const float inv_s[3] = {
        s[0] != 0.0f ? (1.0f / s[0]) : 0.0f,
        s[1] != 0.0f ? (1.0f / s[1]) : 0.0f,
        s[2] != 0.0f ? (1.0f / s[2]) : 0.0f,
    };

    const size_t plane = static_cast<size_t>(width) * height;
    const size_t bytes = plane * 3 * sizeof(float);
    brass::gpu::VulkanBuffer d_src = brass::gpu::VulkanBuffer::alloc(bytes);
    brass::gpu::VulkanBuffer d_dst = brass::gpu::VulkanBuffer::alloc(bytes);
    if (!d_src.valid() || !d_dst.valid() || !d_src.upload(src_chw, bytes)) return false;

    if (!launch_vulkan_1d(
            *spv, static_cast<uint32_t>(plane),
            {d_src, d_dst, static_cast<int32_t>(width), static_cast<int32_t>(height),
             m[0], m[1], m[2], inv_s[0], inv_s[1], inv_s[2]})) {
        return false;
    }
    return d_dst.download(dst_chw, bytes);
}

bool MirImageKernelCompiler::execute_separable_blur_1d_f32_gpu(
    const float* src, float* dst, const float* weights,
    int width, int height, int radius, bool horizontal
) {
    if (!src || !dst || !weights || width <= 0 || height <= 0 || radius < 0) return false;
    if (!is_gpu_vulkan_available()) return false;
    const MirKernelKind kind = horizontal ? MirKernelKind::SeparableBlurHF32
                                          : MirKernelKind::SeparableBlurVF32;
    const auto* spv = get_or_compile_spirv(kind);
    if (!spv) return false;

    const size_t total = static_cast<size_t>(width) * height;
    const size_t img_bytes = total * sizeof(float);
    const size_t w_bytes = static_cast<size_t>(radius * 2 + 1) * sizeof(float);
    brass::gpu::VulkanBuffer d_src = brass::gpu::VulkanBuffer::alloc(img_bytes);
    brass::gpu::VulkanBuffer d_dst = brass::gpu::VulkanBuffer::alloc(img_bytes);
    brass::gpu::VulkanBuffer d_weights = brass::gpu::VulkanBuffer::alloc(w_bytes);
    if (!d_src.valid() || !d_dst.valid() || !d_weights.valid()) return false;
    if (!d_src.upload(src, img_bytes) || !d_weights.upload(weights, w_bytes)) return false;

    if (!launch_vulkan_1d(
            *spv, static_cast<uint32_t>(total),
            {d_src, d_dst, d_weights,
             static_cast<int32_t>(width), static_cast<int32_t>(height), static_cast<int32_t>(radius)})) {
        return false;
    }
    return d_dst.download(dst, img_bytes);
}

bool MirImageKernelCompiler::execute_separable_blur_f32_gpu(
    const float* src, float* dst, const float* weights,
    int width, int height, int radius
) {
    if (!src || !dst || !weights || width <= 0 || height <= 0 || radius < 0) return false;
    if (!is_gpu_vulkan_available()) return false;
    const auto* spv_h = get_or_compile_spirv(MirKernelKind::SeparableBlurHF32);
    const auto* spv_v = get_or_compile_spirv(MirKernelKind::SeparableBlurVF32);
    if (!spv_h || !spv_v) return false;

    const size_t total = static_cast<size_t>(width) * height;
    const size_t img_bytes = total * sizeof(float);
    const size_t w_bytes = static_cast<size_t>(radius * 2 + 1) * sizeof(float);
    brass::gpu::VulkanBuffer d_src = brass::gpu::VulkanBuffer::alloc(img_bytes);
    brass::gpu::VulkanBuffer d_tmp = brass::gpu::VulkanBuffer::alloc(img_bytes);
    brass::gpu::VulkanBuffer d_dst = brass::gpu::VulkanBuffer::alloc(img_bytes);
    brass::gpu::VulkanBuffer d_weights = brass::gpu::VulkanBuffer::alloc(w_bytes);
    if (!d_src.valid() || !d_tmp.valid() || !d_dst.valid() || !d_weights.valid()) return false;
    if (!d_src.upload(src, img_bytes) || !d_weights.upload(weights, w_bytes)) return false;

    if (!launch_vulkan_1d(
            *spv_h, static_cast<uint32_t>(total),
            {d_src, d_tmp, d_weights,
             static_cast<int32_t>(width), static_cast<int32_t>(height), static_cast<int32_t>(radius)})) {
        return false;
    }
    if (!launch_vulkan_1d(
            *spv_v, static_cast<uint32_t>(total),
            {d_tmp, d_dst, d_weights,
             static_cast<int32_t>(width), static_cast<int32_t>(height), static_cast<int32_t>(radius)})) {
        return false;
    }
    return d_dst.download(dst, img_bytes);
}

bool MirImageKernelCompiler::execute_bilinear_resize_f32_gpu(
    const float* src, int src_w, int src_h,
    float* dst, int dst_w, int dst_h
) {
    if (!src || !dst || src_w <= 0 || src_h <= 0 || dst_w <= 0 || dst_h <= 0) return false;
    if (!is_gpu_vulkan_available()) return false;
    const auto* spv = get_or_compile_spirv(MirKernelKind::BilinearResizeF32);
    if (!spv) return false;

    const size_t src_bytes = static_cast<size_t>(src_w) * src_h * sizeof(float);
    const size_t dst_total = static_cast<size_t>(dst_w) * dst_h;
    const size_t dst_bytes = dst_total * sizeof(float);
    brass::gpu::VulkanBuffer d_src = brass::gpu::VulkanBuffer::alloc(src_bytes);
    brass::gpu::VulkanBuffer d_dst = brass::gpu::VulkanBuffer::alloc(dst_bytes);
    if (!d_src.valid() || !d_dst.valid() || !d_src.upload(src, src_bytes)) return false;

    if (!launch_vulkan_1d(
            *spv, static_cast<uint32_t>(dst_total),
            {d_src, d_dst,
             static_cast<int32_t>(src_w), static_cast<int32_t>(src_h),
             static_cast<int32_t>(dst_w), static_cast<int32_t>(dst_h)})) {
        return false;
    }
    return d_dst.download(dst, dst_bytes);
}

} // namespace broimage::jit

namespace broimage {

bool mir_affine_clamp_f32(
    const float* src, float* dst, int width, int height, int channels,
    float alpha, float beta, float clamp_lo, float clamp_hi, bool prefer_gpu
) {
    auto& c = jit::MirImageKernelCompiler::instance();
    if (prefer_gpu && jit::MirImageKernelCompiler::is_gpu_vulkan_available()) {
        if (c.execute_affine_clamp_f32_gpu(src, dst, width, height, channels, alpha, beta, clamp_lo, clamp_hi)) {
            return true;
        }
    }
    return c.execute_affine_clamp_f32_cpu(src, dst, width, height, channels, alpha, beta, clamp_lo, clamp_hi);
}

bool mir_rgb_to_gray_f32(
    const float* src_rgb, float* dst_gray, int width, int height, bool prefer_gpu
) {
    auto& c = jit::MirImageKernelCompiler::instance();
    if (prefer_gpu && jit::MirImageKernelCompiler::is_gpu_vulkan_available()) {
        if (c.execute_rgb_to_gray_f32_gpu(src_rgb, dst_gray, width, height)) {
            return true;
        }
    }
    return c.execute_rgb_to_gray_f32_cpu(src_rgb, dst_gray, width, height);
}

bool mir_chw_normalize_f32(
    const float* src_chw, float* dst_chw, int width, int height,
    const float mean[3], const float std_dev[3], bool prefer_gpu
) {
    auto& c = jit::MirImageKernelCompiler::instance();
    if (prefer_gpu && jit::MirImageKernelCompiler::is_gpu_vulkan_available()) {
        if (c.execute_chw_normalize_f32_gpu(src_chw, dst_chw, width, height, mean, std_dev)) {
            return true;
        }
    }
    return c.execute_chw_normalize_f32_cpu(src_chw, dst_chw, width, height, mean, std_dev);
}

bool mir_separable_blur_f32(
    const float* src, float* dst, int width, int height,
    const float* weights, int radius, bool prefer_gpu
) {
    auto& c = jit::MirImageKernelCompiler::instance();
    if (prefer_gpu && jit::MirImageKernelCompiler::is_gpu_vulkan_available()) {
        if (c.execute_separable_blur_f32_gpu(src, dst, weights, width, height, radius)) {
            return true;
        }
    }
    return c.execute_separable_blur_f32_cpu(src, dst, weights, width, height, radius);
}

bool mir_bilinear_resize_f32(
    const float* src, int src_w, int src_h,
    float* dst, int dst_w, int dst_h, bool prefer_gpu
) {
    auto& c = jit::MirImageKernelCompiler::instance();
    if (prefer_gpu && jit::MirImageKernelCompiler::is_gpu_vulkan_available()) {
        if (c.execute_bilinear_resize_f32_gpu(src, src_w, src_h, dst, dst_w, dst_h)) {
            return true;
        }
    }
    return c.execute_bilinear_resize_f32_cpu(src, src_w, src_h, dst, dst_w, dst_h);
}

} // namespace broimage
