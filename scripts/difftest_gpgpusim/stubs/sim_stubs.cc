// Link-time stand-ins for the parts of GPGPU-Sim that the PTX functional
// simulator references but that belong to the timing model (src/gpgpu-sim),
// the CUDA runtime front end (libcuda/cuda_runtime_api.cc), the stream
// manager or the statistics library.  Written for the PTX-formalization
// differential tester; nothing here is copied from NVIDIA software.
//
// Only functions on the functional path do real work:
//   * gpgpu_sim's constructor / getShaderCoreConfig / simd_model: a "GPU"
//     object that just carries the functional configuration;
//   * gpgpu_sim_config::reg_options: registers the functional options and
//     fixes the few shader-core parameters the functional simulator reads;
//   * increment_x_then_y_then_z: CTA-id iteration (same algorithm as
//     src/gpgpu-sim/gpu-sim.cc);
//   * LOGB2: floor(log2) (same as src/gpgpu-sim/gpu-misc.cc).
// Everything else is a no-op (statistics, timing-model hooks) or aborts if
// it is ever reached (ptxplus, SASS, interactive debugger).
#include <cstdio>
#include <cstdlib>

#include "../src/abstract_hardware_model.h"
#include "../src/cuda-sim/ptx_ir.h"
#include "../src/gpgpu-sim/gpu-cache.h"
#include "../src/gpgpu-sim/gpu-sim.h"
#include "../src/gpgpu-sim/shader.h"
#include "../src/statwrapper.h"
#include "../src/stream_manager.h"
#include "gpgpu_context.h"

[[noreturn]] static void unreachable(const char *what) {
  fprintf(stderr, "gpgpusim-ptx: unsupported path reached: %s\n", what);
  abort();
}

// ----------------------------------------------------------- GPU object
namespace {
class difftest_gpu : public gpgpu_sim {
 public:
  difftest_gpu(const gpgpu_sim_config &c, gpgpu_context *ctx)
      : gpgpu_sim(c, ctx) {}

 protected:
  void createSIMTCluster() override {}
};
}  // namespace

gpgpu_sim::gpgpu_sim(const gpgpu_sim_config &config, gpgpu_context *ctx)
    : gpgpu_t(config, ctx), m_config(config) {
  gpgpu_ctx = ctx;
  m_cluster = NULL;
  m_cuda_properties = NULL;
  m_shader_config = &m_config.m_shader_config;
  m_memory_config = &m_config.m_memory_config;
  ctx->ptx_parser->set_ptx_warp_size(m_shader_config);
  m_shader_stats = NULL;
  m_memory_stats = NULL;
  m_power_stats = NULL;
  m_gpgpusim_wrapper = NULL;
  gpu_sim_insn = gpu_tot_sim_insn = 0;
}

const shader_core_config *gpgpu_sim::getShaderCoreConfig() {
  return m_shader_config;
}
enum divergence_support_t gpgpu_sim::simd_model() const {
  return m_shader_config->model;
}
void gpgpu_sim::perf_memcpy_to_gpu(size_t, size_t) {}
void gpgpu_sim::hit_watchpoint(unsigned, ptx_thread_info *,
                               const ptx_instruction *) {}

void gpgpu_sim_config::reg_options(option_parser_t opp) {
  gpgpu_functional_sim_config::reg_options(opp);
  // The functional simulator only reads these shader-core parameters (to
  // size CTAs and warps); the values are those of the bundled SM7 configs.
  m_shader_config.warp_size = 32;
  m_shader_config.n_thread_per_shader = 2048;
  m_shader_config.gpgpu_shmem_size = 98304;
  m_shader_config.gpgpu_shader_registers = 65536;
  m_shader_config.max_cta_per_core = 32;
  m_shader_config.model = POST_DOMINATOR;
}

gpgpu_sim *difftest_make_gpu(gpgpu_sim_config *cfg, gpgpu_context *ctx) {
  gpgpu_sim *g = new difftest_gpu(*cfg, ctx);
  ctx->the_gpgpusim->g_the_gpu_config = cfg;
  ctx->the_gpgpusim->g_the_gpu = g;
  // Only register_finished_kernel() is reached, which ignores its object.
  ctx->the_gpgpusim->g_stream_manager =
      (stream_manager *)calloc(1, sizeof(stream_manager));
  ctx->the_gpgpusim->g_simulation_starttime = time(NULL);
  return g;
}

// ------------------------------------------------- small helper functions
unsigned int LOGB2(unsigned int v) {
  unsigned int r = 0;
  while (v >>= 1) r++;
  return r;
}

void increment_x_then_y_then_z(dim3 &i, const dim3 &bound) {
  i.x++;
  if (i.x >= bound.x) {
    i.x = 0;
    i.y++;
    if (i.y >= bound.y) {
      i.y = 0;
      if (i.z < bound.z) i.z++;
    }
  }
}

bool g_interactive_debugger_enabled = false;
int ptx_debug = 0;

// ------------------------------------------------- timing-model classes
cache_stats::cache_stats() {}
unsigned cache_config::set_index(new_addr_type) const {
  unreachable("cache_config::set_index");
}
unsigned l2_cache_config::set_index(new_addr_type) const {
  unreachable("l2_cache_config::set_index");
}
linear_to_raw_address_translation::linear_to_raw_address_translation() {}
void shader_core_config::set_pipeline_latency() {
  unreachable("set_pipeline_latency");
}

// ------------------------------------------------- statistics (no-ops)
class Stats *StatCreate(const char *, double, int) { return NULL; }
void StatAddSample(void *, int) {}
void StatDisp(void *) {}

// ------------------------------------------------- streams (single launch)
CUstream_st::CUstream_st() { m_pending = false; m_uid = 0; }
bool stream_manager::register_finished_kernel(unsigned) { return true; }
void stream_manager::add_stream(CUstream_st *) {
  unreachable("device-side stream creation");
}
void stream_manager::destroy_stream(CUstream_st *) {
  unreachable("device-side stream destruction");
}
void stream_manager::push(stream_operation) {
  unreachable("device-side kernel launch");
}

// ------------------------------------------------- CUDA runtime front end
void register_ptx_function(const char *, function_info *) {}
void ptxinfo_data::ptxinfo_addinfo() {}

// ------------------------------------------------- ptxplus (decuda) table
bool pred_lookup(int, int) { unreachable("ptxplus predicate lookup"); }
