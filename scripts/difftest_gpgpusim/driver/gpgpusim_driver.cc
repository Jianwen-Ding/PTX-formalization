// gpgpusim-ptx: load a PTX file into GPGPU-Sim's PTX front end and run
// kernels on its functional simulator (no timing model, no CUDA runtime).
//
//   gpgpusim-ptx JOBFILE OUTFILE
//
// JOBFILE is a line-oriented script written by run_compare.py:
//   ptx PATH                      parse this PTX module (once, first line)
//   alloc NAME NBYTES HEX|-       allocate a global buffer, initial bytes
//   kernel NAME                   select the entry for the next launch
//   grid X Y Z / block X Y Z      launch geometry
//   param OFFSET NBYTES HEX       a kernel parameter given as raw LE bytes
//   paramptr OFFSET NBYTES BUF ADDEND
//                                 a kernel parameter holding &BUF + ADDEND
//   launch                        run the kernel to completion (functional)
//   dump                          write "BUF NAME HEX" lines to OUTFILE
// GPGPU-Sim's own chatter goes to stdout; OUTFILE only holds "PARSED" (once
// the module is loaded), the dump and a final "DONE" line.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "../src/abstract_hardware_model.h"
#include "../src/cuda-sim/cuda-sim.h"
#include "../src/cuda-sim/ptx_ir.h"
#include "../src/gpgpu-sim/gpu-sim.h"
#include "../src/option_parser.h"
#include "gpgpu_context.h"

// Provided by sim_stubs.cc.
gpgpu_sim *difftest_make_gpu(gpgpu_sim_config *cfg, gpgpu_context *ctx);

static gpgpu_context *g_ctx;

gpgpu_context *GPGPU_Context() {
  if (!g_ctx) g_ctx = new gpgpu_context();
  return g_ctx;
}

namespace {

struct Buffer {
  std::string name;
  size_t addr;
  size_t size;
};

std::vector<unsigned char> unhex(const std::string &h) {
  std::vector<unsigned char> v;
  if (h == "-") return v;
  for (size_t i = 0; i + 1 < h.size(); i += 2)
    v.push_back((unsigned char)strtoul(h.substr(i, 2).c_str(), NULL, 16));
  return v;
}

[[noreturn]] void die(const std::string &m) {
  fprintf(stderr, "gpgpusim-ptx: %s\n", m.c_str());
  exit(3);
}

// Write the explicit initializers of module-scope .global and .const
// variables into simulated memory (both live in GPGPU-Sim's global memory
// space).  This is the job the CUDA runtime front end does when a module is
// registered; it is redone here because that front end is not built.
void load_initializers(symbol_table *symtab, gpgpu_t *gpu) {
  for (int pass = 0; pass < 2; pass++) {
    symbol_table::iterator b = pass ? symtab->const_iterator_begin()
                                    : symtab->global_iterator_begin();
    symbol_table::iterator e = pass ? symtab->const_iterator_end()
                                    : symtab->global_iterator_end();
    for (symbol_table::iterator g = b; g != e; ++g) {
      symbol *s = *g;
      if (!s->has_initializer()) continue;
      size_t bits;
      int basic;
      s->type()->get_key().type_decode(bits, basic);
      size_t elem = bits / 8, off = 0;
      std::list<operand_info> init = s->get_initializer();
      for (const operand_info &op : init) {
        ptx_reg_t v = op.get_literal_value();
        gpu->get_global_memory()->write(s->get_address() + off, elem, &v,
                                        NULL, NULL);
        off += elem;
      }
    }
  }
}

}  // namespace

int main(int argc, char **argv) {
  if (argc != 3) die("usage: gpgpusim-ptx JOBFILE OUTFILE");
  std::ifstream job(argv[1]);
  if (!job) die("cannot open job file");
  FILE *out = fopen(argv[2], "w");
  if (!out) die("cannot open output file");
  // Line-buffered, so that the last message before an abort() is kept.
  setvbuf(stdout, NULL, _IOLBF, 0);

  gpgpu_context *ctx = GPGPU_Context();
  ctx->func_sim->read_sim_environment_variables();
  ctx->ptx_parser->read_parser_environment_variables();
  ctx->func_sim->g_ptx_sim_mode = 1;

  // Register the options the PTX front end and functional simulator read, so
  // that they take their documented defaults.
  option_parser_t opp = option_parser_create();
  ctx->ptx_reg_options(opp);
  ctx->func_sim->ptx_opcocde_latency_options(opp);
  gpgpu_sim_config *cfg = new gpgpu_sim_config(ctx);
  cfg->reg_options(opp);
  const char *no_args[] = {"gpgpusim-ptx"};
  option_parser_cmdline(opp, 1, no_args);
  gpgpu_sim *gpu = difftest_make_gpu(cfg, ctx);

  symbol_table *symtab = NULL;
  std::map<std::string, Buffer> bufs;
  std::vector<std::string> buf_order;
  std::string kname;
  dim3 grid(1, 1, 1), block(1, 1, 1);
  struct PendingParam {
    size_t offset, nbytes;
    std::vector<unsigned char> data;
  };
  std::vector<PendingParam> params;

  std::string line;
  while (std::getline(job, line)) {
    std::istringstream ls(line);
    std::string cmd;
    if (!(ls >> cmd)) continue;
    if (cmd == "ptx") {
      std::string path;
      ls >> path;
      symtab = ctx->gpgpu_ptx_sim_load_ptx_from_filename(path.c_str());
      if (!symtab) die("PTX parse failed");
      load_initializers(symtab, gpu);
      fprintf(out, "PARSED\n");
      fflush(out);
    } else if (cmd == "alloc") {
      std::string name, hex;
      size_t n;
      ls >> name >> n >> hex;
      Buffer b;
      b.name = name;
      b.size = n;
      b.addr = (size_t)gpu->gpu_malloc(n ? n : 1);
      std::vector<unsigned char> init = unhex(hex);
      init.resize(n, 0);
      if (n) gpu->memcpy_to_gpu(b.addr, init.data(), n);
      bufs[name] = b;
      buf_order.push_back(name);
    } else if (cmd == "kernel") {
      ls >> kname;
    } else if (cmd == "grid") {
      ls >> grid.x >> grid.y >> grid.z;
    } else if (cmd == "block") {
      ls >> block.x >> block.y >> block.z;
    } else if (cmd == "param" || cmd == "paramptr") {
      PendingParam p;
      ls >> p.offset >> p.nbytes;
      if (cmd == "param") {
        std::string hex;
        ls >> hex;
        p.data = unhex(hex);
      } else {
        std::string bname;
        long long addend;
        ls >> bname >> addend;
        if (!bufs.count(bname)) die("unknown buffer " + bname);
        unsigned long long a = bufs[bname].addr + addend;
        p.data.resize(8);
        memcpy(p.data.data(), &a, 8);
      }
      // GPGPU-Sim copies parameters in 4-byte words: pad the source.
      p.data.resize(((p.nbytes + 3) / 4) * 4 + 8, 0);
      params.push_back(p);
    } else if (cmd == "launch") {
      if (!symtab) die("no PTX loaded");
      function_info *entry = symtab->lookup_function(kname);
      if (!entry) die("no kernel " + kname);
      if (entry->num_args() != params.size()) {
        fprintf(stderr, "warning: %s declares %u params, %zu given\n",
                kname.c_str(), entry->num_args(), params.size());
      }
      // No ptxas: resource usage is "unknown" (all zero), as GPGPU-Sim does
      // when no ptxinfo is available.
      gpgpu_ptx_sim_info info;
      memset(&info, 0, sizeof(info));
      entry->set_kernel_info(info);
      kernel_info_t *k = new kernel_info_t(grid, block, entry,
                                           gpu->getNameArrayMapping(),
                                           gpu->getNameInfoMapping());
      std::vector<gpgpu_ptx_sim_arg> args(params.size());
      for (size_t i = 0; i < params.size(); i++) {
        args[i] = gpgpu_ptx_sim_arg(params[i].data.data(), params[i].nbytes,
                                    params[i].offset);
        entry->add_param_data(i, &args[i]);
      }
      entry->finalize(k->get_param_memory());
      ctx->func_sim->g_ptx_kernel_count++;
      // As the stream manager does for a functional-mode launch.
      gpu->functional_launch(k);
      ctx->func_sim->gpgpu_cuda_ptx_sim_main_func(*k);
      gpu->finish_functional_sim(k);
      fflush(stdout);
      params.clear();
    } else if (cmd == "dump") {
      for (const std::string &n : buf_order) {
        const Buffer &b = bufs[n];
        std::vector<unsigned char> v(b.size);
        if (b.size) gpu->memcpy_from_gpu(v.data(), b.addr, b.size);
        fprintf(out, "BUF %s ", n.c_str());
        for (unsigned char c : v) fprintf(out, "%02x", c);
        if (v.empty()) fprintf(out, "-");
        fprintf(out, "\n");
      }
    } else {
      die("unknown job command " + cmd);
    }
  }
  fprintf(out, "DONE\n");
  fclose(out);
  return 0;
}
