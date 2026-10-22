// Stand-in for GPGPU-Sim src/cuda-sim/decuda_pred_table/decuda_pred_table.h.
// The decuda-derived predicate table is only used when running ptxplus
// (SASS-translated) code, which the differential tester never does; the
// function is defined in sim_stubs.cc and aborts if called.
bool pred_lookup(int condition, int flags);
