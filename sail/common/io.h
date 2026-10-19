#pragma once
#include "sail.h"
void sail_tok_str(sail_string *rop, const unit u);
void sail_tok_int(sail_int *rop, const unit u);
void sail_mem_rd(sail_int *rop, const sail_int sp, const sail_int a);
unit sail_mem_wr(const sail_int sp, const sail_int a, const sail_int v);
void sail_mem_rdn(sail_int *rop, const sail_int sp, const sail_int a, const sail_int n);
unit sail_mem_wrn(const sail_int sp, const sail_int a, const sail_int n, const sail_int v);
unit sail_mem_copy(const sail_int dsp, const sail_int da, const sail_int ssp, const sail_int sa, const sail_int n);
void sail_smap_get(sail_int *rop, const sail_int sp, const_sail_string k);
bool sail_smap_has(const sail_int sp, const_sail_string k);
unit sail_smap_set(const sail_int sp, const_sail_string k, const sail_int v);
unit sail_smap_del(const sail_int sp, const_sail_string k);
unit sail_smap_clear(const sail_int sp);
void sail_smap_size(sail_int *rop, const sail_int sp);
void sail_smap_key(sail_string *rop, const sail_int sp, const sail_int idx);
unit sail_smap_copy(const sail_int dst, const sail_int src);
