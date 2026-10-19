#pragma once
#include "sail.h"
void sail_tok_str(sail_string *rop, const unit u);
void sail_tok_int(sail_int *rop, const unit u);
void sail_mem_rd(sail_int *rop, const sail_int sp, const sail_int a);
unit sail_mem_wr(const sail_int sp, const sail_int a, const sail_int v);
