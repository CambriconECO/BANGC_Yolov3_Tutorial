#ifndef _CNPLUGIN_MACRO_H_
#define _CNPLUGIN_MACRO_H_
#include "stdint.h"

#define MAX_PARA 2048
#define NUM_SERIALS(x, num_parallels) ((x + num_parallels - 1) / num_parallels) 
#define ALIGN(x) (NUM_SERIALS(x, 16) * 16)
#define ALIGN_SIZE 64
#define ALIGN_UP_TO(x, n) ((((x)-1) / (n) + 1) * (n))
typedef  uint16_t half;

#endif