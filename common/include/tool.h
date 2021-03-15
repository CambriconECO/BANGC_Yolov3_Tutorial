/*************************************************************************
 * Copyright (C) [2018] by Cambricon, Inc.
 *************************************************************************/
#ifndef CNPLUGIN_TOOL_H_
#define CNPLUGIN_TOOL_H_
#include "cnml.h"
#include <string>
#include <sstream>
#include "stdlib.h"
#include <vector>
#include <fstream>

cnmlStatus_t cnmlDumpData2File(const char *filename, void *addr, unsigned long count);

void getFix8PositionScale(float *buffer, int size, int *position, float *scale);

cnmlStatus_t SetFixedPartition(cnmlBaseOp_t op, int n, int c, int h, int w);

typedef unsigned short half;

inline void cnrtConvertFloatToHalfArray(uint16_t* x, const float* y, int len) {
  for (int i = 0; i < len; i++) {
    cnrtConvertFloatToHalf(x + i, y[i]);
  }
}

inline void cnrtConvertHalfToFloatArray(float* x, const uint16_t* y, int len) {
  for (int i = 0; i < len; i++) {
    cnrtConvertHalfToFloat(x + i, y[i]);
  }
}

inline void cnrtConvertFloatToHalfArray(uint16_t* x, float* y, int len) {
  for (int i = 0; i < len; i++) {
    cnrtConvertFloatToHalf(x + i, y[i]);
  }
}

inline void cnrtConvertHalfToFloatArray(float* x, uint16_t* y, int len) {
  for (int i = 0; i < len; i++) {
    cnrtConvertHalfToFloat(x + i, y[i]);
  }
}

inline void cnrtMallocAndMemcpy(int* mlu_a, int* a, int len) {
  cnrtMalloc((void**)&mlu_a, len * sizeof(int));
  cnrtMemcpy(mlu_a, a, len * sizeof(int), CNRT_MEM_TRANS_DIR_HOST2DEV);
}

inline void cnrtMemcpyFloatToHalf(half* mlu_a, const float* a, const int len) {
  half* half_a = (half*)malloc(len * sizeof(half));
  cnrtConvertFloatToHalfArray(half_a, a, len);
  cnrtMemcpy((void*)mlu_a, (void*)half_a, len * sizeof(half),
             CNRT_MEM_TRANS_DIR_HOST2DEV);
  free(half_a);
}

inline void cnrtMemcpyHalfToFloat(float* a, const half* mlu_a, const int len) {
  half* half_a = (half*)malloc(len * sizeof(half));
  cnrtMemcpy((void*)half_a, (void*)mlu_a, len * sizeof(half),
             CNRT_MEM_TRANS_DIR_DEV2HOST);
  cnrtConvertHalfToFloatArray(a, half_a, len);
  free(half_a);
}


class CSVReader
{
  std::string fileName;
  std::string delimeter;

public:
  CSVReader(std::string filename, std::string delm = ",") : fileName(filename), delimeter(delm)
  {
  }

  // Function to fetch data from a CSV File
  std::vector<std::vector<float>> getData();
};

#ifdef WIN32
int rand_r(unsigned int *seed);
#endif

#endif  // CNPLUGIN_TOOL_H_
