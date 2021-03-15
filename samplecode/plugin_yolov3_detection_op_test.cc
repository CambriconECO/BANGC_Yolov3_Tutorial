/*************************************************************************
 * Copyright (C) [2019] by Cambricon, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *************************************************************************/
#include "cnplugin.h"

int main() {
  // Yolov3DetectionOutputOp performs the computation described in the yolo-layer proposed by
  // Joseph Redmon and his team. The detailed algorithm can be found on the offical github weisite
  // of darknet. This op decodes, filtrates, and "NMNs" input feature maps and generates object
  // bounding bboxes in terms of normalized coordinates, class, and score.
  //
  // The following demo shows an example of creating and forwarding
  // PluginYolov3DetectionOutputOp. This demo consists of:
  // 1. Preparation of opParams and input data.
  // 2. Creation and computation of yolov3 op.
  // 3. Management of hardware resources.
  // 4. Expected results

  // Initiate MLU device & get device handle. This can be run any time before any other "cnrt API",
  // like cnrtMalloc.
  cnmlInit(0);
  unsigned dev_num;
  CNRT_CHECK(cnrtGetDeviceCount(&dev_num));
  if (dev_num == 0)
    return CNRT_RET_ERR_NODEV;
  cnrtDev_t dev;
  CNRT_CHECK(cnrtGetDeviceHandle(&dev, 0));
  CNRT_CHECK(cnrtSetCurrentDevice(dev));

  // Prepare params & data for yolov3 detection op
  const int num_batches = 8;
  const int num_classes = 80;
  const int num_anchors = 3;
  const int num_inputs = 3;
  const int num_outputs = 2;
  const int mask_size = num_anchors * num_inputs;
  int dp = 1;
  int num_boxes = 1024 * 2;
  int num_cores = 16;
  int float_mode = 1;
  std::vector<int> bias_shape = {1, 2 * mask_size, 1, 1};
  int c_arr_data[3] = {255, 255, 255};
  int im_w = 416;
  int im_h = 416;
  int w_arr_data[3] = {13, 26, 52};
  int h_arr_data[3] = {13, 26, 52};
  float bias_arr_data[] = {116, 90, 156, 198, 373, 326, 30, 61, 62,
                           45,  59, 119, 10,  13,  16,  30, 33, 23};
  float confidence_thresh = 0.5;
  float nms_thresh = 0.45;
  cnmlCoreVersion_t core_version = CNML_MLU270;

  // Set data_type & op mode for yolov3 detection op.
  // For float32 input, mlu mainly use float32-type inst and output must also be float32.
  // For float16 input, mlu mainly use float16-type inst and output must also be float16.
  cnmlDataType_t data_type;
  cnrtDataType_t cast_type;
  int data_width = 0;
  if (float_mode) {
    data_type = CNML_DATA_FLOAT32;
    cast_type = CNRT_FLOAT32;
    data_width = 4;
  } else {
    data_type = CNML_DATA_FLOAT16;
    cast_type = CNRT_FLOAT16;
    data_width = 2;
  }

  // Create params for yolov3 detection op.
  cnmlPluginYolov3DetectionOutputOpParam_t param;
  cnmlCreatePluginYolov3DetectionOutputOpParam(
      &param, num_batches, num_inputs, num_classes, num_anchors, num_boxes, im_w, im_h,
      confidence_thresh, nms_thresh, core_version, w_arr_data, h_arr_data, bias_arr_data);

  // Set Input/Output tensor shapes
  std::vector<int> output_shape(4, 1);
  output_shape[0] = num_batches;
  output_shape[1] = 7 * num_boxes + 64;

  std::vector<int> input0_shape(4, 1);
  std::vector<int> input1_shape(4, 1);
  std::vector<int> input2_shape(4, 1);
  input0_shape[0] = num_batches;
  input0_shape[1] = c_arr_data[0];
  input0_shape[2] = h_arr_data[0];
  input0_shape[3] = w_arr_data[0];

  input1_shape[0] = num_batches;
  input1_shape[1] = c_arr_data[1];
  input1_shape[2] = h_arr_data[1];
  input1_shape[3] = w_arr_data[1];

  input2_shape[0] = num_batches;
  input2_shape[1] = c_arr_data[2];
  input2_shape[2] = h_arr_data[2];
  input2_shape[3] = w_arr_data[2];

  // Set working space shape
  int buffer_size = (num_classes + 5) * num_anchors *
                    (h_arr_data[0] * w_arr_data[0] + h_arr_data[1] * w_arr_data[1] +
                     h_arr_data[2] * w_arr_data[2]);
  std::vector<int> buffer_shape = {num_batches, buffer_size, 1, 1};

  cnmlTensor_t *cnml_input_tensor = (cnmlTensor_t *)malloc(sizeof(cnmlTensor_t) * 3);
  cnmlTensor_t *cnml_output_tensor = (cnmlTensor_t *)malloc(sizeof(cnmlTensor_t) * 2);
  cnmlCreateTensor(&cnml_input_tensor[0], CNML_TENSOR, data_type, input0_shape[0], input0_shape[1],
                   input0_shape[2], input0_shape[3]);

  cnmlCreateTensor(&cnml_input_tensor[1], CNML_TENSOR, data_type, input1_shape[0], input1_shape[1],
                   input1_shape[2], input1_shape[3]);

  cnmlCreateTensor(&cnml_input_tensor[2], CNML_TENSOR, data_type, input2_shape[0], input2_shape[1],
                   input2_shape[2], input2_shape[3]);

  cnmlCreateTensor(&cnml_output_tensor[0], CNML_TENSOR, data_type, output_shape[0], output_shape[1],
                   output_shape[2], output_shape[3]);

  cnmlCreateTensor(&cnml_output_tensor[1], CNML_TENSOR, data_type, buffer_shape[0], buffer_shape[1],
                   buffer_shape[2], buffer_shape[3]);

  // Create op_ptr
  cnmlBaseOp_t op;
  cnmlCreatePluginYolov3DetectionOutputOp(&op, param, cnml_input_tensor, cnml_output_tensor);

  // Set op layout
  cnmlSetOperationComputingLayout(op, CNML_NHWC);

  // Compile op
  cnmlCompileBaseOp(op, CNML_MLU270, num_cores);

  // Load input data, there are 3 inputs.
  // Multi-batch data are duplicated from the original one batch data.
  // The data order in input files is NHWC.
  // MLU forward with NWHC data
  // CPU forward with NCHW data
  std::ifstream inFile0("scale1.txt");
  std::ifstream inFile1("scale2.txt");
  std::ifstream inFile2("scale3.txt");
  int inCount[num_inputs] = {1};
  inCount[0] = input0_shape[0] * input0_shape[1] * input0_shape[2] * input0_shape[3];
  inCount[1] = input1_shape[0] * input1_shape[1] * input1_shape[2] * input1_shape[3];
  inCount[2] = input2_shape[0] * input2_shape[1] * input2_shape[2] * input2_shape[3];
  std::cout << inCount[0] << std::endl;
  std::cout << inCount[1] << std::endl;
  std::cout << inCount[2] << std::endl;
  int outCount[num_outputs] = {1};
  outCount[0] = output_shape[0] * output_shape[1] * output_shape[2] * output_shape[3];
  outCount[1] = buffer_shape[0] * buffer_shape[1] * buffer_shape[2] * buffer_shape[3];

  // Malloc cpu space for input data. Here trams_inputX ptrs are used by cpuForward function.
  float *input0 = (float *)malloc(inCount[0] * sizeof(float));
  float *input1 = (float *)malloc(inCount[1] * sizeof(float));
  float *input2 = (float *)malloc(inCount[2] * sizeof(float));
  float *trans_input0 = (float *)malloc(inCount[0] * sizeof(float));
  float *trans_input1 = (float *)malloc(inCount[1] * sizeof(float));
  float *trans_input2 = (float *)malloc(inCount[2] * sizeof(float));
  float *predicts_cpu = (float *)malloc(outCount[0] * sizeof(float));
  float *predicts_mlu = (float *)malloc(outCount[0] * sizeof(float));
  float *cast_input0 = (float *)malloc(inCount[0] * data_width);
  float *cast_input1 = (float *)malloc(inCount[1] * data_width);
  float *cast_input2 = (float *)malloc(inCount[2] * data_width);
  float *cast_predicts_mlu = (float *)malloc(outCount[0] * data_width);

  // Load inputs and trans_inputs
  double data = 0.0;
  int count = 0;
  while (!inFile0.eof() && count < inCount[0] / num_batches) {
    inFile0 >> data;
    for (int batchIdx = 0; batchIdx < num_batches; batchIdx++) {
      int offset = batchIdx * input0_shape[1] * input0_shape[2] * input0_shape[3] + count;
      input0[offset] = (float)data;
    }
    count++;
  }
  inFile0.close();
  for (int batchIdx = 0; batchIdx < input0_shape[0]; batchIdx++) {
    for (int h = 0; h < input0_shape[2] * input0_shape[3]; h++) {
      for (int w = 0; w < input0_shape[1]; w++) {
        int trans_offset = w * input0_shape[2] * input0_shape[3] + h +
                           batchIdx * input0_shape[2] * input0_shape[3] * input0_shape[1];
        int orig_offset = h * input0_shape[1] + w +
                          batchIdx * input0_shape[2] * input0_shape[3] * input0_shape[1];
        trans_input0[trans_offset] = input0[orig_offset];
      }
    }
  }
  count = 0;
  while (!inFile1.eof() && count < inCount[1] / num_batches) {
    inFile1 >> data;
    for (int batchIdx = 0; batchIdx < num_batches; batchIdx++) {
      input1[batchIdx * input1_shape[1] * input1_shape[2] * input1_shape[3] + count] = (float)data;
    }
    count++;
  }
  inFile1.close();
  for (int batchIdx = 0; batchIdx < input1_shape[0]; batchIdx++) {
    for (int h = 0; h < input1_shape[2] * input1_shape[3]; h++) {
      for (int w = 0; w < input1_shape[1]; w++) {
        int trans_offset = w * input1_shape[2] * input1_shape[3] + h +
                           batchIdx * input1_shape[2] * input1_shape[3] * input1_shape[1];
        int orig_offset = h * input1_shape[1] + w +
                          batchIdx * input1_shape[2] * input1_shape[3] * input1_shape[1];
        trans_input1[trans_offset] = input1[orig_offset];
      }
    }
  }
  count = 0;
  while (!inFile2.eof() && count < inCount[2] / num_batches) {
    inFile2 >> data;
    for (int batchIdx = 0; batchIdx < num_batches; batchIdx++) {
      input2[batchIdx * input2_shape[1] * input2_shape[2] * input2_shape[3] + count] = (float)data;
    }
    count++;
  }
  inFile2.close();
  for (int batchIdx = 0; batchIdx < input2_shape[0]; batchIdx++) {
    for (int h = 0; h < input2_shape[2] * input2_shape[3]; h++) {
      for (int w = 0; w < input2_shape[1]; w++) {
        int trans_offset = w * input2_shape[2] * input2_shape[3] + h +
                           batchIdx * input2_shape[2] * input2_shape[3] * input2_shape[1];
        int orig_offset = h * input2_shape[1] + w +
                          batchIdx * input2_shape[2] * input2_shape[3] * input2_shape[1];
        trans_input2[trans_offset] = input2[orig_offset];
      }
    }
  }

  // Cast data for proper datatype
  CNRT_CHECK(cnrtCastDataType(input0, CNRT_FLOAT32, cast_input0, cast_type, inCount[0], nullptr));
  CNRT_CHECK(cnrtCastDataType(input1, CNRT_FLOAT32, cast_input1, cast_type, inCount[1], nullptr));
  CNRT_CHECK(cnrtCastDataType(input2, CNRT_FLOAT32, cast_input2, cast_type, inCount[2], nullptr));

  // Malloc and memcpy from host to device
  void **cpu_addrs = (void **)malloc(sizeof(void *) * num_inputs);
  void **cpu_org_addrs = (void **)malloc(sizeof(void *) * num_inputs);
  void **input_addrs = (void **)malloc(sizeof(void *) * num_inputs);
  void **output_addrs = (void **)malloc(sizeof(void *) * num_outputs);
  cpu_org_addrs[0] = (void *)trans_input0;
  cpu_org_addrs[1] = (void *)trans_input1;
  cpu_org_addrs[2] = (void *)trans_input2;
  cpu_addrs[0] = (void *)cast_input0;
  cpu_addrs[1] = (void *)cast_input1;
  cpu_addrs[2] = (void *)cast_input2;

  for (int i = 0; i < num_inputs; i++) {
    CNRT_CHECK(cnrtMalloc(&(input_addrs[i]), inCount[i] * data_width));
    CNRT_CHECK(cnrtMemcpy(input_addrs[i], cpu_addrs[i], inCount[i] * data_width,
                          CNRT_MEM_TRANS_DIR_HOST2DEV));
  }
  for (int i = 0; i < num_outputs; i++) {
    CNRT_CHECK(cnrtMalloc(&(output_addrs[i]), outCount[i] * data_width));
  }

  // Forward cpu
  cnmlCpuComputePluginYolov3DetectionOutputOpForward(param, cpu_org_addrs, (void *)predicts_cpu);

  // Forward mlu
  cnrtQueue_t queue;
  CNRT_CHECK(cnrtCreateQueue(&queue));
  cnrtInvokeFuncParam_t compute_forw_param;
  u32_t affinity = 0x01;
  compute_forw_param.data_parallelism = &dp;
  compute_forw_param.affinity = &affinity;
  compute_forw_param.end = CNRT_PARAM_END;

  cnmlComputePluginYolov3DetectionOutputOpForward(op, input_addrs, num_inputs, output_addrs,
                                                  num_outputs, &compute_forw_param, queue);
  CNRT_CHECK(cnrtSyncQueue(queue));
  CNRT_CHECK(cnrtDestroyQueue(queue));

  // Memcpy from device to host
  CNRT_CHECK(cnrtMemcpy(cast_predicts_mlu, output_addrs[0], outCount[0] * data_width,
                        CNRT_MEM_TRANS_DIR_DEV2HOST));
  CNRT_CHECK(cnrtCastDataType(cast_predicts_mlu, cast_type, predicts_mlu, CNRT_FLOAT32, outCount[0],
                              nullptr));

  std::cout << std::endl;
  std::cout << "==================== MLU TEST ====================" << std::endl;

  // Check result. The expected result is 13 boxes for each batch.
  int result_boxes = 0;
  int result_status = 0;
  for (int batchIdx = 0; batchIdx < num_batches; batchIdx++) {
    int batchSize = batchIdx * (64 + 7 * num_boxes);
    result_boxes = (int)predicts_mlu[batchSize];
    for (int i = 0; i < result_boxes; i++) {
      for (int j = 0; j < 7; j++) {
        std::cout << (float)predicts_mlu[i * 7 + j + 64 + batchSize] << " ";
      }
      std::cout << std::endl;
      std::cout << std::endl;
    }
    std::cout << "========= Num of valid box from mlu for batch: " << batchIdx << " is "
              << result_boxes << " =========" << std::endl;

    if (result_boxes != (int)predicts_cpu[batchSize]) {
      result_status = -1;
    }
  }

  if (result_status < 0) {
    printf("FAILED!\n");
  } else {
    printf("PASSED!\n");
  }

  // Free resources
  for (int i = 0; i < num_inputs; i++) {
    cnmlDestroyTensor(&cnml_input_tensor[i]);
    CNRT_CHECK(cnrtFree(input_addrs[i]));
  }
  for (int i = 0; i < num_outputs; i++) {
    cnmlDestroyTensor(&cnml_output_tensor[i]);
    CNRT_CHECK(cnrtFree(output_addrs[i]));
  }
  cnmlDestroyPluginYolov3DetectionOutputOpParam(&param);
  cnmlDestroyBaseOp(&op);

  free(cpu_addrs);
  free(cpu_org_addrs);
  free(input_addrs);
  free(output_addrs);
  free(input0);
  free(input1);
  free(input2);
  free(trans_input0);
  free(trans_input1);
  free(trans_input2);
  free(predicts_mlu);
  free(predicts_cpu);
  free(cast_input0);
  free(cast_input1);
  free(cast_input2);
  free(cast_predicts_mlu);
  free(cnml_input_tensor);
  free(cnml_output_tensor);
  cnmlExit();
  return result_status;
}
