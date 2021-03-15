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
#include "cnml.h"
#include "cnrt.h"
#include "stdlib.h"
#include <stdio.h>
#include <sys/time.h>
#include <string.h>
#include <assert.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <memory.h>
#include <algorithm>
#include <cmath>

#ifndef CNPLUGIN_H_
#define CNPLUGIN_H_

#define CNPLUGIN_MAJOR_VERSION 1
#define CNPLUGIN_MINOR_VERSION 6
#define CNPLUGIN_PATCH_VERSION 0

using std::vector;
typedef uint16_t half;

#define CNPLUGIN_VERSION (CNPLUGIN_MAJOR_VERSION * 10000 + CNPLUGIN_MINOR_VERSION * 100 + CNPLUGIN_PATCH_VERSION)

/* ====================== */
/* enum definitions start */
/* ====================== */
/*!
 *  @enum cnmlPluginSsdCodeType_t
 *  @breif An enum.
 *
 *  ``cnmlPluginSsdCodeType_t`` is an enum holding the description of CodeType
 *  used in PluginSsdDetectionOutoutOp, including:
 *
 *    CodeType_CORNER:      (x1, y1) + (x2, y2)
 *
 *    CodeType_CENTER_SIZE: (xc, yc) + (w , h )
 *
 *    CodeType_CORNER_SIZE: (x1, y1) + (w , h )
 *
 *    where (x1, y1) represents the top-left corner,
 *          (x2, y2) represents the bottom-right corner, and
 *          (w , h ) represents the (w)idth and (h)eight.
 */
typedef enum {
  CodeType_CORNER = 0,
  CodeType_CENTER_SIZE = 1,
  CodeType_CORNER_SIZE = 2,
} cnmlPluginSsdCodeType_t;

/*!
 *  @enum cnmlPluginColorCvt_t
 *  @brief An enum.
 *
 *  ``cnmlPluginColorCvt_t`` is an num holding the description of color
 *  conversion mode used in ``ResizeAndColorCvt`` kind of operations, including:
 *  Resize, ResizeYuvToRgba, CropAndResize, and YuvToRgba. More will come.
 */
typedef enum {
  RGBA_TO_RGBA = 0,
  YUV_TO_RGBA_NV12 = 1,
  YUV_TO_RGBA_NV21 = 2,
  YUV_TO_BGRA_NV12 = 3,
  YUV_TO_BGRA_NV21 = 4,
  YUV_TO_ARGB_NV12 = 5,
  YUV_TO_ARGB_NV21 = 6,
  YUV_TO_ABGR_NV12 = 7,
  YUV_TO_ABGR_NV21 = 8,
  YUV_TO_RGB_NV12 = 9,
  YUV_TO_RGB_NV21 = 10,
  YUV_TO_BGR_NV12 = 11,
  YUV_TO_BGR_NV21 = 12,
  GRAY_TO_GRAY = 13
} cnmlPluginColorCvt_t;

/*!
 *  @enum cnmlPluginDataType_t
 *  @brief An enum.
 *
 *  ``cnmlPluginDataType_t`` is an num holding the description of datatype
 *  conversion mode used in ``ResizeAndColorCvt`` kind of operations, including:
 *  Resize, ResizeYuvToRgba, CropAndResize, and YuvToRgba. More will come.
 */
typedef enum {
  FP16_TO_FP16 = 0,
  FP16_TO_UINT8 = 1,
  UINT8_TO_FP16 = 2,
  UINT8_TO_UINT8 = 3
} cnmlPluginDataType_t;

/*!
 *  @enum cnmlBoxCodeType_t
 *  @brief An enum.
 *
 *  ``cnmlBoxCodeType_t`` is an enum holding the description of box code type
 *  used in ``PluginBoxCoder`` operation.
 */
typedef enum {
    Encode = 0,
    Decode = 1
} cnmlBoxCodeType_t;
/* -------------------- */
/* enum definitions end */
/* -------------------- */

/* ======================== */
/* struct definitions start */
/* ======================== */
/*!
 *  @struct roiParams
 *  @brief A struct.
 *
 *  ``roiParams`` is a struct holding the description of bounding box info.
 *  CORNER_SIZE mode is used here, which means all bounding boxes are discribed in
 *  terms of (x1, y1) + (w, h).
 */
typedef struct roiParams {
  int roi_x;
  int roi_y;
  int roi_w;
  int roi_h;
} roiParams;

/*!
 *  @struct ioParams
 *  @brief A struct
 *
 *  ``ioParams`` is a struct holding the descroption of color and datatype
 *  conversion mode used in ``ResizeAndColorCvt`` kind of operations, including:
 *  Resize, ResizeYuvToRgba, CropAndResize, and YuvToRgba. More will come.
 */
typedef struct ioParams {
  cnmlPluginColorCvt_t color;
  cnmlPluginDataType_t datatype;
} ioParams;

/*!
 *  @struct cnmlPluginResizeAndColorCvtParam
 *  @brief A struct
 *
 *  ``cnmlPluginResizeAndColorCvtParam`` is a struct holding the parameters used
 *  in ``ResizeAndColotCvt`` kind of operations. In this struct, users only need
 *  to provide "user params". Others will be parsed through the ioParams chosen
 *  by users.
 */
struct cnmlPluginResizeAndColorCvtParam {
  int s_row;  //!< Height(or number of row) of src image.
  int s_col;  //!< Width(or number of column) of src image.
  int d_row;  //!< Height(or number of row) of dst image.
  int d_col;  //!< Width(or number of column) of src image.
  int roi_x;  //!< X-coord of centor of Region of Interest(roi).
  int roi_y;  //!< Y-coord of centor of Region of Interest(roi).
  int roi_h;  //!< Height of Region of Interest(roi).
  int roi_w;  //!< Width of Region of Interest(roi).
  int batchNum;  //!< Batch number of inputs/outputs.
  int keepAspectRatio;  //!< A flag indicates if resize op keeps aspect ratio, 0: No, 1: Yes.
  int padMethod;  //!< A flag indicates the pad method, 0: padding is on both sides, 1: padding is on the right or bottom side
  ioParams mode;  //!< Operation mode. See ioParams for details.
  cnmlCoreVersion_t core_version;  //!< Hardware version.
  bool input_series;  //!< A flag indicates if y and yv input address is continuation, true: Yes, false: No

  int inputType;  //!< Input color space type.
  int outputType;  //!< Ouptut color space type.
  int channelIn;  //!< Number of input channel, used in image resize/convert.
  int channelOut;  //!< Number of outpyut channel, used in image resize/convert.
  int layerIn;  //!< Number of input layer. Some inputs may have un-uniform data layouts, like YUV420SP.
  int layerOut;  //!< Number of output layer. Some outputs may have un-uniform data layouts, like YUV420SP.
  int reverseChannel;  //!< A flag indicates if the order of input channel is reversed, 0: NV12, 1: NV21.
  int input2half;  //!< A flag indicates if input data should be converted to FLOAT16, 0: No, 1: Yes.
  int output2uint;  //!< A flag indicates if input data should be converted to UINT8, 0: No, 1: Yes.
  int input_num;  //!< Number of input tensors.
  int output_num;  //!< Number of output tensors.
  int static_num;  //!< Number of const-value tensors.
  cnmlDataType_t inputDT_MLU;  //!< Input datatype on mlu in terms of cnmlDataType_t.
  cnmlDataType_t inputDT_CPU;  //!< Input datatype on cpu in terms of cnmlDataType_t.
  cnmlDataType_t outputDT_MLU;  //!< Output datatype on mlu in terms of cnmlDataType_t.
  cnmlDataType_t outputDT_CPU;  //!< Output datatype on cpu in terms of cnmlDataType_t.

  int depth;  //!< Number of input channel, used in feature map resize/convert.
  int box_number; //!< Number of roi.
  int pad_size;  //!< Value used to pad up depth. Default is 1.
  float extrapolation_value;  //!< Value used for extrapolation_value, when applicable.

  cnmlTensor_t *cnml_input_ptr;  //!< Input cnmlTensors.
  cnmlTensor_t *cnml_output_ptr;  //!< Ouptut cnmlTensors.
  cnmlTensor_t *cnml_static_ptr;  //!< Const-valued cnmlTensors.
  void **static_data_ptr;  //!< Const-valued data if need.
};

/*! ``cnmlPluginResizeAndColorCvtParam_t`` is a pointer to a
    structure (cnmlPluginResizeAndColorCvtParam) holding the description of CV operations param.
*/
typedef cnmlPluginResizeAndColorCvtParam *cnmlPluginResizeAndColorCvtParam_t;
/* ---------------------- */
/* struct definitions end */
/* ---------------------- */

/* =============================================== */
/* cnmlPluginYolov3DetectionOutout operation start */
/* =============================================== */
/*!
 *  @struct cnmlPluginYolov3DetectionOutputOpParam
 *  @brief A struct.
 *
 *  cnmlPluginYolov3DetectionOutputOpParam is a structure describing the "param"
 *  parameter of Yolov3DetectionOutput operation.
 *  cnmlCreatePluginYolov3DetectionOutputOpParam() is used to create
 *  an instance of cnmlPluginYolov3DetectionOutputOpParam_t.
 *  cnmlDestroyPluginYolov3DetectionOutputOpParam() is used to destroy
 *  an instance of cnmlPluginYolov3DetectionOutputOpParam_t.
 *  Detailed information of parameters, see cnmlCreatePluginYolov3DetectionOutputOpParam().
 */
struct cnmlPluginYolov3DetectionOutputOpParam;

/*! ``cnmlPluginYolov3DetectionOutputOpParam_t`` is a pointer to a
    structure (cnmlPluginYolov3DetectionOutputOpParam) holding the description of a Yolov3DetectionOutput operation param.
*/
typedef cnmlPluginYolov3DetectionOutputOpParam
*cnmlPluginYolov3DetectionOutputOpParam_t;

/*!
 *  @brief A function.
 *
 *  This function creates a PluginYolov3DetectionOutputOp param object with
 *  the pointer and parameters provided by user.
 *
 *  **Supports MLU220 and MLU270.**
 *
 *  @param[out] param
 *    Output. The returning param descriptor.
 *  @param[in] batchNum
 *    Input. The number of input batches.
 *           No default value, a valid batchNum must be in the range of [1, inf).
 *  @param[in] inputNum
 *    Input. The number of input tensors.
 *           No default value, a valid inputNum must be in the range of [1, 7].
 *  @param[in] classNum
 *    Input. The number of input classes.
 *           No default value, a valid classNum must be in the range of [1, 4096].
 *  @param[in] maskGroupNum
 *    Input. The number of anchors used by every input tensors.
 *           No default value, a valid maskGroupNum must be in the range of [1, inf].
 *  @param[in] maxBoxNum
 *    Input. The largest possible number of output boxes.
 *           Default value is 1024, a valid maxBoxNum must be in the range of [1, inf].
 *  @param[in] netw
 *    Input. Width of input image of backbone network.
 *           No default value, a valid netw must be in the range of [1, inf).
 *  @param[in] neth
 *    Input. Height of input image of backbone network.
 *           No default value, a valid neth must be in the range of [1, inf).
 *  @param[in] confidence_thresh
 *    Input. Confidence threshold.
 *           No default value, a valid confidence_thresh must be in the range of [0, 1].
 *  @param[in] nms_thresh.
 *    Input. IOU threshold used in NMS function.
 *           No default value, a valid nms_thresh must be in the range of [0, 1].
 *  @param[in] core_version
 *    Input. Supported core version.
 *           No default value, a valid core_version must be either MLU220 or MLU270.
 *  @param[in] inputWs
 *    Input. Width of every input tensor. Must have the same order as inputHs
 *           No default value, the number of valid elements must be equal with inputNum.
 *  @param[in] inputHs
 *    Input. Height of every input tensor. Must have the same order as inputWs
 *           No default value, the number of valid elements must be equal with inputNum.
 *  @param[in] biases
 *    Input. Anchors of every input tensor.
 *           No default value. The number of valid elements must be equal with 2 x inputNum x maskGroupNum.
 *           The order of data from high to low, is [N(1) H(inputNum) W(maskGroupNum) C(2)]. For example:
 *           - Width of anchor for mask0 input0, Height of anchor for mask0 input0,
 *           - Width of anchor for mask1 input0, Height of anchor for mask1 input0,
 *           - ......
 *           - Width of anchor for maskN input0, Height of anchor for maskN input0,
 *           - Width of anchor for mask0 input1, Height of anchor for mask0 input1,
 *           - ......
 *
 *  @retval CNML_STATUS_SUCCESS
 *    The object was set successfully.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    The inputH/Ws ptr is nullptr or input param is invalid.
 */
cnmlStatus_t cnmlCreatePluginYolov3DetectionOutputOpParam(
    cnmlPluginYolov3DetectionOutputOpParam_t *param,
    int batchNum,
    int inputNum,
    int classNum,
    int maskGroupNum,
    int maxBoxNum,
    int netw,
    int neth,
    float confidence_thresh,
    float nms_thresh,
    cnmlCoreVersion_t core_version,
    int *inputWs,
    int *inputHs,
    float *biases);

/*!
 *  @brief A function.
 *
 *  This function frees the PluginYolov3DetectionOutputOpParam struct, pointed
 *  by the pointer provided by user.
 *
 *  **Supports MLU220 and MLU270.**
 *
 *  @param[in]  param
 *    Input. A pointer to the address of the struct of computation parameters
 *    for PluginYolov3DetectionOutput operator.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Param is a null pointer.
 *    - The pointer content pointed by param is already freed.
 */
cnmlStatus_t cnmlDestroyPluginYolov3DetectionOutputOpParam(
    cnmlPluginYolov3DetectionOutputOpParam_t *param);

/*!
 *  @brief A function.
 *
 *  This function creates PluginYolov3DetectionOutputOp with proper param,
 *  input, and output tensors.
 *
 *  PluginYolov3DetectionOutputOp takes in feature maps and network
 *  parameters and computes valid bounding boxes based on two thresholds
 *  you have chosen.
 *
 *  **Reference:**
 *
 *    This implementation is based on the project on ``github/pjreddie/darknet`` .
 *
 *  **Formula:**
 *
 *    This op contains two steps:
 *
 *    1. DecodeAllBBoxes.
 *
 *       Convert input feature maps into real ojectness score and coordinates.
 *
 *       for inputIdx in (0, inputNum - 1)
 *
 *       obj = sigmoid(obj_feature);
 *
 *       x   = (x_offset + sigmoid(x_feature)) / inputWs[inputIdx]
 *
 *       y   = (y_offset + sigmoid(y_feature)) / inputHs[inputIdx]
 *
 *       w   = (w_biases * exp(w_feature)) / netw
 *
 *       h   = (h_biases * exp(h_feature)) / neth
 *
 *       where obj, x_feature, y_feature, w_feature, h_feature are data from input feature maps.
 *
 *       x_offset, y_offset are the coordinates of the grid cell in the feature map.
 *
 *       w_offset, h_biases are the shape of the anchor box.
 *
 *    2. Non-maximum Suppression.
 *
 *       For each class of data, compute IOU score for every pair of bounding boxes.
 *
 *       If IOU score exceeds the IOU threshold, keep the box with larger score.
 *
 *       x1 = x - w / 2
 *
 *       y1 = y - y / 2
 *
 *       x2 = x + w / 2
 *
 *       y2 = y + y / 2
 *
 *       for classIdx in (0, classNum - 1)
 *
 *        conf = obj * probability[classIdx]
 *
 *        max, maxIdx = findMaxValueAndIndex(conf)
 *
 *        if (max >= confidence_thresh)
 *
 *          for boxIdx in (0, boxNum - 1)
 *
 *            iou = computeIOU(coord_maxIdx, coord_boxIdx)  // where "coords" means x1,y1,x2,y2
 *
 *            if (iou < nms_thresh)
 *
 *              keep coords and conf for boxIdx
 *
 *  **DataType:**
 *
 *    Support only half(float16) type for both input and output tensors.
 *
 *  **Performance Optimization:**
 *
 *    The performance of detection layer depends on both the data size and the value.
 *    However, this op achieves relatively better performance when
 *    all of the following conditions are met:
 *
 *    - inputH/Ws are 64-aligned(unit in number of data).
 *
 *    - (5 + classNum) is 64-aligned(unit in number of data).
 *
 *    The bigger the remainder of the value of param divided by 64, the better performance the op will achieve.
 *
 *  Supports both MLU220 and MLU270.
 *
 *  @param[out]  op
 *    Output. A pointer to the base operator address.
 *  @param[in]  param
 *    Input. A PluginYolov3DetectionOutput parameter struct pointer.
 *  @param[in]  yolov3_input_tensors
 *    Input. An array of four-dimensional cnmlTensors with a shape of
 *           [batchNum, (5 + classNum) * numMaskGroup, inputH, inputW](NCHW).
 *           Currently support FLOAT16/FlOAT32 dataType.
 *  @param[in] yolov3_ output_tensors
 *    Input. An array of four-dimensional cnmlTensors with a shape of
 *           [batchNum, 64 + 7 * numMaxBox, 1, 1](NCHW).
 *           Currently support FLOAT16/FLOAT32 dataType.
 *           The first numbers of each batch store the number of
 *           detected boxes. The data for each box starts from the 65th number,
 *           with an order of [batchId, classId, score, x1, y1, x2, y2], where
 *           (x1, y1) and (x2, y2) are the coordinates of top-left and bottom-
 *           -right points accordingly. Since FLOAT16 data's integer part
 *           becomes inaccurate when value is larger than 2048, users must be
 *           careful when numMaxBox is larger than 2048.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Base op pointer is nullptr
 *    - Param is nullptr or not initialized
 *    - Input and output tensor desps is nullptr or inconsistent with param.
 */
cnmlStatus_t cnmlCreatePluginYolov3DetectionOutputOp(
    cnmlBaseOp_t *op,
    cnmlPluginYolov3DetectionOutputOpParam_t param,
    cnmlTensor_t *yolov3_input_tensors,
    cnmlTensor_t *yolov3_output_tensors);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginYolov3DetectionOutputOp on MLU.
 *
 *  **Supports MLU220 and MLU270.**
 *
 *  @param[out]  op
 *    Input. A pointer to the base operator address.
 *  @param[in]  inputs
 *    Input. An array stores the address of all input tensors.
 *  @param[in]  num_inputs
 *    Input. Number of input tensors.
 *  @param[out]  outputs
 *    Output. An array stores the address of all output tensors.
 *  @param[in]  num_outputs
 *    Input. Number of output tensors.
 *  @param[in]  compute_forw_param
 *    Input. A pointer to the struct address, which records runtime degree of
 *    data parallelism and equipment affinity.
 *  @param[in]  queue
 *    Input. A computation queue pointer.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Base op is nullptr.
 *    - Input and output addrs is nullptr.
 *    - Input and output nums are inconsistent.
 *    - Task type is invalid at runtime.
 */
cnmlStatus_t cnmlComputePluginYolov3DetectionOutputOpForward(
    cnmlBaseOp_t op,
    void *input[],
    int num_inputs,
    void *output[],
    int num_outputs,
    cnrtInvokeFuncParam_t *compute_forw_param,
    cnrtQueue_t queue);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginYolov3DetectionOutputOp on CPU.
 *
 *  @param[in]  param
 *    Input. A PluginYolov3DetectionOutput parameter struct pointer.
 *  @param[in]  inputs
 *    Input. An array stores the address of all cpu input data
 *  @param[out]  outputs
 *    Output. An array stores the address of all cpu output data.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Param is nullptr or inconsistent.
 *    - Input and output addrs is nullptr or malloced with wrong sizes.
 */
cnmlStatus_t cnmlCpuComputePluginYolov3DetectionOutputOpForward(
    cnmlPluginYolov3DetectionOutputOpParam_t param,
    void *input[],
    void *output);
/* --------------------------------------------- */
/* cnmlPluginYolov3DetectionOutout operation end */
/* --------------------------------------------- */

/* ================================================== */
/* cnmlPluginDetRetinaDetectionOutout operation start */
/* ================================================== */
/*!
 *  @struct cnmlPluginDetRetinaDetectionOutputOpParam
 *  @brief A struct.
 *
 *  cnmlPluginDetRetinaDetectionOutputOpParam is a structure describing the "param"
 *  parameter of DetRetinaDetectionOutput operation.
 *  cnmlCreatePluginDetRetinaDetectionOutputOpParam() is used to create
 *  an instance of cnmlPluginDetRetinaDetectionOutputOpParam_t.
 *  cnmlDestroyPluginDetRetinaDetectionOutputOpParam() is used to destroy
 *  an instance of cnmlPluginDetRetinaDetectionOutputOpParam_t.
 *  Detailed information of parameters, see cnmlCreatePluginDetRetinaDetectionOutputOpParam().
 */
struct cnmlPluginDetRetinaDetectionOutputOpParam;
/*! ``cnmlPluginDetRetinaDetectionOutputOpParam_t`` is a pointer to a
    structure (cnmlPluginDetRetinaDetectionOutputOpParam) holding the description of a
   DetRetinaDetectionOutput operation param.
*/
typedef cnmlPluginDetRetinaDetectionOutputOpParam *cnmlPluginDetRetinaDetectionOutputOpParam_t;


/*!
 *  @brief A function.
 *
 *  This function creates a PluginDetRetinaDetectionOutputOp param object with
 *  the pointer and parameters provided by user. This implementation is based
 *  on the official caffe website.
 *
 *  Supports Pytorch on MLU220 and MLU270.
 *
 *  @param[out] param
 *    Output. The returning param descriptor.
 *  @param[in] batchNum
 *    Input. The number of input batches.
 *           No default value. A valid batchNum must be in the range of [1, inf).
 *  @param[in] boxNum
 *    Input. The number of input boxes.
 *           No default value. A valid inputNum must be in the range of [1, inf).
 *  @param[in] classNum
 *    Input. The number of input classes.
 *           No default value. A valid classNum must be in the range of [1, 4096].
 *  @param[in] shareLocation
 *    Input. The mark of whether boxes in different classes share coordinates.
 *           Default value is 1. A valid shareLocation must be either 0 or 1.
 *  @param[in] backgroundLabelId
 *    Input. The class index of background.
 *           Default value is 0. A valid backgroundLabelId must be in the range of [0, classNum).
 *  @param[in] codeType
 *    Input. The encoding type of four coordinates of boxes.
 *           Default value is CodeType_CENTER_SIZE. A valid codeType must be from enum
 *           cnmlPluginSsdCodeType_t.
 *  @param[in] variance_encoded_in_target
 *    Input. The mark of whether variance information has been encoded in coordinates.
 *           Default value is 0. A valid variance_encoded_in_target is either 0 or 1.
 *  @param[in] clip
 *    Input. The mark of whether coordinates are restricted in the range of [0, 1];
 *           Default value is 1. A valid variance_encoded_in_target is either 0 or 1.
 *  @param[in] topkNum
 *    Input. The number of topk process.
 *           No default value. A valid topkNum should be in the range of [1, boxNum).
 *  @param[in] keepNum
 *    Input. The number of boxes kept in detretina_detection op.
 *           No default value. A valid keepNum should be in the range of [1, boxNum).
 *  @param[in] const_prior_tensor
 *    Input. The mark of whether prior tensor is const tensor.
 *           Default value is 0. A valid const_prior_tensor is either 0 or 1.
 *  @param[in] pad_size
 *    Input. Padding size of boxNum.
 *           Default value is 64. A valid pad_size is divisible by 64.
 *  @param[in] pad_size_const
 *    Input. Padding size of const prior tensor.
 *           Default value is 64. A valid pad_size_const is divisible by 64.
 *  @param[in] confidence_thresh
 *    Input. Confidence score threshold used in topk process.
 *           No default value. A valid nms_thresh must be in the range of [0, 1].
 *  @param[in] nms_thresh
 *    Input. IOU threshold used in NMS function.
 *           No default value. A valid nms_thresh must be in the range of [0, 1].
 *  @param[in] bbox_xform_clip
 *    Input. max_value for clip used in decode function.
 *           No default value.
 *  @param[in] core_version
 *    Input. Supported core version.
 *           No default value. A valid core_version must be either MLU220 or MLU270.
 *  @param[in] data_type
 *    Input. Data type of input data, either CNML_DATA_FLOAT16 or CNML_DATA_FLOAT32.
 *  @retval CNML_STATUS_SUCCESS
 *    The object was set successfully.
 */
cnmlStatus_t cnmlCreatePluginDetRetinaDetectionOutputOpParam(
    cnmlPluginDetRetinaDetectionOutputOpParam_t *param,
    int batchNum,
    int boxNum,
    int classNum,
    int shareLocation,
    int backgroundLabelId,
    int codeType,
    int variance_encoded_in_target,
    int clip,
    int topkNum,
    int keepNum,
    int const_prior_tensor,
    int pad_size,
    int pad_size_const,
    float confidence_thresh,
    float nms_thresh,
    float bbox_xform_clip,
    cnmlCoreVersion_t core_version,
    cnmlDataType_t data_type);

/*!
 *  @brief A function.
 *
 *  This function frees the PluginDetRetinaDetectionOutputParam struct, pointed by
 *  the pointer provided by user.
 *
 *  Supports Pytorch on MLU220 and MLU270.
 *
 *  @param[in]  param
 *    Input. A pointer to the address of the struct of computation parameters for
 * PluginDetRetinaDetectionOutput operator.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Param is a null pointer.
 *    - The pointer content pointed by param is already freed.
 */
cnmlStatus_t cnmlDestroyPluginDetRetinaDetectionOutputOpParam(
    cnmlPluginDetRetinaDetectionOutputOpParam_t *param);

/*!
 *  @brief A function.
 *
 *  This function creates PluginDetRetinaDetectionOutputOp with proper param,
 *  input, and output tensors.
 *
 *  PluginDetRetinaDetectionOutputOp takes in feature maps and network parameters
 *  and selects valid bounding boxes based on given thresholds.
 *
 *  **Reference:**
 *
 *    This implementation is based on the project provided by customer.
 *
 *  **Formula:**
 *
 *    This op contains four steps:
 *
 *    1. TopkByClass
 *
 *       Select out the boxes in every class according to topk_indices.
 *
 *       for box_conf_idx in (0, boxNum)
 *
 *           filtered_box_conf.push(box_conf(topk_indices))
 *
 *       where box_conf is box scores, box_conf_idx is the index of box_conf.
 *
 *       confidence_thresh is the thresold of confidence scores.
 *
 *       topkNum is the left box number in topk.
 *
 *       filtered_box_conf is the filtered boxes score.
 *
 *       filtered_box is the filtered boxes after topk.
 *
 *    2. DecodeAllBBoxes
 *
 *       Convert input feature maps into real objectness score and coordinates.
 *
 *       for inputIdx in (0, inputNum - 1)
 *
 *          obj = sigmoid(obj_feature);
 *
 *          x   = (x_offset + sigmoid(x_feature)) / inputWs[inputIdx]
 *
 *          y   = (y_offset + sigmoid(y_feature)) / inputHs[inputIdx]
 *
 *          w   = (w_biases * exp(w_feature)) / netw
 *
 *          h   = (h_biases * exp(h_feature)) / neth
 *
 *       where obj, x_feature, y_feature, w_feature, h_feature are data from input feature maps.
 *
 *       x_offset, y_offset are the coordinates of the grid cell in the feature map.
 *
 *       w_biases, h_biases are the shape of the anchor box.
 *
 *    3. Non-maximum Suppression
 *
 *       For each class of data, compute IOU score for every pair of bounding boxes.
 *
 *       If IOU score exceeds the IOU threshold, keep the box with larger score.
 *
 *       x1 = x - w / 2
 *
 *       y1 = y - y / 2
 *
 *       x2 = x + w / 2
 *
 *       y2 = y + y / 2
 *
 *       for classIdx in (0, classNum - 1)
 *
 *        conf = obj * probability[classIdx]
 *
 *        max, maxIdx = findMaxValueAndIndex(conf)
 *
 *        if (max >= confidence_thresh)
 *
 *          for boxIdx in (0, boxNum - 1)
 *
 *            iou = computeIOU(coord_maxIdx, coord_boxIdx)  // where "coords" means x1,y1,x2,y2
 *
 *            if (iou < nms_thresh)
 *
 *              keep coords and conf for boxIdx
 *
 *    4. KeepOnlyTopKResults
 *
 *       Filter out the boxes in all classes with top k scores.
 *
 *       filtered_box = topk(filtered_box_conf, keepNum)
 *
 *       keepNum is the left box number in topk.
 *
 *       filtered_box_conf is the box confidence of all boxes after step Non-maximum Suppression.
 *
 *       filtered_box is the left box after KeepOnlyTopKResults.
 *
 *  **DataType:**
 *
 *    Support half(float16) and float32 type for both input and output tensors.
 *
 *  **Architecture:**
 *
 *    Supports MLU220 and MLU270.
 *
 *  **Framework:**
 *
 *    Support Pytorch.
 *
 *  @param[out]  op
 *    Output. A pointer to the base operator address.
 *  @param[in]  param
 *    Input. A PluginDetRetinaDetectionOutput parameter struct pointer.
 *  @param[in]  detretina_input_tensors
 *    Input. An array of four-dimensional cnmlTensors with a shape of
 *           [batchNum, (5 + classNum) * numMaskGroup, inputH, inputW](NCHW).
 *           Support only FLOAT16 dataType currently.
 *  @param[in]  detretina_output_tensors
 *    Input. An array of four-dimensional cnmlTensors with a shape of
 *           [batchNum, 64 + 7 * numMaxBox, 1, 1](NCHW).
 *           Support only FLOAT16 dataType currently.
 *           The first two numbers of each batch store the number of
 *           detected boxes. The data for each box starts from the 65th number,
 *           with an order of [batchId, classId, score, x1, y1, x2, y2], where
 *           (x1, y1) and (x2, y2) are the coordinates of top-left and bottom-
 *           -right points accordingly.
 *  @param[in]  detretina_static_tensors
 *    Input. An array of prior tensors when CONST_PRIOR_TENSOR is set true.
 *           Otherwise just pass nullptr.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Base op pointer is nullptr.
 *    - Param is nullptr or not initialized.
 *    - Input and output tensor desps is nullptr or inconsistent with param.
 */
cnmlStatus_t cnmlCreatePluginDetRetinaDetectionOutputOp(
    cnmlBaseOp_t *op,
    cnmlPluginDetRetinaDetectionOutputOpParam_t param,
    cnmlTensor_t *detretina_input_tensors,
    cnmlTensor_t *detretina_output_tensor,
    cnmlTensor_t *detretina_static_tensor);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginDetRetinaDetectionOutputOp on MLU.
 *
 *  Supports Pytorch on MLU220 and MLU270.
 *
 *  @param[out]  op
 *    Output. A pointer to the base operator address.
 *  @param[in]  inputs
 *    Input. An array stores the address of all input tensors.
 *  @param[in]  num_inputs
 *    Input. Number of input tensors.
 *  @param[in]  outputs
 *    Input. An array stores the address of all output tensors.
 *  @param[in]  num_outputs
 *    Input. Number of output tensors.
 *  @param[in]  compute_forw_param
 *    Input. A pointer to the struct address, which records runtime degree of
 *    data parallelism and equipment affinity.
 *  @param[in]  queue
 *    Input. A computation queue pointer.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Base op is nullptr.
 *    - Input and output addrs is nullptr.
 *    - Input and output nums are inconsistent.
 *    - Task type is invalid at runtime.
 */
cnmlStatus_t cnmlComputePluginDetRetinaDetectionOutputOpForward(
    cnmlBaseOp_t op,
    void *input[],
    int num_inputs,
    void *output[],
    int num_outputs,
    cnrtInvokeFuncParam_t *compute_forw_param,
    cnrtQueue_t queue);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginDetRetinaDetectionOutputOp on CPU.
 *
 *  Supports Pytorch on MLU220 and MLU270.
 *
 *  @param[in]  param
 *    Input. A PluginDetRetinaDetectionOutput parameter struct pointer.
 *  @param[in]  loc_data
 *    Input. An array stores the bbox location data with a shape of [N C H W].
 *  @param[in]  conf_data
 *    Input. An array stores the bbox confidence data with a shape of [N C H W].
 *  @param[in]  pri_data
 *    Input. An array stores the prior bbox location/variance data with a shape
 *    of [N C H W].
 *  @param[in]  topk_indices_data
 *    Input. An array stores topk box indexes with a shape of [N C].
 *  @param[out]  outputs
 *    Output. An array stores the address of all cpu output data with a shape
 *    of 1 + [N H W 7]. The first number is the number of detected bboxes. The
 *    rest stores the bbox info with an order:
 *    [batchId, classId, score, x1, y1, x2, y2], where (x1, y1) and (x2, y2)
 *    are the coordinates of top-left and bottom-right points accordingly.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Param is nullptr or inconsistent.
 *    - Input and output addrs is nullptr or malloced with wrong sizes.
 */
cnmlStatus_t cnmlCpuComputePluginDetRetinaDetectionOutputOpForward(
    cnmlPluginDetRetinaDetectionOutputOpParam_t param,
    void *loc_data,
    void *conf_data,
    void *pri_data,
    void *topk_indices_data,
    void *output);
/* ------------------------------------------------ */
/* cnmlPluginDetRetinaDetectionOutout operation end */
/* ------------------------------------------------ */


/* ================================ */
/* cnmlPluginOneHot operation start */
/* ================================ */
/*!
 *  @struct cnmlPluginOneHotOpParam
 *  @brief A struct.
 *
 *  cnmlPluginOneHotOpParam is a structure describing the "param"
 *  parameter of OneHot operation.
 *  cnmlCreatePluginOneHotOpParam() is used to create
 *  an instance of cnmlPluginOneHotOpParam_t.
 *  cnmlDestroyPluginOneHotOpParam() is used to destroy
 *  an instance of cnmlPluginOneHotOpParam_t.
 *  Detailed information of parameters, see cnmlCreatePluginOneHotOpParam().
 */
struct cnmlPluginOneHotOpParam;
/*! ``cnmlPluginOneHotOpParam_t`` is a pointer to a
    structure (cnmlPluginOneHotOpParam) holding the description of a OneHot operation param.
*/
typedef cnmlPluginOneHotOpParam *cnmlPluginOneHotOpParam_t;

/*!
 *  @brief A function.
 *
 *  This function creates a PluginOneHotOp param object with
 *  the pointer and parameters provided by user.
 *
 *  **Supports MLU220 and MLU270.**
 *
 *  @param[out] param
 *    Output. The returning param descriptor.
 *  @param[in] core_version
 *    Input. Supported core version, including MLU270.
 *  @param[in] N
 *    Input. The number of batches.
 *  @param[in] H
 *    Input. Height of input tensors.
 *  @param[in] W
 *    Input. The number of classes.
 *  @param[in] C
 *    Input. The number of anchors for every input tensors.
 *  @param[in] depth
 *    Input. The number of classes.
 *  @param[in] onvalue
 *    Input. The locations represented by indices take value onvalue.
 *  @param[in] offvalue
 *    Input. All other locations take value offvalue.
 *  @param[in] axis
 *    Input. The new axis is created at dimension axis.
 *  @retval CNML_STATUS_SUCCESS
 *    The object was set successfully.
 */
cnmlStatus_t cnmlCreatePluginOneHotOpParam(
    cnmlPluginOneHotOpParam_t *param,
    cnmlCoreVersion_t core_version,
    int N,
    int H,
    int W,
    int C,
    int depth,
    float onvalue,
    float offvalue,
	int axis);

/*!
 *  @brief A function.
 *
 *  This function frees the PluginOneHotOpParam struct, pointed
 *  by the pointer provided by user.
 *
 *  **Supports MLU220 and MLU270.**
 *
 *  @param[in]  param
 *    Input. A pointer to the address of the struct of computation parameters.
 *    for PluginOneHot operator.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Param is a null pointer.
 *    - The pointer content pointed by param is already freed.
 */
cnmlStatus_t cnmlDestroyPluginOneHotOpParam(
    cnmlPluginOneHotOpParam_t *param);

/*!
 *  @brief A function.
 *
 *  This function creates PluginOneHotOp with proper param,
 *  input, and output tensors.
 *
 *  **Supports MLU220 and MLU270.**
 *
 *  @param[out]  op
 *    Output. A pointer to the base operator address.
 *  @param[in]  param
 *    Input. A PluginOneHot parameter struct pointer.
 *  @param[in]  input_tensors
 *    Input. An array of four-dimensional cnmlTensors.
 *           Support only INT32 dataType currently.
 *  @param[in]  output_tensors
 *    Input. An array of four-dimensional cnmlTensors.
 *           Support only FLOAT32 dataType currently.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Base op pointer is nullptr.
 *    - Param is nullptr or not initialized.
 *    - Input and output tensor desps is nullptr or inconsistent with param.
 */
cnmlStatus_t cnmlCreatePluginOneHotOp(
    cnmlBaseOp_t *op,
    cnmlPluginOneHotOpParam_t param,
    cnmlTensor_t *input_tensors,
    cnmlTensor_t *output_tensors);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginOneHotOp on MLU.
 *
 *  **Supports MLU220 and MLU270.**
 *
 *  @param[out]  op
 *    Input. A pointer to the base operator address.
 *  @param[in]  input_addrs
 *    Input. An array stores the address of all input tensors.
 *  @param[in]  num_inputs
 *    Input. Number of input tensors.
 *  @param[out]  output_addrs
 *    Output. An array stores the address of all output tensors.
 *  @param[in]  num_outputs
 *    Input. Number of output tensors.
 *  @param[in]  queue
 *    Input. A computation queue pointer.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Base op is nullptr.
 *    - Input and output addrs is nullptr.
 *    - Input and output nums are inconsistent.
 *    - Task type is invalid at runtime.
 */
cnmlStatus_t cnmlComputePluginOneHotOpForward(
    cnmlBaseOp_t op,
    void *input_addrs[],
    int num_inputs,
    void *output_addrs[],
    int num_outputs,
    cnrtQueue_t queue);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginOneHotOp on CPU.
 *
 *  @param[in]  param
 *    Input. A PluginOneHot parameter struct pointer.
 *  @param[in]  indices
 *    Input. An array stores the address of all cpu input data.
 *  @param[out]  dst
 *    Output. An array stores the address of all cpu output data.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Param is nullptr or inconsistent.
 *    - Input and output addrs is nullptr or malloced with wrong sizes.
 */
cnmlStatus_t cnmlCpuComputePluginOneHotOpForward(
   cnmlPluginOneHotOpParam_t param,
   int* indeces,
   float *dst);
/* ------------------------------ */
/* cnmlPluginOneHot operation end */
/* ------------------------------ */

/* =============================== */
/* cnmlPluginRange operation start */
/* =============================== */
/*!
 *  @struct cnmlPluginRangeOpParam
 *  @brief A struct.
 *
 *  cnmlPluginRangeOpParam is a structure describing the "param"
 *  parameter of Range operation.
 *  cnmlCreatePluginRangeOpParam() is used to create
 *  an instance of cnmlPluginRangeOpParam_t.
 *  cnmlDestroyPluginRangeOpParam() is used to destroy
 *  an instance of cnmlPluginRangeOpParam_t.
 *  Detailed information of parameters, see cnmlCreatePluginRangeOpParam().
 */
struct cnmlPluginRangeOpParam;
/*! ``cnmlPluginRangeOpParam_t`` is a pointer to a
    structure (cnmlPluginRangeOpParam) holding the description of a Range operation param.
*/
typedef cnmlPluginRangeOpParam *cnmlPluginRangeOpParam_t;

/*!
 *  @brief A function.
 *
 *  This function creates a PluginRangeOp param object with
 *  the pointer and parameters provided by user.
 *
 *  **Supports MLU220 and 270.**
 *
 *  @param[out] param
 *    Output. The returning param descriptor.
 *  @param[in] core_version
 *    Input. Supported core version, including MLU220 and MLU270.
 *  @retval CNML_STATUS_SUCCESS
 *    The object was set successfully.
 */
cnmlStatus_t cnmlCreatePluginRangeOpParam(
    cnmlPluginRangeOpParam_t *param,
    cnmlCoreVersion_t core_version);

/*!
 *  @brief A function.
 *
 *  This function frees the PluginRangeOpParam struct, pointed
 *  by the pointer provided by user.
 *
 *  **Supports MLU220 and MLU270.**
 *
 *  @param[in]  param
 *    Input. A pointer to the address of the struct of computation parameters
 *    for Range operator.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Param is a null pointer.
 *    - The pointer content pointed by param is already freed.
 */
cnmlStatus_t cnmlDestroyPluginRangeOpParam(
    cnmlPluginRangeOpParam_t *param);

/*!
 *  @brief A function.
 *
 *  This function creates PluginRangeOp with proper param,
 *
 *  **Supports TensorFlow on MLU270.**
 *
 *  @param[out]  op
 *    Output. A pointer to the base operator address.
 *  @param[in]  param
 *    Input. A PluginRangeOp parameter struct pointer.
 *  @param[in]  input_tensors
 *    Input. An array of four-dimensional cnmlTensors.
 *           Support only FLOAT32 dataType currently.
 *  @param[in]  output_tensors
 *    Output. An array of four-dimensional cnmlTensors.
 *           Support only FLOAT32 dataType currently.
 *           The size is the length of result.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Base op pointer is nullptr.
 *    - Param is nullptr or not initialized.
 *    - Input and output tensor desps is nullptr or inconsistent with param.
 */
cnmlStatus_t cnmlCreatePluginRangeOp(
    cnmlBaseOp_t *op,
    cnmlPluginRangeOpParam_t param,
    cnmlTensor_t *input_tensors,
    cnmlTensor_t *output_tensors);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginRangeOp on MLU.
 *
 *  **Supports MLU220 and MLU270.**
 *
 *  @param[out]  op
 *    Input. A pointer to the base operator address.
 *  @param[in]  input_addrs
 *    Input. An array stores the address of all input tensors.
 *  @param[in]  num_inputs
 *    Input. Number of input tensors.
 *  @param[out]  output_addrs
 *    Output. An array stores the address of all output tensors.
 *  @param[in]  num_outputs
 *    Input. Number of output tensors.
 *  @param[in]  queue
 *    Input. A computation queue pointer.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Base op is nullptr.
 *    - Input and output addrs is nullptr.
 *    - Input and output nums are inconsistent.
 *    - Task type is invalid at runtime.
 */
cnmlStatus_t cnmlComputePluginRangeOpForward(
    cnmlBaseOp_t op,
    void *input_addrs[],
    int num_inputs,
    void *output_addrs[],
    int num_outputs,
    cnrtQueue_t queue);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginRangeOp on CPU.
 *
 *  @param[in]  start
 *    Input.Scalar.Acts as first entry in the range.
 *  @param[in]  limit
 *    Input. Scalar. Upper limit of sequence.
 *  @param[in]  delta
 *    Input. Scalar. Number that increments start.
 *  @param[out]  output
 *    Output. An address of all cpu output data.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Output addrs is nullptr or malloced with wrong sizes.
 */
cnmlStatus_t cnmlCpuComputePluginRangeOpForward(
    float start,
    float limit,
    float delta,
    float *output);
/* ----------------------------- */
/* cnmlPluginRange operation end */
/* ----------------------------- */

/* =============================================== */
/* cnmlPluginRetinaDetectionOutout operation start */
/* =============================================== */
/*!
 *  @struct cnmlPluginRetinaDetectionOutputOpParam
 *  @brief A struct.
 *
 *  cnmlPluginRetinaDetectionOutputOpParam is a structure describing the "param"
 *  parameter of RetinaDetectionOutput operation.
 *  cnmlCreatePluginRetinaDetectionOutputOpParam() is used to create
 *  an instance of cnmlPluginRetinaDetectionOutputOpParam_t.
 *  cnmlDestroyPluginRetinaDetectionOutputOpParam() is used to destroy
 *  an instance of cnmlPluginRetinaDetectionOutputOpParam_t.
 *  Detailed information of parameters, see cnmlCreatePluginRetinaDetectionOutputOpParam().
 */
struct cnmlPluginRetinaDetectionOutputOpParam;
/*! ``cnmlPluginRetinaDetectionOutputOpParam_t`` is a pointer to a
    structure (cnmlPluginRetinaDetectionOutputOpParam) holding the description of a RetinaDetectionOutput operation param.
*/
typedef cnmlPluginRetinaDetectionOutputOpParam
    *cnmlPluginRetinaDetectionOutputOpParam_t;

/*!
 *  @brief A function.
 *
 *  This function creates a PluginRetinaDetectionOutputOp param object with
 *  the pointer and parameters provided by user. This implementation is based
 *  on the official caffe website.
 *
 *  **Supports Caffe and Pytorch on MLU220 and MLU270.**
 *
 *  @param[out] param
 *    Output. The returning param descriptor.
 *  @param[in] batchNum
 *    Input. The number of input batches.
 *           No default value. A valid batchNum must be in the range of [1, inf).
 *  @param[in] boxNum
 *    Input. The number of input boxes.
 *           No default value. A valid inputNum must be in the range of [1, inf).
 *  @param[in] classNum
 *    Input. The number of input classes.
 *           No default value. A valid classNum must be in the range of [1, 4096].
 *  @param[in] shareLocation
 *    Input. The mark of whether boxes in different classes share coordinates.
 *           Default value is 1. A valid shareLocation must be either 0 or 1.
 *  @param[in] backgroundLabelId
 *    Input. The class index of background.
 *           Default value is 0. A valid backgroundLabelId must be in the range of [0, classNum).
 *  @param[in] codeType
 *    Input. The encoding type of four coordinates of boxes.
 *           Default value is CodeType_CENTER_SIZE. A valid codeType must be from enum
 *           cnmlPluginRetinaCodeType_t.
 *  @param[in] variance_encoded_in_target
 *    Input. The mark of whether variance information has been encoded in coordinates.
 *           Default value is 0. A valid variance_encoded_in_target is either 0 or 1.
 *  @param[in] clip
 *    Input. The mark of whether coordinates are restricted in the range of [0, 1].
 *           Default value is 1. A valid variance_encoded_in_target is either 0 or 1.
 *  @param[in] keepNum
 *    Input. The number of boxes kept in retina_detection op.
 *           No default value. A valid keepNum should be in the range of [1, boxNum).
 *  @param[in] const_prior_tensor
 *    Input. The mark of whether prior tensor is const tensor.
 *           Default value is 0. A valid const_prior_tensor is either 0 or 1.
 *  @param[in] pad_size
 *    Input. Padding size of boxNum.
 *           Default value is 64. A valid pad_size is divisible by 64.
 *  @param[in] pad_size_const
 *    Input. Padding size of const prior tensor.
 *           Default value is 64. A valid pad_size_const is divisible by 64.
 *  @param[in] confidence_thresh
 *    Input. Confidence score threshold used in topk process.
 *           No default value. A valid nms_thresh must be in the range of [0, 1].
 *  @param[in] nms_thresh
 *    Input. IOU threshold used in NMS function.
 *           No default value. A valid nms_thresh must be in the range of [0, 1].
 *  @param[in] core_version
 *    Input. Supported core version.
 *           No default value. A valid core_version must be either MLU220 or MLU270.
 *  @retval CNML_STATUS_SUCCESS
 *    The object was set successfully.
 */
cnmlStatus_t cnmlCreatePluginRetinaDetectionOutputOpParam(
    cnmlPluginRetinaDetectionOutputOpParam_t *param,
    int batchNum,
    int boxNum,
    int classNum,
    int shareLocation,
    int backgroundLabelId,
    int codeType,
    int variance_encoded_in_target,
    int clip,
    int keepNum,
    int const_prior_tensor,
    int pad_size,
    int pad_size_const,
    float confidence_thresh,
    float nms_thresh,
    cnmlCoreVersion_t core_version);

/*!
 *  @brief A function.
 *
 *  This function frees the PluginRetinaDetectionOutputParam struct, pointed by
 *  the pointer provided by user.
 *
 *  **Supports Caffe and Pytorch on MLU220 and MLU270.**
 *
 *  @param[in]  param
 *    Input. A pointer to the address of the struct of computation parameters for PluginRetinaDetectionOutput operator.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Param is a null pointer.
 *    - The pointer content pointed by param is already freed.
 */
cnmlStatus_t cnmlDestroyPluginRetinaDetectionOutputOpParam(
    cnmlPluginRetinaDetectionOutputOpParam_t *param);

/*!
 *  @brief A function.
 *
 *  This function creates PluginRetinaDetectionOutputOp with proper param,
 *  input, and output tensors.
 *
 *  PluginRetinaDetectionOutputOp takes in feature maps and network parameters
 *  and selects valid bounding boxes based on given thresholds.
 *
 *  **Reference:**
 *
 *    This implementation is based on the project on ``github/weiliu89``.
 *
 *  **Formula:**
 *
 *    This op contains three steps:
 *
 *    1. DecodeAllBBoxesAndRemoveSmallBoxes
 *
 *       Convert input feature maps into real objectness score and coordinates.
 *
 *       for inputIdx in (0, inputNum - 1)
 *
 *          obj = sigmoid(obj_feature);
 *
 *          x   = (x_offset + sigmoid(x_feature)) / inputWs[inputIdx]
 *
 *          y   = (y_offset + sigmoid(y_feature)) / inputHs[inputIdx]
 *
 *          w   = (w_biases * exp(w_feature)) / netw
 *
 *          h   = (h_biases * exp(h_feature)) / neth
 *
 *       Where obj, x_feature, y_feature, w_feature, h_feature are data from input feature maps.
 *
 *       x_offset, y_offset are the coordinates of the grid cell in the feature map.
 *
 *       w_biases, h_biases are the shape of the anchor box.
 *
 *       Each box has class_num scores, firstly select max score in class_num scores for every box
 *       Then reset boxes' score to zero when its' scores are lower than confidence_thresh
 *
 *    2. Non-maximum Suppression.
 *       Compute IOU score for every pair of bounding boxes.
 *
 *       If IOU score exceeds the IOU threshold, keep the box with larger score.
 *
 *       x1 = x - w / 2
 *
 *       y1 = y - y / 2
 *
 *       x2 = x + w / 2
 *
 *       y2 = y + y / 2
 *
 *       for classIdx in (0, classNum - 1)
 *
 *        conf = obj * probability[classIdx]
 *
 *        max, maxIdx = findMaxValueAndIndex(conf)
 *
 *        if (max >= confidence_thresh)
 *
 *          for boxIdx in (0, boxNum - 1)
 *
 *            iou = computeIOU(coord_maxIdx, coord_boxIdx)  // where "coords" means x1,y1,x2,y2
 *
 *            if (iou < nms_thresh)
 *
 *              keep coords and conf for boxIdx
 *
 *    3. KeepOnlyTopKResults
 *
 *       filtered_box_conf is the box confidence of all boxes after step Non-maximum Suppression.
 *
 *       filtered_box is the left box after KeepOnlyTopKResults.
 *
 *  **DataType:**
 *
 *    Support half(float16) and float32 type for both input and output tensors.
 *
 *  **Performance Optimization:**
 *
 *    The performance of detection layer depends on both the data size and
 *    the value. This op achieves relatively better performance
 *    when all following conditions are met:
 *
 *    - keepNum is 64-aligned and less or equal to 512.
 *
 *  **Supports MLU220 and MLU270.**
 *
 *  @param[out]  op
 *    Output. A pointer to the base operator address.
 *  @param[in]  param
 *    Input. A PluginRetinaDetectionOutput parameter struct pointer.
 *  @param[in]  retina_input_tensors
 *    Input. An array of four-dimensional cnmlTensors with a shape of
 *           [batchNum, (5 + classNum) * numMaskGroup, inputH, inputW](NCHW).
 *           Support only FLOAT16 dataType currently.
 *  @param[in]  retina_output_tensors
 *    Input. An array of four-dimensional cnmlTensors with a shape of
 *           [batchNum, 64 + 7 * numMaxBox, 1, 1](NCHW).
 *           Support only FLOAT16 dataType currently.
 *           The first two numbers of each batch store the number of
 *           detected boxes. The data for each box starts from the 65th number,
 *           with an order of [batchId, classId, score, x1, y1, x2, y2], where
 *           (x1, y1) and (x2, y2) are the coordinates of top-left and bottom-
 *           -right points accordingly.
 *  @param[in]  retina_static_tensors
 *    Input. An array of prior tensors when CONST_PRIOR_TENSOR is set true.
 *           Otherwise just pass nullptr.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Base op pointer is nullptr.
 *    - Param is nullptr or not initialized.
 *    - Input and output tensor desps is nullptr or inconsistent with param.
 */
cnmlStatus_t cnmlCreatePluginRetinaDetectionOutputOp(
    cnmlBaseOp_t *op,
    cnmlPluginRetinaDetectionOutputOpParam_t param,
    cnmlTensor_t *retina_input_tensors,
    cnmlTensor_t *retina_output_tensor,
    cnmlTensor_t *retina_static_tensor);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginRetinaDetectionOutputOp on MLU.
 *
 *  **Supports Caffe and Pytorch on MLU220 and MLU270.**
 *
 *  @param[out]  op
 *    Output. A pointer to the base operator address.
 *  @param[in]  inputs
 *    Input. An array stores the address of all input tensors.
 *  @param[in]  num_inputs
 *    Input. Number of input tensors.
 *  @param[in]  outputs
 *    Input. An array stores the address of all output tensors.
 *  @param[in]  num_outputs
 *    Input. Number of output tensors.
 *  @param[in]  compute_forw_param
 *    Input. A pointer to the struct address, which records runtime degree of
 *    data parallelism and equipment affinity.
 *  @param[in]  queue
 *    Input. A computation queue pointer.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Base op is nullptr.
 *    - Input and output addrs is nullptr.
 *    - Input and output nums are inconsistent.
 *    - Task type is invalid at runtime.
 */
cnmlStatus_t cnmlComputePluginRetinaDetectionOutputOpForward(
    cnmlBaseOp_t op,
    void *input[],
    int num_inputs,
    void *output[],
    int num_outputs,
    cnrtInvokeFuncParam_t *compute_forw_param,
    cnrtQueue_t queue);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginRetinaDetectionOutputOp on CPU.
 *
 *  **Supports Caffe and Pytorch on MLU220 and MLU270.**
 *
 *  @param[in]  param
 *    Input. A PluginRetinaDetectionOutput parameter struct pointer.
 *  @param[in]  loc_data
 *    Input. An array stores the bbox location data with a shape of [N C H W].
 *  @param[in]  conf_data
 *    Input. An array stores the bbox confidence data with a shape of [N C H W].
 *  @param[in]  pri_data
 *    Input. An array stores the prior bbox location and variance data with a shape
 *    of [N C H W].
 *  @param[out]  outputs
 *    Output. An array stores the address of all CPU output data with a shape
 *    of 1 + [N H W 7]. The first number is the number of detected bboxes. The
 *    rest stores the bbox info with an order:
 *    [score, x1, y1, x2, y2, batchId, classId], where (x1, y1) and (x2, y2)
 *    are the coordinates of top-left and bottom-right points accordingly.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Param is nullptr or inconsistent.
 *    - Input and output addrs is nullptr or malloced with wrong sizes.
 */
cnmlStatus_t cnmlCpuComputePluginRetinaDetectionOutputOpForward(
    cnmlPluginRetinaDetectionOutputOpParam_t param,
    void *loc_data,
    void *conf_data,
    void *pri_data,
    void *output);
/* --------------------------------------------- */
/* cnmlPluginRetinaDetectionOutout operation end */
/* --------------------------------------------- */

/* ============================================ */
/* cnmlPluginSsdDetectionOutout operation start */
/* ============================================ */
/*!
 *  @struct cnmlPluginSsdDetectionOutputOpParam
 *  @brief A struct.
 *
 *  cnmlPluginSsdDetectionOutputOpParam is a structure describing the "param"
 *  parameter of SsdDetectionOutput operation.
 *  cnmlCreatePluginSsdDetectionOutputOpParam() is used to create
 *  an instance of cnmlPluginSsdDetectionOutputOpParam_t.
 *  cnmlDestroyPluginSsdDetectionOutputOpParam() is used to destroy
 *  an instance of cnmlPluginSsdDetectionOutputOpParam_t.
 *  Detailed information of parameters, see cnmlCreatePluginSsdDetectionOutputOpParam()
 *  or cnmlCreatePluginSsdDetectionOutputOpParam_V2().
 */
struct cnmlPluginSsdDetectionOutputOpParam;
/*! ``cnmlPluginSsdDetectionOutputOpParam_t`` is a pointer to a
    structure (cnmlPluginSsdDetectionOutputOpParam) holding the description of a SsdDetectionOutput operation param.
*/
typedef cnmlPluginSsdDetectionOutputOpParam
    *cnmlPluginSsdDetectionOutputOpParam_t;

/*!
 *  @brief A function.
 *
 *  This function creates a PluginSsdDetectionOutputOp param object with
 *  the pointer and parameters provided by user. This implementation is based
 *  on the official caffe website.
 *
 *  **Supports Caffe and Pytorch on MLU220 and MLU270.**
 *
 *  @param[out] param
 *    Output. The returning param descriptor.
 *  @param[in] batchNum
 *    Input. The number of input batches.
 *           No default value. A valid batchNum must be in the range of [1, inf).
 *  @param[in] boxNum
 *    Input. The number of input boxes.
 *           No default value. A valid inputNum must be in the range of [1, inf).
 *  @param[in] classNum
 *    Input. The number of input classes.
 *           No default value. A valid classNum must be in the range of [1, 4096].
 *  @param[in] shareLocation
 *    Input. The mark of whether boxes in different classes share coordinates.
 *           Default value is 1. A valid shareLocation must be either 0 or 1.
 *  @param[in] backgroundLabelId
 *    Input. The class index of background.
 *           Default value is 0. A valid backgroundLabelId must be in the range of [0, classNum).
 *  @param[in] codeType
 *    Input. The encoding type of four coordinates of boxes.
 *           Default value is CodeType_CENTER_SIZE. A valid codeType must be from enum
 *           cnmlPluginSsdCodeType_t.
 *  @param[in] variance_encoded_in_target
 *    Input. The mark of whether variance information has been encoded in coordinates.
 *           Default value is 0. A valid variance_encoded_in_target is either 0 or 1.
 *  @param[in] clip
 *    Input. The mark of whether coordinates are restricted in the range of [0, 1];
 *           Default value is 1. A valid variance_encoded_in_target is either 0 or 1.
 *  @param[in] topkNum
 *    Input. The number of topk process.
 *           No default value. A valid topkNum should be in the range of [1, boxNum).
 *  @param[in] keepNum
 *    Input. The number of boxes kept in ssd_detection op.
 *           No default value. A valid keepNum should be in the range of [1, boxNum).
 *  @param[in] const_prior_tensor
 *    Input. The mark of whether prior tensor is const tensor.
 *           Default value is 0. A valid const_prior_tensor is either 0 or 1.
 *  @param[in] pad_size
 *    Input. Padding size of boxNum.
 *           Default value is 64. A valid pad_size is divisible by 64.
 *  @param[in] pad_size_const
 *    Input. Padding size of const prior tensor.
 *           Default value is 64. A valid pad_size_const is divisible by 64.
 *  @param[in] confidence_thresh
 *    Input. Confidence score threshold used in topk process.
 *           No default value. A valid nms_thresh must be in the range of [0, 1].
 *  @param[in] nms_thresh
 *    Input. IOU threshold used in NMS function.
 *           No default value. A valid nms_thresh must be in the range of [0, 1].
 *  @param[in] core_version
 *    Input. Supported core version.
 *           No default value. A valid core_version must be either MLU220 or MLU270.
 *  @retval CNML_STATUS_SUCCESS
 *    The object was set successfully.
 */
cnmlStatus_t cnmlCreatePluginSsdDetectionOutputOpParam(
    cnmlPluginSsdDetectionOutputOpParam_t *param,
    int batchNum,
    int boxNum,
    int classNum,
    int shareLocation,
    int backgroundLabelId,
    int codeType,
    int variance_encoded_in_target,
    int clip,
    int topkNum,
    int keepNum,
    int const_prior_tensor,
    int pad_size,
    int pad_size_const,
    float confidence_thresh,
    float nms_thresh,
    cnmlCoreVersion_t core_version);

/*!
 *  @brief A function.
 *
 *  This function creates a PluginSsdDetectionOutputOp param object with
 *  the pointer and parameters provided by user. This implementation is based
 *  on the official caffe website.
 *
 *  **Supports Caffe and Pytorch on MLU220 and MLU270.**
 *
 *  @param[out] param
 *    Output. The returning param descriptor.
 *  @param[in] batchNum
 *    Input. The number of input batches.
 *           No default value. A valid batchNum must be in the range of [1, inf).
 *  @param[in] boxNum
 *    Input. The number of input boxes.
 *           No default value. A valid inputNum must be in the range of [1, inf).
 *  @param[in] classNum
 *    Input. The number of input classes.
 *           No default value. A valid classNum must be in the range of [1, 4096].
 *  @param[in] shareLocation
 *    Input. The mark of whether boxes in different classes share coordinates.
 *           Default value is 1. A valid shareLocation must be either 0 or 1.
 *  @param[in] backgroundLabelId
 *    Input. The class index of background.
 *           Default value is 0. A valid backgroundLabelId must be in the range of [0, classNum).
 *  @param[in] codeType
 *    Input. The encoding type of four coordinates of boxes.
 *           Default value is CodeType_CENTER_SIZE. A valid codeType must be from enum
 *           cnmlPluginSsdCodeType_t.
 *  @param[in] variance_encoded_in_target
 *    Input. The mark of whether variance information has been encoded in coordinates.
 *           Default value is 0. A valid variance_encoded_in_target is either 0 or 1.
 *  @param[in] clip
 *    Input. The mark of whether coordinates are restricted in the range of [0, 1];
 *           Default value is 1. A valid variance_encoded_in_target is either 0 or 1.
 *  @param[in] topkNum
 *    Input. The number of topk process.
 *           No default value. A valid topkNum should be in the range of [1, boxNum).
 *  @param[in] keepNum
 *    Input. The number of boxes kept in ssd_detection op.
 *           No default value. A valid keepNum should be in the range of [1, boxNum).
 *  @param[in] const_prior_tensor
 *    Input. The mark of whether prior tensor is const tensor.
 *           Default value is 0. A valid const_prior_tensor is either 0 or 1.
 *  @param[in] pad_size
 *    Input. Padding size of boxNum.
 *           Default value is 64. A valid pad_size is divisible by 64.
 *  @param[in] pad_size_const
 *    Input. Padding size of const prior tensor.
 *           Default value is 64. A valid pad_size_const is divisible by 64.
 *  @param[in] confidence_thresh
 *    Input. Confidence score threshold used in topk process.
 *           No default value. A valid nms_thresh must be in the range of [0, 1].
 *  @param[in] nms_thresh
 *    Input. IOU threshold used in NMS function.
 *           No default value. A valid nms_thresh must be in the range of [0, 1].
 *  @param[in] core_version
 *    Input. Supported core version.
 *           No default value. A valid core_version must be either MLU220 or MLU270.
 *  @param[in] data_type
 *    Input. Data type of input data, either CNML_DATA_FLOAT16 or CNML_DATA_FLOAT32.
 *  @retval CNML_STATUS_SUCCESS
 *    The object was set successfully.
 */
cnmlStatus_t cnmlCreatePluginSsdDetectionOutputOpParam_V2(
    cnmlPluginSsdDetectionOutputOpParam_t *param,
    int batchNum,
    int boxNum,
    int classNum,
    int shareLocation,
    int backgroundLabelId,
    int codeType,
    int variance_encoded_in_target,
    int clip,
    int topkNum,
    int keepNum,
    int const_prior_tensor,
    int pad_size,
    int pad_size_const,
    float confidence_thresh,
    float nms_thresh,
    cnmlCoreVersion_t core_version,
    cnmlDataType_t data_type);

/*!
 *  @brief A function.
 *
 *  This function frees the PluginSsdDetectionOutputParam struct, pointed by
 *  the pointer provided by user.
 *
 *  **Supports Caffe and Pytorch on MLU220 and MLU270.**
 *
 *  @param[in]  param
 *    Input. A pointer to the address of the struct of computation parameters for PluginSsdDetectionOutput operator.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Param is a null pointer.
 *    - The pointer content pointed by param is already freed.
 */
cnmlStatus_t cnmlDestroyPluginSsdDetectionOutputOpParam(
    cnmlPluginSsdDetectionOutputOpParam_t *param);

/*!
 *  @brief A function.
 *
 *  This function creates PluginSsdDetectionOutputOp with proper param,
 *  input, and output tensors.
 *
 *  PluginSsdDetectionOutputOp takes in feature maps and network parameters
 *  and selects valid bounding boxes based on given thresholds.
 *
 *  **Reference:**
 *
 *    This implementation is based on the project on ``github/weiliu89``.
 *
 *  **Formula:**
 *
 *    This op contains three steps:
 *
 *    1. TopkByClass
 *
 *       Filter out the boxes in every class with top k scores, and filtered
 *       boxes's scores must be larger than confidence_thresh.
 *
 *       for box_conf_idx in (0, boxNum)
 *
 *         if (box_conf(box_conf_idx) >= confidence_thresh)
 *
 *           filtered_box_conf.push(box_conf(box_conf_idx))
 *
 *       filtered_box = topk(filtered_box_conf, topkNum)
 *
 *       where box_conf is box scores, box_conf_idx is the index of box_conf.
 *
 *       confidence_thresh is the thresold of confidence scores.
 *
 *       topkNum is the left box number in topk.
 *
 *       filtered_box_conf is the filtered boxes by thresh confidence_thresh.
 *
 *       filtered_box is the filtered boxes after topk.
 *
 *    2. DecodeAllBBoxes
 *
 *       Convert input feature maps into real objectness score and coordinates.
 *
 *       for inputIdx in (0, inputNum - 1)
 *
 *          obj = sigmoid(obj_feature);
 *
 *          x   = (x_offset + sigmoid(x_feature)) / inputWs[inputIdx]
 *
 *          y   = (y_offset + sigmoid(y_feature)) / inputHs[inputIdx]
 *
 *          w   = (w_biases * exp(w_feature)) / netw
 *
 *          h   = (h_biases * exp(h_feature)) / neth
 *
 *       where obj, x_feature, y_feature, w_feature, h_feature are data from input feature maps.
 *
 *       x_offset, y_offset are the coordinates of the grid cell in the feature map.
 *
 *       w_biases, h_biases are the shape of the anchor box.
 *
 *    3. Non-maximum Suppression
 *
 *       For each class of data, compute IOU score for every pair of bounding boxes.
 *
 *       If IOU score exceeds the IOU threshold, keep the box with larger score.
 *
 *       x1 = x - w / 2
 *
 *       y1 = y - y / 2
 *
 *       x2 = x + w / 2
 *
 *       y2 = y + y / 2
 *
 *       for classIdx in (0, classNum - 1)
 *
 *        conf = obj * probability[classIdx]
 *
 *        max, maxIdx = findMaxValueAndIndex(conf)
 *
 *        if (max >= confidence_thresh)
 *
 *          for boxIdx in (0, boxNum - 1)
 *
 *            iou = computeIOU(coord_maxIdx, coord_boxIdx)  // where "coords" means x1,y1,x2,y2
 *
 *            if (iou < nms_thresh)
 *
 *              keep coords and conf for boxIdx
 *
 *    4. KeepOnlyTopKResults
 *
 *       Filter out the boxes in all classes with top k scores.
 *
 *       filtered_box = topk(filtered_box_conf, keepNum)
 *
 *       keepNum is the left box number in topk.
 *
 *       filtered_box_conf is the box confidence of all boxes after step Non-maximum Suppression.
 *
 *       filtered_box is the left box after KeepOnlyTopKResults.
 *
 *  **DataType:**
 *
 *    Support half(float16) and float32 type for both input and output tensors.
 *
 *  **Performance Optimization:**
 *
 *    The performance of detection layer depends on both the data size and
 *    the value. This op achieves relatively better performance
 *    when all following conditions are met:
 *
 *    - topkNum is 64-aligned and less or equal to 512.
 *
 *    - keepNum is 64-aligned and less or equal to 512.
 *
 *  **Supports MLU220 and MLU270.**
 *
 *  @param[out]  op
 *    Output. A pointer to the base operator address.
 *  @param[in]  param
 *    Input. A PluginSsdDetectionOutput parameter struct pointer.
 *  @param[in]  ssd_input_tensors
 *    Input. An array of four-dimensional cnmlTensors with a shape of
 *           [batchNum, (5 + classNum) * numMaskGroup, inputH, inputW](NCHW).
 *           Support only FLOAT16 dataType currently.
 *  @param[in]  ssd_output_tensors
 *    Input. An array of four-dimensional cnmlTensors with a shape of
 *           [batchNum, 64 + 7 * numMaxBox, 1, 1](NCHW).
 *           Support only FLOAT16 dataType currently.
 *           The first two numbers of each batch store the number of
 *           detected boxes. The data for each box starts from the 65th number,
 *           with an order of [batchId, classId, score, x1, y1, x2, y2], where
 *           (x1, y1) and (x2, y2) are the coordinates of top-left and bottom-
 *           -right points accordingly.
 *  @param[in]  ssd_static_tensors
 *    Input. An array of prior tensors when CONST_PRIOR_TENSOR is set true.
 *           Otherwise just pass nullptr.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Base op pointer is nullptr.
 *    - Param is nullptr or not initialized.
 *    - Input and output tensor desps is nullptr or inconsistent with param.
 */
cnmlStatus_t cnmlCreatePluginSsdDetectionOutputOp(
    cnmlBaseOp_t *op,
    cnmlPluginSsdDetectionOutputOpParam_t param,
    cnmlTensor_t *ssd_input_tensors,
    cnmlTensor_t *ssd_output_tensor,
    cnmlTensor_t *ssd_static_tensor);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginSsdDetectionOutputOp on MLU.
 *
 *  **Supports Caffe and Pytorch on MLU220 and MLU270.**
 *
 *  @param[out]  op
 *    Output. A pointer to the base operator address.
 *  @param[in]  inputs
 *    Input. An array stores the address of all input tensors.
 *  @param[in]  num_inputs
 *    Input. Number of input tensors.
 *  @param[in]  outputs
 *    Input. An array stores the address of all output tensors.
 *  @param[in]  num_outputs
 *    Input. Number of output tensors.
 *  @param[in]  compute_forw_param
 *    Input. A pointer to the struct address, which records runtime degree of
 *    data parallelism and equipment affinity.
 *  @param[in]  queue
 *    Input. A computation queue pointer.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Base op is nullptr.
 *    - Input and output addrs is nullptr.
 *    - Input and output nums are inconsistent.
 *    - Task type is invalid at runtime.
 */
cnmlStatus_t cnmlComputePluginSsdDetectionOutputOpForward(
    cnmlBaseOp_t op,
    void *input[],
    int num_inputs,
    void *output[],
    int num_outputs,
    cnrtInvokeFuncParam_t *compute_forw_param,
    cnrtQueue_t queue);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginSsdDetectionOutputOp on CPU.
 *
 *  **Supports Caffe and Pytorch on MLU220 and MLU270.**
 *
 *  @param[in]  param
 *    Input. A PluginSsdDetectionOutput parameter struct pointer.
 *  @param[in]  loc_data
 *    Input. An array stores the bbox location data with a shape of [N C H W].
 *  @param[in]  conf_data
 *    Input. An array stores the bbox confidence data with a shape of [N C H W].
 *  @param[in]  pri_data
 *    Input. An array stores the prior bbox location/variance data with a shape
 *    of [N C H W].
 *  @param[out]  outputs
 *    Output. An array stores the address of all cpu output data with a shape
 *    of 1 + [N H W 7]. The first number is the number of detected bboxes. The
 *    rest stores the bbox info with an order:
 *    [batchId, classId, score, x1, y1, x2, y2], where (x1, y1) and (x2, y2)
 *    are the coordinates of top-left and bottom-right points accordingly.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Param is nullptr or inconsistent.
 *    - Input and output addrs is nullptr or malloced with wrong sizes.
 */
cnmlStatus_t cnmlCpuComputePluginSsdDetectionOutputOpForward(
    cnmlPluginSsdDetectionOutputOpParam_t param,
    void *loc_data,
    void *conf_data,
    void *pri_data,
    void *output);
/* ------------------------------------------ */
/* cnmlPluginSsdDetectionOutout operation end */
/* ------------------------------------------ */

/* ========================================= */
/* cnmlPluginAnchorGenerator operation start */
/* ========================================= */
/*!
 *  @struct cnmlPluginAnchorGeneratorOpParam
 *  @brief A struct.
 *
 *  cnmlPluginAnchorGeneratorOpParam is a structure describing the "param"
 *  parameter of cnmlPluginAnchorGenerator operation.
 *  cnmlCreatePluginAnchorGereratorOpParam() is used to create
 *  an instance of cnmlPluginAnchorGeneratorOpParam_t.
 *  cnmlDestroyPluginAnchorGereratorOpParam() is used to destroy
 *  an instance of cnmlPluginAnchorGeneratorOpParam_t.
 *  Detailed information of parameters, see cnmlCreatePluginAnchorGereratorOpParam().
 */
struct cnmlPluginAnchorGeneratorOpParam;
/*! ``cnmlPluginAnchorGeneratorOpParam_t`` is a pointer to a
    structure (cnmlPluginAnchorGeneratorOpParam) holding the description of a AnchorGenerator operation param.
*/
typedef struct cnmlPluginAnchorGeneratorOpParam *cnmlPluginAnchorGeneratorOpParam_t;

/*!
 *  @brief A function.
 *
 *  This function creates PluginAnchorGeneratorOp with proper param,
 *  input, and output tensors.
 *
 *  **Supports MLU220 and MLU270.**
 *
 *  @param[out]  op
 *    Output. A pointer to the base operator address.
 *  @param[in]  feature_map_shape_mlu_tensor
 *    Input. A cnmlTensors with a shape of [1, 2, 1, 1](NCHW).
 *           Support only FLOAT16 dataType currently.
 *  @param[in]  grid_anchors_mlu_tensor
 *    Input. A cnmlTensors with a shape of
 *           [1, len(scales) * len(aspect_ratios) * 4,
 *           featuremap_height, featuremap_width](NCHW).
 *           Support only FLOAT16 dataType currently.
 *  @param[in]  param
 *    Input. A PluginAnchorGenerator parameter struct pointer.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 */
cnmlStatus_t cnmlCreatePluginAnchorGeneratorOp(
    cnmlBaseOp_t *op_ptr,
    cnmlPluginAnchorGeneratorOpParam_t param,
    cnmlTensor_t feature_map_shape_mlu_tensor,
    cnmlTensor_t grid_anchors_mlu_tensor
    );

/*!
 *  @brief A function.
 *
 *  This function forwards PluginAnchorGeneratorOp on MLU.
 *
 *  **Supports MLU220 and MLU270.**
 *
 *  @param[in]  param
 *    Input. A PluginAnchorGenerator parameter struct pointer.
 *  @param[out]  anchors
 *    Output. The address cpu output data.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 */
void cnmlCpuComputePluginAnchorGeneratorOpForward(
    cnmlPluginAnchorGeneratorOpParam_t param,
    float *anchors);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginAnchorGeneratorOp on MLU.
 *
 *  **Supports MLU220 and MLU270.**
 *
 *  @param[out]  op
 *    Output. A pointer to the base operator address.
 *  @param[in]  feature_map_shape_mlu
 *    Input. An address of input tensors.
 *  @param[in]  grid_anchors_mlu
 *    Input. An address of output tensors.
 *  @param[in]  forward_param
 *    Input. Which records runtime degree of data parallelism and equipment affinity.
 *  @param[in]  queue
 *    Input. A computation queue pointer.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 */
cnmlStatus_t cnmlComputePluginAnchorGeneratorOpForward(
    cnmlBaseOp_t op,
    void *feature_map_shape_mlu,
    void *grid_anchors_mlu,
    cnrtInvokeFuncParam_t forward_param,
    cnrtQueue_t queue);

/*!
 *  @brief A function.
 *
 *  This function creates a PluginSsdDetectionOutputOp param object with
 *  the pointer and parameters provided by user.
 *
 *  **Supports MLU220 and MLU270.**
 *
 *  @param[out] param
 *    Output. The returning param descriptor.
 *  @param[in] scales
 *    Input. The scales.
 *  @param[in] aspect_ratios
 *    Input. The aspect ratios.
 *  @param[in] base_anchor_sizes
 *    Input. The base anchor sizes.
 *  @param[in] anchor_strides
 *    Input. The strides of anchor.
 *  @param[in] anchor_offsets
 *    Input. The offsets of anchor.
 *  @param[in] image_shape
 *    Input. The shape of image.
 *  @param[in] corner_bbox
 *    Input. If true, the anchor box will be like [x1, y1, x2, y2], else [xc, yc, w, h].
 *  @param[in] clip_window
 *    Input. If true, the anchor will be limited to [0, image_shape].
 *  @param[in] normalize.
 *    Input. If true, the anchor box will be normalized to 0 - 1.
 *  @param[in] x_before
 *    Input. If true, the anchor box will be like [x1, y1, x2, y2] or [xc, yc, w, h], else [y1, x1, y2, x2] or [yc, xc, h, w].
 *  @param[in] grid_height
 *    Input. The height of the grid.
 *  @param[in] grid_width
 *    Input. The width of the grid.
 *  @retval CNML_STATUS_SUCCESS
 *    The object was set successfully.
 */
cnmlStatus_t cnmlCreatePluginAnchorGereratorOpParam(
    cnmlPluginAnchorGeneratorOpParam_t *param_ptr,
    vector<float> scales,
    vector<float> aspect_ratios,
    vector<float> base_anchor_sizes,
    vector<float> anchor_strides,
    vector<float> anchor_offsets,
    vector<int> image_shape,
    bool corner_bbox,
    bool clip_window,
    bool normalize,
    bool x_before,
    int channel,
    int grid_height,
    int grid_width);

/*!
 *  @brief A function.
 *
 *  This function frees the PluginAnchorGereratorParam struct, pointed by
 *  the pointer provided by user.
 *
 *  **Supports MLU220 and MLU270.**
 *
 *  @param[in]  param
 *    Input. A pointer to the address of the struct of computation parameters for PluginSsdDetectionOutput operator.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 */
cnmlStatus_t cnmlDestroyPluginAnchorGereratorOpParam(
    cnmlPluginAnchorGeneratorOpParam_t param);
/* --------------------------------------- */
/* cnmlPluginAnchorGenerator operation end */
/* --------------------------------------- */

/* ================================= */
/* cnmlPluginRoiPool operation start */
/* ================================= */
/*!
 *  @struct cnmlPluginRoiPoolOpParam
 *  @brief A struct.
 *
 *  cnmlPluginRoiPoolOpParam is a structure describing the "param"
 *  parameter of RoiPool operation.
 *  cnmlCreatePluginRoiPoolOpParam() is used to create
 *  an instance of cnmlPluginRoiPoolOpParam_t.
 *  cnmlDestroyPluginRoiPoolOpParam() is used to destroy
 *  an instance of cnmlPluginRoiPoolOpParam_t.
 *  Detailed information of parameters, see cnmlCreatePluginRoiPoolOpParam().
 */
struct cnmlPluginRoiPoolOpParam;
/*! ``cnmlPluginRoiPoolOpParam_t`` is a pointer to a
    structure (cnmlPluginRoiPoolOpParam) holding the description of a RoiPool operation param.
*/
typedef cnmlPluginRoiPoolOpParam *cnmlPluginRoiPoolOpParam_t;

/*!
 *  @brief cnmlCreatePluginRoiPoolOpParam.
 *
 *  This function creates a RoiPoolOp param object with the pointer
 *  and parameters provided by user.
 *
 *  **Supports MLU220 and MLU270.**
 *
 *  @param[out] param
 *    Output. The returning param descriptor.
 *  @param[in] channels
 *    Input. The number of channels.
 *  @param[in] height
 *    Input. The number of height of bottom date.
 *  @param[in] width
 *    Input. The number of width of bottom_data.
 *  @param[in] pooled_height
 *    Input. The number of height after pooling.
 *  @param[in] pooled_width
 *    Input. The number of width after pooling.
 *  @param[in] rois_num
 *    Input. The number of rois.
 *  @param[in] roi_cols
 *    Input. The size of one roi.
 *  @param[in] batch_size
 *    Input. The number of batches.
 *  @param[in] spatial_scale
 *    Input. The scaling ratio.
 *  @param[in] int8_mode
 *    Input. Whether the datatype of input is int8.
 *  @param[in] coreVersion
 *    Input. Supported core version, including MLU220 and MLU270.
 *  @retval CNML_STATUS_SUCCESS
 *    The object was set successfully.
 */
cnmlStatus_t cnmlCreatePluginRoiPoolOpParam(
    cnmlPluginRoiPoolOpParam_t *param,
    int channels,
    int height,
    int width,
    int pooled_height,
    int pooled_width,
    int rois_num,
    int roi_cols,
    int batch_size,
    float spatial_scale,
    int int8_mode,
    cnmlCoreVersion_t coreVersion);

/*!
 *  @brief cnmlDestroyPluginRoiPoolOpParam.
 *
 *  This function frees the PluginRoiPoolOpParam struct, pointed by
 *  the pointer provided by user.
 *
 *  **Supports MLU220 and MLU270.**
 *
 *  @param[in] param
 *    Input. A pointer to the address of the struct of computation parameters
 *           for PluginRoiPool operator.
 *  @retval CNML_STATUS_SUCCESS
 *    The object was set successfully.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Param is a null pointer.
 *    - The pointer content pointed by param is already freed.
 */
cnmlStatus_t cnmlDestroyPluginRoiPoolOpParam(
    cnmlPluginRoiPoolOpParam_t *param);

/*!
 *  @brief cnmlCreatePluginRoiPoolOp.
 *
 *  This function creates PluginRoiPoolOp with proper param, input,
 *  and output tensors.
 *
 *  **Supports MLU220 and MLU270.**
 *
 *  @param[out] op
 *    Output. A pointer to the base operator address.
 *  @param[in] param
 *    Input. A PluginRoiPool parameter struct pointer.
 *  @param[in] roiPool_input_tensors
 *    Input. An array of two four-dimensional cnmlTensors. One is with the shape
 *           of [batch_size, channels, height, width](NCHW), and the other
 *           is with the shape of [batch_size, rois_num, 1, roi_cols](NCHW).
 *           Support FLOAT32 and FLOAT16 datatype.
 *  @param[in] roiPool_output_tensors
 *    Input. An array of a four-dimensional cnmlTensor with a shape of
 *           [batch_size * rois_num, channels, pooled_height, pooled_width](NCHW).
 *  @retval CNML_STATUS_SUCCESS
 *    The object was set successfully.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions are met:
 *    - The operator pointer is null.
 *    - The input pointer is null.
 */
cnmlStatus_t cnmlCreatePluginRoiPoolOp(
    cnmlBaseOp_t *op,
    cnmlPluginRoiPoolOpParam_t param,
    cnmlTensor_t *roiPool_input_tensors,
    cnmlTensor_t *roiPool_output_tensors);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginRoiPoolOp on MLU.
 *
 *  **Supports MLU220 and MLU270.**
 *
 *  @param[out] op
 *    Output. A pointer to the base operator address.
 *  @param[in] inputs
 *    Input. An array stores the address of all input tensor.
 *  @param[in] input_num
 *    Input. The number of input tensors.
 *  @param[in] outputs
 *    Input. An array stores the address of all output tensor.
 *  @param[in] output_num
 *    Input. The number of output tensors.
 *  @param[in] queue
 *    Input. A computation queue pointer.
 *  @retval CNML_STATUS_SUCCESS
 *    The object was set successfully.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions are met:
 *    - The operator pointer is null.
 *    - The input pointer is null.
 */
cnmlStatus_t cnmlComputePluginRoiPoolOpForward(
    cnmlBaseOp_t op,
    void **inputs,
    int input_num,
    void **outputs,
    int output_num,
    cnrtQueue_t queue);

/*!
 *  @brief cnmlCpuComputePluginRoiPoolOpForward.
 *
 *  This function forwards PluginRoiPoolOp on MLU.
 *
 *  **Supports MLU220 and MLU270.**
 *
 *  @param[in] param
 *    Input. A PluginRoiPool parameter struct pointer.
 *  @param[in] input_data
 *    Input. Cpu input bottom data.
 *  @param[in] input_rois
 *    Input. Cpu input bottom rois.
 *  @param[out] output_data
 *    Output. Cpu output data.
 *  @retval CNML_STATUS_SUCCESS
 *    The object was set successfully.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions are met:
 *    - The param pointer is null.
 *    - The input pointer is null.
 */
cnmlStatus_t cnmlCpuComputePluginRoiPoolOpForward(
    cnmlPluginRoiPoolOpParam_t param,
    float *input_data,
    float *input_rois,
    float *output_data);
/* ------------------------------- */
/* cnmlPluginRoiPool operation end */
/* ------------------------------- */

/* ======================================= */
/* cnmlPluginRoiPoolForFpn operation start */
/* ======================================= */
/*!
 *  @struct cnmlPluginRoiPoolForFpnOpParam
 *  @brief A struct.
 *
 *  cnmlPluginRoiPoolForFpnOpParam is a structure describing the "param"
 *  parameter of RoiPoolForFpn operation.
 *  cnmlCreatePluginRoiPoolForFpnOpParam() is used to create
 *  an instance of cnmlPluginRoiPoolForFpnOpParam_t.
 *  cnmlDestroyPluginRoiPoolForFpnOpParam() is used to destroy
 *  an instance of cnmlPluginRoiPoolForFpnOpParam_t.
 */
struct cnmlPluginRoiPoolForFpnOpParam;

/*! ``cnmlPluginRoiPoolForFpnOpParam_t`` is a pointer to a
    structure (cnmlPluginRoiPoolForFpnOpParam) holding the description of
    a RoiPoolForFpn operation param.
*/
typedef cnmlPluginRoiPoolForFpnOpParam *cnmlPluginRoiPoolForFpnOpParam_t;

/*!
 *  @brief cnmlCreatePluginRoiPoolForFpnOpParam.
 *
 *  This function creates a RoiPoolForFpnOp param object with the pointer
 *  and parameters provided by user.
 *
 *  **Supports MLU270**
 *
 *  @param[out] param
 *    Output. The returning param descriptor.
 *  @param[in] channels
 *    Input. The number of channels.
 *  @param[in] pooled_height
 *    Input. The number of height after pooling.
 *  @param[in] pooled_width
 *    Input. The number of width after pooling.
 *  @param[in] rois_num
 *    Input. The number of rois.
 *  @param[in] roi_cols
 *    Input. The size of one roi.
 *  @param[in] roi_offset
 *    Input. The Data order of one roi.
 *           0: (x1, y1, x2, y2, batch_index)
 *           1: (batch_index, x1, y1, x2, y2)
 *  @param[in] batch_size
 *    Input. The number of batches.
 *  @param[in] k_min
 *    Input. The Finest level of FPN (e.g., 2).
 *  @param[in] k_max
 *    Input. The Coarsest level of FPN (e.g., 5).
 *  @param[in] canonical_scale
 *    Input. The canonical pre-training size (e.g., 224).
 *  @param[in] canonical_level
 *    Input. The target level on which an ROI should be mapped into (e.g., 4).
 *  @param[in] num_level
 *    Input. The level number of input.
 *  @param[in] coreVersion
 *    Input. Supported core version, including MLU270.
 *  @retval CNML_STATUS_SUCCESS
 *    The object was set successfully.
 *  @retval CNML_STATUS_INVALIDPARAM
 */
cnmlStatus_t cnmlCreatePluginRoiPoolForFpnOpParam(
    cnmlPluginRoiPoolForFpnOpParam_t *param,
    int channels,
    int pooled_height,
    int pooled_width,
    int rois_num,
    int roi_cols,
    int roi_offset,
    int batch_size,
    int k_min,
    int k_max,
    int canonical_scale,
    int canonical_level,
    int num_level,
    cnmlCoreVersion_t coreVersion);

/*!
 *  @brief cnmlDestroyPluginRoiPoolForFpnOpParam.
 *
 *  This function frees the PluginRoiPoolForFpnOpParam struct, pointed by
 *  the pointer provided by user.
 *
 *  **Supports MLU270**
 *
 *  @param[in] param
 *    Input. A pointer to the address of the struct of computation parameters
 *           for PluginRoiPoolForFpn operator.
 *  @retval CNML_STATUS_SUCCESS
 *    The object was set successfully.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Param is a null pointer.
 *    - The pointer content pointed by param is already freed.
 */
cnmlStatus_t cnmlDestroyPluginRoiPoolForFpnOpParam(
    cnmlPluginRoiPoolForFpnOpParam_t *param);

/*!
 *  @brief cnmlCreatePluginRoiPoolForFpnOp.
 *
 *  This function creates PluginRoiPoolForFpnOp with proper param,
 *  input, and output tensors.
 *
 *  PluginRoiPoolForFpnOp mainly takes feature maps of FPN and rois as inputs.
 *  Each region proposal's "area" of rois is extracted from the corresponding
 *  feature maps and converted into a fixed-size feature map.
 *
 *  **Reference:**
 *    This implementation is based on the project on ``github/guoruoqian/DetNet_pytorch`` .
 *
 *  **Formula:**
 *    Compared with the normal RoiPooling Op, the RoiPoolingForFpn Op adds one mapping process
 *    from roi to multi-scale feature maps before the normal roipooling process.
 *    This op mainly contains two steps:
 *
 *    1. Select the corresponding feature map from multi-scale feature maps based on the roi.
 *
 *       From the following formula, it can get the corresponding feature map of this roi.
 *
 *       k = floor(canonical_level + log2(sqrt(w * h) / canonical_scale))
 *
 *           k: the selected index of multi-scale feature maps
 *
 *           canonical_level: the target level on which an ROI should be mapped into (e.g., 4).
 *
 *           canonical_scale: The canonical pre-training size (e.g., 224).
 *
 *           w: the width of this roi
 *
 *           h: the height of this roi
 *
 *    2. Normal roipooling process
 *       Ater the roi has found its corresponding feature map, then it will do the normal
 *       roipooling process to get the fixed-size feature map.
 *
 *  **DataType:**
 *
 *    Support half(float16) and float32 type for both input and output tensors.
 *
 *  **Supports MLU270**
 *
 *  @param[out] op
 *    Output. A pointer to the base operator address.
 *  @param[in] param
 *    Input. A PluginRoiPoolForFpn parameter struct pointer.
 *  @param[in] roiPoolForFpn_input_tensors
 *    Input. An array of eight four-demensional cnmlTensors. The seven inputs in front
 *           are the max feature map output number of FPN. They are with the shape
 *           of [batch_size, channels, height, width](NCHW). The last input is the
 *           rois and is with the shape of [batch_size, rois_num, 1, roi_cols](NCHW).
 *           Support FLOAT32 and FLOAT16 datatype.
 *  @param[in] roiPoolForFpn_output_tensors
 *    Input. An array of a four-demensional cnmlTensor with a shape of
 *           [batch_size * rois_num, channels, pooled_height, pooled_width](NCHW).
 *  @param[in] roiPoolForFpn_static_tensors
 *    Input. An array of two four-demensional cnmlTensors. The one is with a shape of
 *           [1, 2 * num_level, 1, 1](NCHW) and it is used to keep the height and widht
 *           of feature maps. The other is with a shape of [batch_size, 3, 1, 1](NCHW)
 *           and it is used to keep the image height and width info.
 *  @retval CNML_STATUS_SUCCESS
 *    The object was set successfully.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions are met:
 *    - The operator pointer is null.
 *    - The input pointer is null.
 */
cnmlStatus_t cnmlCreatePluginRoiPoolForFpnOp(
    cnmlBaseOp_t *op,
    cnmlPluginRoiPoolForFpnOpParam_t param,
    cnmlTensor_t *roiPoolForFpn_input_tensors,
    cnmlTensor_t *roiPoolForFpn_output_tensors,
    cnmlTensor_t *roiPoolForFpn_static_tensors);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginRoiPoolForFpnOp on MLU.
 *
 *  **Supports MLU270**
 *
 *  @param[out] op
 *    Output. A pointer to the base operator address.
 *  @param[in] inputs
 *    Input. An array stores the address of all input tensor.
 *  @param[in] input_num
 *    Input. The number of input tensors.
 *  @param[in] outputs
 *    Input. An array stores the address of all output tensor.
 *  @param[in] output_num
 *    Input. The number of output tensors.
 *  @param[in] queue
 *    Input. A computation queue pointer.
 *  @retval CNML_STATUS_SUCCESS
 *    The object was set successfully.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions are met:
 *    - The operator pointer is null.
 *    - The input pointer is null.
 */
cnmlStatus_t cnmlComputePluginRoiPoolForFpnOpForward(
    cnmlBaseOp_t op,
    void **inputs,
    int input_num,
    void **outputs,
    int output_num,
    cnrtQueue_t queue);

/*!
 *  @brief cnmlCpuComputePluginRoiPoolForFpnOpForward.
 *
 *  This function forwards PluginRoiPoolForFpnOp on CPU.
 *
 *  @param[in] param
 *    Input. A PluginRoiPoolForFpn parameter struct pointer.
 *  @param[in] input_data
 *    Input. Cpu input bottom data.
 *  @param[in] input_rois
 *    Input. Cpu input bottom rois.
 *  @param[in] height_width
 *    Input. The height and width input of feature maps.
 *  @param[in] image_info
 *    Input. The image info of origin input.
 *  @param[out] output_data
 *    Output. Cpu output data.
 *  @retval CNML_STATUS_SUCCESS
 *    The object was set successfully.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions are met:
 *    - The param pointer is null.
 *    - The input pointer is null.
 */
cnmlStatus_t cnmlCpuComputePluginRoiPoolForFpnOpForward(
    cnmlPluginRoiPoolForFpnOpParam_t param,
    void *input_data[],
    void *input_rois,
    int *height_width,
    float *image_info,
    void *output);
/* ------------------------------------- */
/* cnmlPluginRoiPoolForFpn operation end */
/* ------------------------------------- */

/* ================================== */
/* cnmlPluginProposal operation start */
/* ================================== */
/*!
 *  @struct cnmlPluginProposalOpParam
 *  @brief A struct.
 *
 *  cnmlPluginProposalOpParam is a structure describing the "param"
 *  parameter of Proposal operation.
 *  cnmlCreatePluginProposalOpParam() is used to create
 *  an instance of cnmlPluginProposalOpParam_t.
 *  cnmlDestroyPluginProposalOpParam() is used to destroy
 *  an instance of cnmlPluginProposalOpParam_t.
 *  Detailed information of parameters, see cnmlCreatePluginProposalOpParam().
 */
struct cnmlPluginProposalOpParam;
/*! ``cnmlPluginProposalOpParam_t`` is a pointer to a
    structure (cnmlPluginProposalOpParam) holding the description of a Proposal operation param.
*/
typedef cnmlPluginProposalOpParam *cnmlPluginProposalOpParam_t;

/*!
 *  @brief A function.
 *
 *  This function creates a ProposalOp param object with
 *  the pointer and parameters provided by user.
 *
 *  **Supports MLU220 and MLU270.**
 *
 *  @param[out] param
 *    Output. The returning param descriptor.
 *  @param[in] batch_size
 *    Input. Batch_size of input images for network.
 *  @param[in] height
 *    Input. Height of the feature map.
 *  @param[in] width
 *    Input. Width of the feature map.
 *  @param[in] anchor_num
 *    Input. Number of anchors of every point in the feature map.
 *  @param[in] nms_num
 *    Input. Number of boxes to be select in nms process.
 *  @param[in] top_thresh
 *    Input. Number of boxes selected in TopK process.
 *  @param[in] im_min_h
 *    Input. The minimum size of height for boxes selected.
 *  @param[in] im_min_w
 *    Input. The minimum size of width for boxes selected.
 *  @param[in] nms_scale
 *    Input. The scaling rate of boxes when computing areas of box in nms process.
 *  @param[in] stride
 *    Input. Stride in computing anchor. Unused.
 *  @param[in] nms_thresh
 *    Input. Threshold of IOU in nms process.
 *  @param[in] im_h
 *    Input. Height of input images for network.
 *  @param[in] im_w
 *    Input. Width of input images for network.
 *  @param[in] scale
 *    Input. The scaling rate of the size of input images.
 *  @param[in] fix8
 *    Input. Type of input. Unused.
 *  @param[in] core_version
 *    Input. Supported core version, including MLU220 and MLU270.
 *  @param[in] anchor
 *    Input. The anchor of boxes' coordinates.
 *   @retval CNML_STATUS_SUCCESS
 *     The object was set successfully.
 */
cnmlStatus_t cnmlCreatePluginProposalOpParam(
    cnmlPluginProposalOpParam_t *param,
    int batch_size,
    int height,
    int width,
    int anchor_num,
    int nms_num,
    int top_thresh,
    float im_min_h,
    float im_min_w,
    float nms_scale,
    float stride,
    float nms_thresh,
    float im_h,
    float im_w,
    float scale,
    int fix8,
    cnmlCoreVersion_t core_version,
    float *anchor);

/*!
 *  @brief A function.
 *
 *  This function creates a ProposalOp param object with
 *  the pointer and parameters provided by user.
 *
 *  **Supports MLU220 and MLU270.**
 *
 *  @param[out] param
 *    Output. The returning param descriptor.
 *  @param[in] batch_size
 *    Input. Batch_size of input images for network.
 *  @param[in] height
 *    Input. Height of the feature map.
 *  @param[in] width
 *    Input. Width of the feature map.
 *  @param[in] anchor_num
 *    Input. Number of anchors of every point in the feature map.
 *  @param[in] nms_num
 *    Input. Number of boxes to be select in nms process.
 *  @param[in] top_thresh
 *    Input. Number of boxes selected in TopK process.
 *  @param[in] im_min_h
 *    Input. The minimum size of height for boxes selected.
 *  @param[in] im_min_w
 *    Input. The minimum size of width for boxes selected.
 *  @param[in] nms_scale
 *    Input. The scaling rate of boxes when computing areas of box in nms process.
 *  @param[in] stride
 *    Input. Stride in computing anchor. Unused.
 *  @param[in] nms_thresh
 *    Input. Threshold of IOU in nms process.
 *  @param[in] im_h
 *    Input. Height of input images for network.
 *  @param[in] im_w
 *    Input. Width of input images for network.
 *  @param[in] scale
 *    Input. The scaling rate of the size of input images.
 *  @param[in] fix8
 *    Input. Type of input. Unused.
 *  @param[in] core_version
 *    Input. Supported core version, including MLU220 and MLU270.
 *  @param[in] data_type
 *    Input. Datatype of input and output data.
 *  @param[in] anchor
 *    Input. The anchor of boxes' coordinates.
 *   @retval CNML_STATUS_SUCCESS
 *     The object was set successfully.
 */
cnmlStatus_t cnmlCreatePluginProposalOpParam_V2(
    cnmlPluginProposalOpParam_t *param,
    int batch_size,
    int height,
    int width,
    int anchor_num,
    int nms_num,
    int top_thresh,
    float im_min_h,
    float im_min_w,
    float nms_scale,
    float stride,
    float nms_thresh,
    float im_h,
    float im_w,
    float scale,
    int fix8,
    cnmlCoreVersion_t core_version,
    cnmlDataType_t data_type,
    float *anchor);

/*!
 *  @brief A function.
 *
 *  This function frees the PluginProposalParam struct, pointed by
 *  the pointer provided by user.
 *
 *  **Supports MLU220 and MLU270.**
 *
 *  @param[in]  param
 *    Input. A pointer to the address of the struct of computation parameters
 *    for PluginProposal operator.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Param is a null pointer.
 *    - The pointer content pointed by param is already freed.
 */
cnmlStatus_t cnmlDestroyPluginProposalOpParam(
    cnmlPluginProposalOpParam_t *param);

/*!
 *  @brief A function.
 *
 *  This function creates PluginProposalOp with proper param,
 *  input, and output tensors.
 *
 *  **Supports MLU220 and MLU270.**
 *
 *  @param[out]  op
 *    Output. A pointer to the base operator address.
 *  @param[in]  param
 *    Input. A PluginProposal parameter struct pointer.
 *  @param[in] proposal_input_tensors
 *    Input. This pointer contains two array of four-dimensional cnmlTensors,
 *           first tensor's shape is [barch_size, 4, 1,
 *           anchor_num * width * height](NHWC), second tensor's shape is
 *           [batchNum, 2, 1, anchor_num * width * height](NHWC).
 *           Support only FLOAT16 dataType currently.
 *  @param[in] proposal_output_tensors
 *    Input. This pointer contains an array of four_dimensional cnmlTemsors
 *           with a shape of [batch_size, 5, 1, nms_num](NCHW).
 *           Support only FLOAT16 dataType currently.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Base op pointer is nullptr.
 *    - Param is nullptr or not initialized.
 *    - Input and output tensor desps is nullptr or inconsistent with param.
 */
cnmlStatus_t cnmlCreatePluginProposalOp(
    cnmlBaseOp_t *op,
    cnmlPluginProposalOpParam_t param,
    cnmlTensor_t *proposal_input_tensors,
    cnmlTensor_t *proposal_output_tensors);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginProposalOp on MLU.
 *
 *  **Supports MLU220 and MLU270.**
 *
 *  @param[out]  op
 *    Output. A pointer to the base operator address.
 *  @param[in]  inputs
 *    Input. An array stores the address of all input tensors.
 *  @param[in]  num_inputs
 *    Input. Number of input tensors
 *  @param[in]  outputs
 *    Input. An array stores the address of all output tensors.
 *  @param[in]  num_outputs
 *    Input. Number of output tensors.
 *  @param[in]  compute_forw_param
 *    Input. A pointer to the struct address, which records runtime degree of
 *           data parallelism and equipment affinity.
 *  @param[in]  queue
 *    Input. A computation queue pointer.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Base op is nullptr.
 *    - Input and output addrs is nullptr.
 *    - Input and output nums are inconsistent.
 *    - Task type is invalid at runtime.
 */
cnmlStatus_t cnmlComputePluginProposalOpForward(
    cnmlBaseOp_t op,
    void *inputs[],
    int num_inputs,
    void *outputs[],
    int num_outputs,
    cnrtInvokeFuncParam_t *compute_forw_param,
    cnrtQueue_t queue);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginProposalOp on MLU.
 *
 *  **Supports MLU220 and MLU270.**
 *
 *  @param[in]  param
 *    Input. A PluginProposal parameter struct pointer.
 *  @param[in]  inputs
 *    Input. An array stores the address of all cpu input data.
 *  @param[out]  outputs
 *    Output. An array stores the address of all cpu output data.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Param is nullptr or inconsistent.
 *    - Input and output addrs is nullptr or malloced with wrong sizes.
 */
cnmlStatus_t cnmlCpuComputePluginProposalOpForward(
    cnmlPluginProposalOpParam_t param,
    void *input[],
    void *output);
/* -------------------------------------- */
/* cnmlPluginProposalOutout operation end */
/* -------------------------------------- */

/* ======================================== */
/* cnmlPluginFasterrcnnPost operation start */
/* ======================================== */
/*!
 *  @struct cnmlPluginFasterrcnnPostOpParam
 *  @brief A struct.
 *
 *  cnmlPluginFasterrcnnPostOpParam is a structure describing the "param"
 *  parameter of FasterrcnnPostOp operation.
 *  cnmlCreatePluginFasterrcnnPostOpParam() is used to create
 *  an instance of cnmlPluginFasterrcnnPostOpParam_t.
 *  cnmlDestroyPluginFasterrcnnPostOpParam() is used to destroy
 *  an instance of cnmlPluginFasterrcnnPostOpParam_t.
 *  Detailed information of parameters, see cnmlCreatePluginFasterrcnnPostOpParam().
 */
struct cnmlPluginFasterrcnnPostOpParam;
/*! ``cnmlPluginFasterrcnnPostOpParam_t`` is a pointer to a
    structure (cnmlPluginFasterrcnnPostOpParam) holding the description of a FasterrcnnPost operation param.
*/
typedef struct cnmlPluginFasterrcnnPostOpParam
*cnmlPluginFasterrcnnPostOpParam_t;

/*!
 *  @brief A function.
 *
 *  This function creates a FasterrcnnPostOp param object with
 *  the pointer and parameters provided by user. Faster RCNN model.
 *
 *  **Supports MLU220 and MLU270.**
 *  @param[out]  op
 *    Output. A pointer to the base operator address.
 *  @param[in] num_proposal_mlu_tensor.
 *    Input. The number of proposed boxes in each batch.
 *    A four-dimension cnmlTensor_t which shape is [batchNum,1,1, 1](HCHW).
 *  @param[in] box_encoding_mlu_tensor.
 *    Input. The box_encoding from second stage neural network.
 *    A four-dimension cnmlTensor_t which shape is [batchNum,box_align_size, class_align_size, 4](HCHW).
 *  @param[in] class_predictions_mlu_tensor.
 *    Input. The class predicting logit from second stage network.
 *    A four-dimension cnmlTensor_t which shape is [batchNum,max_num_proposals, 1, class_align_size](HCHW).
 *  @param[in] box_proposal_mlu_tensor.
 *    Input. The predicting boxes from region proposal network.
 *    A four-dimension cnmlTensor_t which shape is [batchNum, box_align_size, 1, 4](HCHW).
 *  @param[in] detection_result_tensor.
 *    Input. Bounding boxes params, classIdx, score, x1, y1, x2, y2, and etc.
 *    A four-dimension cnmlTensor_t which shape is [batchNum, output_align_size, 1, 6](HCHW).
 *  @param[in]  param
 *    Input. A PluginFasterrcnnPost parameter struct pointer.
 */
cnmlStatus_t cnmlCreatePluginFasterrcnnPostOp(
    cnmlBaseOp_t *op_ptr,
    cnmlPluginFasterrcnnPostOpParam_t param,
    cnmlTensor_t num_proposal_mlu_tensor,
    cnmlTensor_t box_encoding_mlu_tensor,
    cnmlTensor_t class_predictions_mlu_tensor,
    cnmlTensor_t box_proposal_mlu_tensor,
    cnmlTensor_t detection_result_tensor);

/*!
 *  @brief A function.
 *
 *  This function creates PluginFasterrcnnPostOp with proper param,
 *  input, and output tensors. Faster RCNN model.
 *
 *  **Supports MLU220 and MLU270.**
 *
 *  @param[out] param
 *    Output. The returning param descriptor.
 *  @param[in] box_encoding_count
 *    Input. Count of box_encoding.
 *  @param[in] box_proposal_count
 *    Input. Count of box_proposal.
 *  @param[in] class_logit_count
 *    Input. Count of class_logit.
 *  @param[in] align_class_logit_count
 *    Input. Count of align_class_logit.
 *  @param[in] batch_size
 *    Input. Batch size of this neural network.
 *  @param[in] num_classes
 *    Input. Number of objects's category.
 *  @param[in] score_thresh
 *    Input. The minimal threshold for marking a box as an object.
 *  @param[in] iou_thresh
 *    Input. The minimal threshold for marking a box as a duplicate.
 *  @param[in] max_size_per_class
 *    Input. The number of boxes kept in each class.
 *  @param[in] max_total_size
 *    Input. The total number of boxes kept finally.
 *  @param[in] max_num_proposals
 *    Input. The maximum number of region proposals kept in the first stage.
 *  @param[in] scale_x
 *    Input. The box decoded factor of coordinate of center point.
 *  @param[in] scale_y
 *    Input. The box decoded factor of height and width of box.
 *  @param[in] int8mode
 *    Input. Whether this op is in int8 mode.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 */
cnmlStatus_t cnmlCreatePluginFasterrcnnPostOpParam(
    cnmlPluginFasterrcnnPostOpParam_t *param,
    int box_encoding_count,
    int box_proposal_count,
    int class_logit_count,
    int align_class_logit_count,
    int batch_size,
    int num_classes,
    float score_threshold,
    float iou_threshold,
    int max_size_per_class,
    int max_total_size,
    int max_num_proposals,
    float scale_x,
    float scale_y,
    int int8mode);

/*!
 *  @brief A function.
 *
 *  This function frees the PluginFasterrcnnPostOpParam struct, pointed by
 *  the pointer provided by user.
 *
 *  **Supports MLU220 and MLU270.**
 *
 *  @param[in]  param
 *    Input. A pointer to the address of the struct of computation parameters.
 *    for PluginFasterrcnnPost operator.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Param is a null pointer.
 *    - The pointer content pointed by param is already freed.
 */
cnmlStatus_t cnmlDestroyPluginFasterrcnnPostOpParam(
    cnmlPluginFasterrcnnPostOpParam_t *param_ptr);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginFasterrcnnPostOp on MLU. Faster RCNN model.
 *
 *  **Supports MLU220 and MLU270.**
 *
 *  @param[in] op
 *    Input. A pointer to the base operator address.
 *  @param[in] num_proposal_mlu
 *    Input. MLU pointer of num_proposal.
 *  @param[in] box_encoding_mlu
 *    Input. MLU pointer of box_encoding.
 *  @param[in] class_predictions_mlu
 *    Input. MLU pointer of class_predictions.
 *  @param[in] box_proposal_mlu
 *    Input. MLU pointer of box_proposal.
 *  @param[out] detection_result
 *    Output. MLU pointer of output.
 *  @param[in]  forward_param
 *    Input. A pointer to the struct address, which records runtime degree of
 *    data parallelism and equipment affinity.
 *  @param[in]  queue
 *    Input. A computation queue pointer.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    - op is a null pointer.
 *    - The pointer content pointed by op is already freed.
 */
cnmlStatus_t cnmlComputePluginFasterrcnnPostOpForward(
    cnmlBaseOp_t op,
    void *num_proposal_mlu,
    void *box_encoding_mlu,
    void *class_predictions_mlu,
    void *box_proposal_mlu,
    void *detection_result,
    cnrtInvokeFuncParam_t *forward_param,
    cnrtQueue_t queue);


/*!
 *  @brief A function.
 *  *  This function forwards PluginFasterrcnnPostOp on CPU.
 *
 *  @param[in]  num_proposal_cpu_tensor
 *    Input. CPU tensor of num_proposal input.
 *  @param[in]  num_proposal
 *    Input. CPU pointer of num_proposal input.
 *  @param[in]  box_encoding_cpu_tensor
 *    Input.  CPU tensor of box_encoding input.
 *  @param[in]  box_encoding
 *    Input. CPU pointer of box_encoding input.
 *  @param[in]  box_proposal_cpu_tensor
 *    Input.  CPU tensor of box_proposal input.
 *  @param[in]  box_proposal
 *    Input. CPU pointer of box_proposal input.
 *  @param[in]  class_predictions_cpu_tensor
 *    Input.  CPU tensor of class_predictions input.
 *  @param[in]  class_predictions
 *    Input. CPU pointer of class_predictions input.
 *  @param[in]  true_image_shape_cpu_tensor
 *    Input.  CPU tensor of true_image_shape input.
 *  @param[in]  true_image_shape
 *    Input. CPU pointer of true_image_shape input.
 *  @param[in]  output_result_cpu_tensor
 *    Input.  CPU tensor of  output_result input.
 *  @param[out]  output_result_cpu
 *    Output. CPU pointer of output_result output.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 */

cnmlStatus_t cnmlCpuComputePluginFasterrcnnPostOpForward(
    cnmlPluginFasterrcnnPostOpParam_t param,
    const float *num_proposal,
    const float *box_encoding,
    const float *box_proposal,
    const float *class_predictions,
    const float *true_image_shape,
    float *output_result_cpu);
/* -------------------------------------- */
/* cnmlPluginFasterrcnnPost operation end */
/* -------------------------------------- */

/* ================================= */
/* cnmlPluginSsdPost operation start */
/* ================================= */
/*!
 *  @struct cnmlPluginSsdPostOpParam
 *  @brief A struct.
 *
 *  cnmlPluginSsdPost Param is a structure describing the "param"
 *  parameter of SSD postprocess operation.
 *  cnmlCreatePluginSsdPostOpParam() is used to create
 *  an instance of cnmlPluginSsdPostOpParam_t.
 *  cnmlDestroyPluginSsdPostOpParam() is used to destroy
 *  an instance of cnmlPluginSsdPostOpParam_t.
 *  Detailed information of parameters, see cnmlCreatePluginSsdPostOpParam().
 *
 */
struct cnmlPluginSsdPostOpParam;
/*! ``cnmlPluginSsdPostOpParam_t`` is a pointer to a
    structure (cnmlPluginSsdPostOpParam) holding the description of a SsdPost operation param.
*/
typedef struct cnmlPluginSsdPostOpParam *cnmlPluginSsdPostOpParam_t;
/*!
 *  @brief A function.
 *
 *  This function creates PluginSsdPostOp with proper param,
 *  input, and output tensors. SSD series model.
 *
 *  **Supports MLU220 and MLU270.**
 *
 *  @param[out]  op
 *    Output. A pointer to the base operator address.
 *  @param[in]  param
 *    Input. A PluginSsdPost parameter struct pointer.
 *  @param[in] detection_result_tensor
 *    Input[in]. A four-dimensional cnmlTensors which shape is
 *        [batchNum,max_toal_size_align,1,6](NCHW).
 *  @param[in] image_shape_mlu_tensor
 *    Input[in]. A four-dimensional cnmlTensors which shape is
 *        [batchNum,3,1,1](NCHW).
 *  @param[in] box_encoding_mlu_tensor
 *    Input[in]. A four-dimensional cnmlTensors which shape is
 *        [batchNum,max_num_proposals_align,1,4](NCHW).
 *  @param[in] anchor_mlu_tensor
 *    Input[in]. A four-dimensional cnmlTensors which shape is
 *        [batchNum,max_num_proposals_align,1,4](NCHW).
 *  @param[in] class_predictions_mlu_tensor
 *    Input[in]. A four-dimensional cnmlTensors which shape is
 *        [batchNum,num_classes_align,1,max_num_proposals](NCHW).
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Base op pointer is nullptr.
 *    - Param is nullptr or not initialized.
 *    - Input and output tensor desps is nullptr or inconsistent with param.
 */
cnmlStatus_t cnmlCreatePluginSsdPostOp(
    cnmlBaseOp_t *op_ptr,
    cnmlPluginSsdPostOpParam_t param,
    cnmlTensor_t detection_result_tensor,
    cnmlTensor_t image_shape_mlu_tensor,
    cnmlTensor_t box_encoding_mlu_tensor,
    cnmlTensor_t anchor_mlu_tensor,
    cnmlTensor_t class_predictions_mlu_tensor);


/*!
 *  @brief A function.
 *
 *  This function forwards PluginSsdPostOp on MLU. SSD series model.
 *
 *  **Supports MLU220 and MLU270.**
 *  @param[in]  op
 *    int. A pointer to the base operator address.
 *  @param[in] image_shape_mlu.
 *    Input. MLU pointer of image_shape input.
 *  @param[in] box_encoding_mlu.
 *    Input. MLU pointer of box_encoding input.
 *  @param[in] anchor_mlu.
 *    Input. MLU pointer of anchor input.
 *  @param[in] class_predictions_mlu.
 *    Input. MLU pointer of class_predictions input.
 *  @param[out] detection_result.
 *    Output. MLU pointer of output.
 *  @param[in]  compute_forw_param
 *    Input. A pointer to the struct address, which records runtime degree of data parallelism and
 *    equipment affinity.
 *  @param[in]  queue
 *    Input. A computation queue pointer.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - op is a null pointer.
 *    - The pointer content pointed by op is already freed.
 */

cnmlStatus_t cnmlComputePluginSsdPostOpForward(
    cnmlBaseOp_t op,
    void *image_shape_mlu,
    void *box_encoding_mlu,
    void *anchor_mlu,
    void *class_predictions_mlu,
    void *detection_result,
    cnrtInvokeFuncParam_t forward_param,
    cnrtQueue_t queue);

/*!
 *  @brief A function.
 *
 *  This function creates a SsdPostOp param object with
 *  the pointer and parameters provided by user. SSD series model.
 *
 *  **Supports MLU220 and MLU270.**
 *  @param[out]  param
 *    Output. A PluginSsdPost parameter struct pointer.
 *  @param[in] box_encoding_count
 *    Input. Count of box_encoding.
 *  @param[in] image_shape_count
 *    Input. Count of image_shape.
 *  @param[in] class_predictions_count
 *    Input. Count of class_prediction.
 *  @param[in] anchor_count.
 *    Input. Count of anchor.
 *  @param[in] batch_size
 *    Input. Batch size of this neural network.
 *  @param[in] num_classes
 *    Input. Number of objects's category.
 *  @param[in] score_thresh
 *    Input. The minimal threshold for marking a box as an object.
 *  @param[in] iou_thresh
 *    Input. The minimal threshold for marking a box as a duplicate.
 *  @param[in] max_size_per_class
 *    Input. The number of boxes kept in each class.
 *  @param[in] max_total_size
 *    Input. The total number of boxes kept finally.
 *  @param[in] anchor_num
 *    Input. The maximum number of anchor_num
 *  @param[in] scale_x
 *    Input. The box decoded factor of coordinate of center point.
 *  @param[in] scale_y
 *    Input. The box decoded factor of height and width of box.
 *  @param[in] int8mode
 *    Input. Whether this op is in int8 mode.
 *  @param[out] param
 *    Output. The returning param descriptor.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 */
cnmlStatus_t cnmlCreatePluginSsdPostOpParam(
    cnmlPluginSsdPostOpParam_t *param_ptr,
    int box_encoding_count,
    int image_shape_count,
    int class_predictions_count,
    int anchor_count,
    int batch_size,
    int num_classes,
    float score_threshold,
    float iou_threshold,
    int max_size_per_class,
    int max_total_size,
    int max_num_proposals,
    float scale_x,
    float scale_y,
    int int8mode);

/*!
 *  @brief A function.
 *
 *  This function frees the PluginSsdPostParam struct, pointed by
 *  the pointer provided by user. SSD series model.
 *
 *  **Supports MLU220 and MLU270.**
 *
 *  @param[in]  param
 *    Input. A pointer to the address of the struct of computation parameters
 *    for PluginProposal operator.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 */
cnmlStatus_t cnmlDestroyPluginSsdPostOpParam(
    cnmlPluginSsdPostOpParam_t *param_ptr);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginSsdPostOp on CPU.
 *
 *  @param[in]  param
 *    Input. A PluginSsdPost parameter struct pointer.
 *  @param[in] boxes_cpu_tensor.
 *    Input. CPU tensor of boxes input.
 *  @param[in] boxes.
 *    Input. CPU pointer of boxes input.
 *  @param[in] scores_cpu_tensor.
 *    Input. CPU tensor of scores input.
 *  @param[in] scores.
 *    Input. CPU pointer of scores input.
 *  @param[in] anchors_cpu_tensor.
 *    Input. CPU tensor of anchors input.
 *  @param[in] anchors.
 *    Input. CPU pointer of anchors input.
 *  @param[in] output_cpu_tensor.
 *    Input. CPU tensor of output.
 *  @param[out] detection_output_cpu.
 *    Output. CPU pointer of output.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 */
cnmlStatus_t cnmlCpuComputePluginSsdPostOpForward(
    cnmlPluginSsdPostOpParam_t param,
    const float *boxes,
    const float *scores,
    const float *anchors,
    float *detection_output_cpu);
/* ------------------------------- */
/* cnmlPluginSsdPost operation end */
/* ------------------------------- */

/* ================================== */
/* cnmlPluginAddpadOp operation start */
/* ================================== */
/*!
 *  @struct cnmlPluginAddpadOpParam
 *  @brief A struct.
 *
 *  cnmlPluginAddpadOpParam is a structure describing the "param"
 *  parameter of cnmlPluginAddpadOp operation.
 *  cnmlCreatePluginAddpadOpParam() is used to create
 *  an instance of cnmlPluginAddpadOpParam_t.
 *  cnmlDestroyPluginAddpadOpParam() is used to destroy
 *  an instance of cnmlPluginAddpadOpParam_t.
 *  Detailed information of parameters, see cnmlCreatePluginAddpadOpParam().
 */
struct cnmlPluginAddpadOpParam;
/*! ``cnmlPluginAddpadOpParam_t`` is a pointer to a
    structure (cnmlPluginAddpadOpParam) holding the description of a Addpad operation param.
*/
typedef cnmlPluginAddpadOpParam * cnmlPluginAddpadOpParam_t;

/*!
 *  @brief A function.
 *
 *  This function creates a cnmlPluginAddpadOp param object with
 *  the pointer and parameters provided by user.
 *
 *  **Supports MLU220 and MLU270.**
 *
 *  @param[out] param
 *    Output. The returning param descriptor.
 *  @param[in] batch_size
 *    Input. The number of batches.
 *  @param[in] src_h
 *    Input. Height of input image.
 *  @param[in] src_w
 *    Input. Width of input image.
 *  @param[in] dst_h
 *    Input. Height of output image.
 *  @param[in] dst_w
 *    Input. Width of output image.
 *  @param[in] type_uint8
 *    Input. input data type is uint8_t or not.
 *  @param[in] type_yuv
 *    Input. Input image is YUV 420SP or not.
 *  @retval CNML_STATUS_SUCCESS
 *    The object was set successfully.
 */
cnmlStatus_t cnmlCreatePluginAddpadOpParam(
    cnmlPluginAddpadOpParam_t *param_ptr,
    int batch_size,
    int src_h,
    int src_w,
    int dst_h,
    int dst_w,
    int type_uint8,
    int type_yuv);

/*!
 *  @brief A function.
 *
 *  This function frees the cnmlPluginAddpadOpParam struct, pointed by
 *  the pointer provided by user.
 *
 *  **Supports MLU220 and MLU270.**
 *
 *  @param[in]  param
 *    Input. A pointer to the address of the struct of computation parameters for cnmlPluginAddpadOp operator.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Param is a null pointer.
 *    - The pointer content pointed by param is already freed.
 */
cnmlStatus_t cnmlDestroyPluginAddpadOpParam(
    cnmlPluginAddpadOpParam_t param);

/*!
 *  @brief A function.
 *
 *  This function forwards cnmlPluginAddpadOp on MLU.
 *
 *  **Supports MLU220 and MLU270.**
 *
 *  @param[out]  op
 *    Output. A pointer to the base operator address.
 *  @param[in]  mlu_input_ptr
 *    Input. MLU address of input image pointer.
 *  @param[in]  mlu_padValue_ptr
 *    Input. MLU address of pad value pointer.
 *  @param[in]  mlu_dst_ptr
 *    Input. MLU address of output image pointer.
 *  @param[in]  compute_forward_param
 *    Input. A pointer to the struct address, which records runtime degree of data parallelism and
 *    equipment affinity.
 *  @param[in]  queue
 *    Input. A computation queue pointer.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Base op is nullptr.
 *    - Input and output addrs is nullptr.
 *    - Input and output nums are inconsistent.
 *    - Task type is invalid at runtime.
 */
cnmlStatus_t cnmlComputePluginAddpadOpForward(
    cnmlBaseOp_t op,
    void *mlu_input_ptr,
    void *mlu_padValue_ptr,
    void *mlu_dst_ptr,
    cnrtInvokeFuncParam_t compute_forward_param,
    cnrtQueue_t queue);

/*!
 *  @brief A function.
 *
 *  This function forwards cnmlPluginAddpadOp on CPU.
 *
 *  @param[in]  param
 *    Input. A cnmlPluginAddpadOp parameter struct pointer.
 *  @param[in]  src_cpu_ptr
 *    Input. CPU address of input image.
 *  @param[in]  padValue_cpu_ptr
 *    Input. CPU address of pad value.
 *  @param[in]  dst_cpu_ptr
 *    Input. CPU address of output image.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Param is nullptr or inconsistent.
 *    - Input and output addrs is nullptr or malloced with wrong sizes.
 */
cnmlStatus_t cnmlCpuComputePluginAddpadOpForwad(
    cnmlPluginAddpadOpParam_t param,
    uint8_t *src_cpu_ptr,
    uint8_t *padValue_cpu_ptr,
    uint8_t *dst_cpu_ptr
);

/*!
 *  @brief A function.
 *
 *  This function creates cnmlPluginAddpadOp with proper param,
 *  input, and output tensors.
 *
 *  **Supports MLU220 and MLU270.**
 *
 *  @param[out]  op
 *    Output. A pointer to the base operator address.
 *  @param[in]  param
 *    Input. A cnmlPluginAddpadOp parameter struct pointer.
 *  @param[in]  dst_tensor
 *    Input. A four-dimensional cnmlTensors with a shape of
 *           [batchNum, 1 or 4, src_h, src_w](NCHW).
 *           Support only UINT8 dataType currently.
 *  @param[in]  src_tensor
 *    Input. A four-dimensional cnmlTensors with a shape of
 *           [batchNum, 1 or 4, dst_h, dst_w](NCHW).
 *           Support only UINT8 dataType currently.
 *  @param[in]  dst_tensor
 *    Input. An array of four-dimensional cnmlTensors with a shape of
 *           [1, 3, 1, 1](NCHW).
 *           Support only UINT8 dataType currently.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Base op pointer is nullptr.
 *    - Param is nullptr or not initialized.
 *    - Input and output tensor desps is nullptr or inconsistent with param.
 */
cnmlStatus_t cnmlCreatePluginAddpadOp(
    cnmlBaseOp_t *op_ptr,
    cnmlPluginAddpadOpParam_t param,
    cnmlTensor_t dst_tensor,
    cnmlTensor_t src_tensor,
    cnmlTensor_t value_tensor);

/* -------------------------------- */
/* cnmlPluginAddpadOp operation end */
/* -------------------------------- */

/* ======================================== */
/* cnmlPluginPostProcessRpn operation start */
/* ======================================== */
/*!
 *  @struct cnmlPluginPostProcessRpnOpParam
 *  @brief A struct.
 *
 *  cnmlPluginPostProcessRpnOpParam is a structure describing the "param"
 *  parameter of cnmlPluginPostProcessRpnOp operation.
 *  cnmlCreatePluginPostProcessRpnOpParam() is used to create an instance of
 *  cnmlPluginPostProcessRpnOpParam_t.
 *  cnmlDestroyPluginPostProcessRpnOpParam() is used to destroy an instance of
 *  cnmlPluginPostProcessRpnOpParam_t.
 *  Detailed information of parameters, see cnmlCreatePluginPostProcessRpnOpParam().
 */
struct cnmlPluginPostProcessRpnOpParam;
/*! ``cnmlPluginPostProcessRpnOpParam_t`` is a pointer to a
    structure (cnmlPluginPostProcessRpnOpParam) holding the description of a PostProcessRpn operation param.
*/
typedef cnmlPluginPostProcessRpnOpParam * cnmlPluginPostProcessRpnOpParam_t;

/*!
 *  @brief A function.
 *
 *  This function creates a cnmlPluginPostProcessRpnOp param object with
 *  the pointer and parameters provided by user.
 *
 *  **Supports MLU220 and MLU270.**
 *
 *  @param[out] param
 *    Output. The returning param descriptor.
 *  @param[in]  batch_size
 *    Input. Size of batch.
 *  @param[in]  num_anchors
 *    Input. Number of anchors.
 *  @param[in]  max_nms_out
 *    Input. The max number of outputs of nms.
 *  @param[in]  iou_thresh
 *    Input. The thresh of iou when computing nms.
 *  @param[in]  im_height
 *    Input. The height of image.
 *  @param[in]  im_width
 *    Input. The width of image.
 *  @param[in]  scaled_xy
 *    Input. Coefficient used in bounding-box regression.
 *  @param[in]  scaled_wh
 *    Input. Coefficient used in bounding-box regression.
  *  @param[in]  min_nms_score
 *    Input. Nin score for bbox going to nms.
 *  @retval CNML_STATUS_SUCCESS
 *    The object was set successfully.
 */
cnmlStatus_t cnmlCreatePluginPostProcessRpnOpParam(
    cnmlPluginPostProcessRpnOpParam_t *param_ptr,
    int batch_size,
    int num_anchors,
    int max_nms_out,
    float iou_thresh_,
    float im_height,
    float im_width,
    float scale_xy,
    float scale_wh,
    float min_nms_score);

/*!
 *  @brief A function.
 *
 *  This function frees the cnmlPluginPostProcessRpnOpParam struct, pointed by
 *  the pointer provided by user.
 *
 *  **Supports MLU220 and MLU270.**
 *
 *  @param[in]  param
 *    Input. A pointer to the address of the struct of computation parameters for cnmlPluginPostProcessRpnOp operator.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Param is a null pointer.
 *    - The pointer content pointed by param is already freed.
 */
cnmlStatus_t cnmlDestroyPluginPostProcessRpnOpParam(
    cnmlPluginPostProcessRpnOpParam_t param);

/*!
 *  @brief A function.
 *
 *  This function creates PluginPostProcessRpnOp with proper param,
 *  input, and output tensors.
 *
 *  **Supports MLU220 and MLU270.**
 *
 *  @param[out]  op_ptr
 *    Output. A pointer to the base operator address.
 *  @param[in]  rpn_box_encodings_batch
 *    Input. A tensor describes the bbox data.
 *  @param[in]  rpn_objectness_predictions_with_background_batch
 *    Input. A tensor describes the scores data.
 *  @param[in]  anchors
 *    Input. A tensor describes the anchors.
 *  @param[in]  tmp_tensor
 *    Input. A tensor used to malloc temporary memory on mlu.
 *  @param[in]  proposal_box
 *    Input. The output tensor.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Base op pointer is nullptr
 *    - Param is nullptr or not initialized
 *    - Input and output tensor desps is nullptr or inconsistent with param.
 */
cnmlStatus_t cnmlCreatePluginPostProcessRpnOp(
    cnmlBaseOp_t *op_ptr,
    cnmlPluginPostProcessRpnOpParam_t param,
    cnmlTensor_t rpn_box_encodings_batch,
    cnmlTensor_t rpn_objectness_predictions_with_background_batch,
    cnmlTensor_t anchors,
    cnmlTensor_t tmp_tensor,
    cnmlTensor_t proposal_box);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginPostProcessRpnOp on MLU.
 *
 *  **Supports MLU220 and MLU270.**
 *
 *  @param[in]  op
 *    Input. A pointer to the base operator address.
 *  @param[in]  rpn_box_encodings_batch_mlu_ptr
 *    Input. A pointer to the bbox data on mlu.
 *  @param[in]  rpn_objectness_predictions_with_background_batch_mlu_ptr
 *    Input. A pointer to the scores data on mlu.
 *  @param[in]  anchors_mlu_ptr
 *    Input. A pointer to the anchors on mlu.
 *  @param[in]  tmp_tensor_mlu_ptr
 *    Input. A pointer to the temporary memory on mlu.
 *  @param[out]  proposal_box_mlu_ptr
 *    Output. A pointer to the output on mlu.
 *  @param[in]  compute_forw_param
 *    Input. A pointer to the struct address, which records runtime degree of data parallelism and
 *    equipment affinity.
 *  @param[in]  queue
 *    Input. A computation queue pointer.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Base op is nullptr.
 *    - Input and output addrs is nullptr.
 *    - Input and output nums are inconsistent.
 *    - Task type is invalid at runtime.
 */
cnmlStatus_t cnmlComputePluginPostProcessRpnForward(
    cnmlBaseOp_t op,
    void *rpn_box_encodings_batch_mlu_ptr,
    void *rpn_objectness_predictions_with_background_batch_mlu_ptr,
    void *anchors_mlu_ptr,
    void *tmp_tensor_mlu_ptr,
    void *proposal_box_mlu_ptr,
    cnrtInvokeFuncParam_t forward_param,
    cnrtQueue_t queue);
/* -------------------------------------- */
/* cnmlPluginPostProcessRpn operation end */
/* -------------------------------------- */

/* ========================================= */
/* cnmlPluginResizeYuvToRgba operation start */
/* ========================================= */
/*!
 *  @brief A function.
 *
 *  Deprecated. This interface will be deleted in future version and
 *  cnmlCreatePluginResizeYuvToRgbaOpParam_V2 is recommended to use.
 *
 *  This function creates a PluginResizeYuvToRgbaOp param object with a pointer
 *  and "user params" provided.
 *
 *  @param[out] param
 *    Output. The returning param descriptor.
 *  @param[in] s_row
 *    Input. The row number of src image.
 *  @param[in] s_col
 *    Input. The col number of src image.
 *  @param[in] d_row
 *    Input. The row number of dst image.
 *  @param[in] d_col
 *    Input. The col number of dst image.
 *  @param[in] roi_x
 *    Input. The x-coordinate of top-left corner of roi.
 *  @param[in] roi_y
 *    Input. The y-coordinate of top-left corner of roi.
 *  @param[in] roi_w
 *    Input. The width of roi.
 *  @param[in] roi_h
 *    Input. The height of roi.
 *  @param[in] mode
 *    Input. The color and datatype conversion mode.
 *  @param[in] batchNum
 *    Input. The number of batch of input images. This op regards one image as
 *           one batch.
 *  @param[in] core_version
 *    Input[in]. The hardware core_version.
 *  @retval CNML_STATUS_SUCCESS
 *    The object was set successfully.
 */
cnmlStatus_t cnmlCreatePluginResizeYuvToRgbaOpParam(
  cnmlPluginResizeAndColorCvtParam_t* param,
  int s_row,
  int s_col,
  int d_row,
  int d_col,
  int roi_x,
  int roi_y,
  int roi_w,
  int roi_h,
  ioParams mode,
  int batchNum,
  cnmlCoreVersion_t core_version);

/*!
 *  @brief A function.
 *
 *  Deprecated. This interface will be deleted in future version and
 *  cnmlDestroyPluginResizeYuvToRgbaOpParam_V2 is recommended to use.
 *
 *  This function frees the PluginResizeYuvToRgbaOpParam struct, pointed by
 *  the pointer provided by user.
 *
 *  @param[in]  param
 *    Input. A pointer to the address of the struct of computation parameters
 *    for PluginResizeYuvToRgba operator.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Param is a null pointer.
 *    - The pointer content pointed by param is already freed.
 */
cnmlStatus_t cnmlDestroyPluginResizeYuvToRgbaOpParam(
    cnmlPluginResizeAndColorCvtParam_t* param);

/*!
 *  @brief A function.
 *
 *  Deprecated. This interface will be deleted in future version and
 *  cnmlCreatePluginResizeYuvToRgbaOp_V2 is recommended to use.
 *
 *  This function creates PluginResizeYuvToRgbaOp with proper param,
 *  input, and output tensors. The current implementation is based on the
 *  official caffe website of weiliu86.
 *
 *  **Supports MLU220 and MLU270.**
 *
 *  @param[out] op
 *    Output. A pointer to the base operator address.
 *  @param[in]  param
 *    Input. A PluginResizeAndColorCvt parameter struct pointer.
 *  @param[in] cnml_input_ptr
 *    Input. An array of four-dimensional cnmlTensors for src image.
 *  @param[in] cnml_output_ptr
 *    Input. An array of four-dimensional cnmlTensors for dst image.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - param is not consistant with tensors.
 *    - tensor shapes does not meet reuqirements of YUV420SP.
 */
cnmlStatus_t cnmlCreatePluginResizeYuvToRgbaOp(
  cnmlBaseOp_t *op,
  cnmlPluginResizeAndColorCvtParam_t param,
  cnmlTensor_t *cnml_input_ptr,
  cnmlTensor_t *cnml_output_ptr);

/*!
 *  @brief A function.
 *
 *  Deprecated. This interface will be deleted in future version and
 *  cnmlComputePluginResizeYuvToRgbaOpForward_V2 is recommended to use.
 *
 *  This function forwards PluginResizeYuvToRgbaOp on MLU.
 *
 *  **Supports MLU220 and MLU270.**
 *
 *  @param[out]  op
 *    Output. A pointer to the base operator address.
 *  @param[in]  input_addrs
 *    Input. An array stores the address of all input tensors.
 *  @param[in]  output_addrs
 *    Input. An array stores the address of all output tensors.
 *  @param[in]  compute_forw_param
 *    Input. A pointer to the struct address, which records runtime degree of
 *    data parallelism and equipment affinity.
 *  @param[in]  queue
 *    Input. A computation queue pointer.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Base op is nullptr.
 *    - Input and output addrs is nullptr.
 *    - Input and output nums are inconsistent.
 *    - Task type is invalid at runtime.
 */
cnmlStatus_t cnmlComputePluginResizeYuvToRgbaOpForward(
    cnmlBaseOp_t op,
    cnmlPluginResizeAndColorCvtParam_t param,
    void **input_addrs,
    void **output_addrs,
    cnrtInvokeFuncParam_t compute_forward_param,
    cnrtQueue_t queue);

/*!
 *  @brief A function.
 *
 *  This function creates a PluginResizeYuvToRgbaOp param object with a pointer
 *  and "user params" provided.
 *
 *  @param[out] param
 *    Output. The returning param descriptor.
 *  @param[in] d_row
 *    Input. The row number of dst image.
 *  @param[in] d_col
 *    Input. The col number of dst image.
 *  @param[in] mode
 *    Input. The color and datatype conversion mode.
 *  @param[in] batchNum
 *    Input. The number of batch of input images. This op regards one image as
 *           one batch.
 *  @param[in] keepAspectRatio
 *    Input. The flag indicate if this op keeps aspect ratio for resized images
 *  @param[in] core_version
 *    Input[in]. The hardware core_version.
 *  @retval CNML_STATUS_SUCCESS
 *    The object was set successfully.
 */
cnmlStatus_t cnmlCreatePluginResizeYuvToRgbaOpParam_V2(
  cnmlPluginResizeAndColorCvtParam_t* param,
  int d_row,
  int d_col,
  ioParams mode,
  int batchNum,
  int keepAspectRatio,
  cnmlCoreVersion_t core_version);

/*!
 *  @brief A function.
 *
 *  This function creates a PluginResizeYuvToRgbaOp param object with a pointer
 *  and "user params" provided.
 *
 *  @param[out] param
 *    Output. The returning param descriptor.
 *  @param[in] d_row
 *    Input. The row number of dst image.
 *  @param[in] d_col
 *    Input. The col number of dst image.
 *  @param[in] mode
 *    Input. The color and datatype conversion mode.
 *  @param[in] batchNum
 *    Input. The number of batch of input images. This op regards one image as
 *           one batch.
 *  @param[in] keepAspectRatio
 *    Input. The flag indicate if this op keeps aspect ratio for resized images
 *  @param[in] padMethod
 *    Input. The flag indicates the pad method, 0: padding is on both sides, 1: padding is on the right or bottom side
 *  @param[in] core_version
 *    Input[in]. The hardware core_version.
 *  @retval CNML_STATUS_SUCCESS
 *    The object was set successfully.
 */
cnmlStatus_t cnmlCreatePluginResizeYuvToRgbaOpParam_V3(
  cnmlPluginResizeAndColorCvtParam_t* param,
  int d_row,
  int d_col,
  ioParams mode,
  int batchNum,
  int keepAspectRatio,
  int padMethod,
  cnmlCoreVersion_t core_version);

/*!
 *  @brief A function.
 *
 *  This function frees the PluginResizeYuvToRgbaOpParam struct, pointed by
 *  the pointer provided by user.
 *
 *  @param[in]  param
 *    Input. A pointer to the address of the struct of computation parameters
 *    for PluginResizeYuvToRgba operator.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Param is a null pointer.
 *    - The pointer content pointed by param is already freed.
 */
cnmlStatus_t cnmlDestroyPluginResizeYuvToRgbaOpParam_V2(
    cnmlPluginResizeAndColorCvtParam_t* param);

/*!
 *  @brief A function.
 *
 *  This function creates PluginResizeYuvToRgbaOp with proper param,
 *  input, and output tensors. The current implementation is based on the
 *  official caffe website of weiliu86.
 *
 *  **Supports MLU220 and MLU270.**
 *
 *  @param[out] op
 *    Output. A pointer to the base operator address.
 *  @param[in]  param
 *    Input. A PluginResizeAndColorCvt parameter struct pointer.
 *  @param[in] cnml_input_ptr
 *    Input. An array of four-dimensional cnmlTensors for src image.
 *  @param[in] cnml_output_ptr
 *    Input. An array of four-dimensional cnmlTensors for dst image.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - param is not consistant with tensors.
 *    - tensor shapes does not meet reuqirements of YUV420SP.
 */
cnmlStatus_t cnmlCreatePluginResizeYuvToRgbaOp_V2(
  cnmlBaseOp_t *op,
  cnmlPluginResizeAndColorCvtParam_t param,
  cnmlTensor_t *cnml_input_ptr,
  cnmlTensor_t *cnml_output_ptr);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginResizeYuvToRgbaOp on MLU.
 *
 *  **Supports MLU220 and MLU270.**
 *
 *  @param[out]  op
 *    Output. A pointer to the base operator address.
 *  @param[in] cnml_input_ptr
 *    Input. An array of four-dimensional cnmlTensors for src image.
 *  @param[in]  input_addrs
 *    Input. An array stores the address of all input tensors.
 *  @param[in] cnml_output_ptr
 *    Input. An array of four-dimensional cnmlTensors for dst image.
 *  @param[in]  output_addrs
 *    Input. An array stores the address of all output tensors.
 *  @param[in]  queue
 *    Input. A computation queue pointer.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Base op is nullptr.
 *    - Input and output addrs is nullptr.
 *    - Input and output nums are inconsistent.
 *    - Task type is invalid at runtime.
 */
cnmlStatus_t cnmlComputePluginResizeYuvToRgbaOpForward_V2(
    cnmlBaseOp_t op,
    cnmlPluginResizeAndColorCvtParam_t param,
    cnmlTensor_t* cnml_input_ptr,
    void **input_addrs,
    cnmlTensor_t* cnml_output_ptr,
    void **output_addrs,
    cnrtQueue_t queue);

/*!
 *  @brief A function.
 *
 *  Deprecated. This interface will be deleted in future version and
 *  cnmlCpuComputePluginResizeYuvToRgbaOpForward_V4 is recommended to use.
 *
 *  This function forwards PluginResizeYuvToRgbaOp on CPU.
 *
 *  @param[out] dst
 *    Output. The pointer of dst image.
 *  @param[in] srcY
 *    Input. The pointer of channel Y for src image.
 *  @param[in] srcUV
 *    Input. The pointer of channel UV for src image.
 *  @param[in] fill_color
 *    Input. The pointer of background color.
 *  @param[in] s_row
 *    Input. The row number of src image.
 *  @param[in] s_col
 *    Input. The col number of src image.
 *  @param[in] d_row
 *    Input. The row number of dst image.
 *  @param[in] d_col
 *    Input. The col number of dst image.
 *  @param[in] roi_x
 *    Input. The x-coordinate of top-left corner of roi.
 *  @param[in] roi_y
 *    Input. The y-coordinate of top-left corner of roi.
 *  @param[in] roi_w
 *    Input. The width of roi.
 *  @param[in] roi_h
 *    Input. The height of roi.
 *  @param[in] keepAspectRatio
 *    Input. The mark to keep aspect ratio.
 *  @param[in] mode
 *    Input. The color and datatype conversion mode.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Param is nullptr or inconsistent.
 *    - Input and output addrs is nullptr or malloced with wrong sizes.
 */
cnmlStatus_t cnmlCpuComputePluginResizeYuvToRgbaOpForward_V2(
    unsigned char* dst,
    unsigned char* srcY,
    unsigned char* srcUV,
    unsigned char* fill_color,
    const int s_row,
    const int s_col,
    const int d_row_final,
    const int d_col_final,
    const int roi_x,
    const int roi_y,
    const int roi_w,
    const int roi_h,
    const int batchNum,
    const int keepAspectRatio,
    ioParams mode);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginResizeYuvToRgbaOp on CPU.
 *
 *  @param[out] dst
 *    Output. The pointer of dst image.
 *  @param[in] srcY
 *    Input. The rank pointer of channel Y for src image.
 *  @param[in] srcUV
 *    Input. The rank pointer of channel UV for src image.
 *  @param[in] srcWH
 *    Input. src image size in gdram.
 *  @param[in] roiRect_gdram
 *    Input. roi Rect of src image in gdram.
 *  @param[in] fill_color
 *    Input. The pointer of background color
 *  @param[in] d_row
 *    Input. The row number of dst image.
 *  @param[in] d_col
 *    Input. The col number of dst image.
 *  @param[in] keepAspectRatio
 *    Input. The mark to keep aspect ratio.
 *  @param[in] padMethod
 *    Input. The flag indicates the pad method, 0: padding is on both sides
 *           1: padding is on the right or bottom side
 *  @param[in] mode
 *    Input. The color and datatype conversion mode.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Param is nullptr or inconsistent.
 *    - Input and output addrs is nullptr or malloced with wrong sizes.
 */
cnmlStatus_t cnmlCpuComputePluginResizeYuvToRgbaOpForward_V4(
    unsigned char* dst_batches,
    unsigned char** srcY_batches,
    unsigned char** srcUV_batches,
    int** srcWH,
    int** roiRect,
    unsigned char* fill_color,
    const int d_row_final, const int d_col_final,
    const int batchNum,
    const int keepAspectRatio,
    const int padMethod,
    ioParams mode);
/* --------------------------------------- */
/* cnmlPluginResizeYuvToRgba operation end */
/* --------------------------------------- */

/* ======================================== */
/* cnmlPluginResizeYuvToYuv operation start */
/* ======================================== */
/*!
 *  @brief A function.
 *
 *  This function creates a PluginResizeYuvToYuvOp param object with a pointer
 *  and "user params" provided.
 *
 *  @param[out] param
 *    Output. The returning param descriptor.
 *  @param[in] s_row
 *    Input. The row number of src image.
 *  @param[in] s_col
 *    Input. The col number of src image.
 *  @param[in] d_row
 *    Input. The row number of dst image.
 *  @param[in] d_col
 *    Input. The col number of dst image.
 *  @param[in] mode
 *    Input. The color and datatype conversion mode.
 *  @param[in] batchNum
 *    Input. The number of batch of input images. This op regards one image as
 *           one batch.
 *  @param[in] core_version
 *    Input[in]. The hardware core_version.
 *  @retval CNML_STATUS_SUCCESS
 *    The object was set successfully.
 */
cnmlStatus_t cnmlCreatePluginResizeYuvToYuvOpParam(
  cnmlPluginResizeAndColorCvtParam_t* param,
  int s_row,
  int s_col,
  int d_row,
  int d_col,
  ioParams mode,
  cnmlCoreVersion_t core_version);

/*!
 *  @brief A function.
 *
 *  This function frees the PluginResizeYuvToYuvOpParam struct, pointed by
 *  the pointer provided by user.
 *
 *  @param[in]  param
 *    Input. A pointer to the address of the struct of computation parameters
 *    for PluginResizeYuvToYuv operator.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Param is a null pointer.
 *    - The pointer content pointed by param is already freed.
 */
cnmlStatus_t cnmlDestroyPluginResizeYuvToYuvOpParam(
    cnmlPluginResizeAndColorCvtParam_t* param);

/*!
 *  @brief A function.
 *
 *  This function creates PluginResizeYuvToYuvOp with proper param,
 *  input, and output tensors. The current implementation is based on the
 *  official caffe website of weiliu86.
 *
 *  **Supports only MLU270.**
 *
 *  @param[out] op
 *    Output. A pointer to the base operator address.
 *  @param[in]  param
 *    Input. A PluginResizeAndColorCvt parameter struct pointer.
 *  @param[in] cnml_input_ptr
 *    Input. An array of four-dimensional cnmlTensors for src image.
 *           Follows an order of: 1. src_y_ptrs 2. src_uv_ptrs
 *  @param[in] cnml_output_ptr
 *    Input. An array of four-dimensional cnmlTensors for dst image.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - param is not consistant with tensors.
 *    - tensor shapes does not meet reuqirements of YUV420SP.
 */
cnmlStatus_t cnmlCreatePluginResizeYuvToYuvOp(
  cnmlBaseOp_t *op,
  cnmlPluginResizeAndColorCvtParam_t param,
  cnmlTensor_t *cnml_input_ptr,
  cnmlTensor_t *cnml_output_ptr);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginResizeYuvToYuvOp on MLU.
 *
 *  **Supports only MLU270.**
 *
 *  @param[out]  op
 *    Output. A pointer to the base operator address.
 *  @param[in] cnml_input_ptr
 *    Input. An array of four-dimensional cnmlTensors for src image.
 *  @param[in]  input_addrs
 *    Input. An array stores the address of all input tensors.
 *  @param[in] cnml_output_ptr
 *    Input. An array of four-dimensional cnmlTensors for dst image.
 *  @param[in]  output_addrs
 *    Input. An array stores the address of all output tensors.
 *  @param[in]  queue
 *    Input. A computation queue pointer.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Base op is nullptr.
 *    - Input and output addrs is nullptr.
 *    - Input and output nums are inconsistent.
 *    - Task type is invalid at runtime.
 */
cnmlStatus_t cnmlComputePluginResizeYuvToYuvOpForward(
    cnmlBaseOp_t op,
    cnmlPluginResizeAndColorCvtParam_t param,
    cnmlTensor_t* cnml_input_ptr,
    void **input_addrs,
    cnmlTensor_t* cnml_output_ptr,
    void **output_addrs,
    cnrtQueue_t queue);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginResizeYuvToYuvOp on CPU. Process one batch at
 *  one time.
 *
 *  @param[out] dst_cpu_ptr
 *    Output. The pointer of dst image.
 *  @param[in] param
 *    Input. The ResizeAndColotCvtParam needed by YuvToYuv op.
 *  @param[in] src_cpu_ptr
 *    Input. The pointer of src image.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Param is nullptr or inconsistent.
 *    - Input and output addrs is nullptr or malloced with wrong sizes.
 */
cnmlStatus_t cnmlCpuComputePluginResizeYuvToYuvOpForward(
    cnmlPluginResizeAndColorCvtParam_t param,
    uint8_t *src_cpu_ptr,
    uint8_t *dst_cpu_ptr);

/* -------------------------------------- */
/* cnmlPluginResizeYuvToYuv operation end */
/* -------------------------------------- */

/* ============================================= */
/* cnmlPluginResizeConvert16B16C operation start */
/* ============================================= */
/*!
 *  @brief A function.
 *
 *  This function creates a PluginResizeConvert16B16COp param object with a pointer
 *  and "user params" provided.
 *
 *  @param[out] param
 *    Output. The returning param descriptor.
 *  @param[in] batchNum
 *    Input. The batch number of src image.
 *  @param[in] s_row
 *    Input. The row number of src image.
 *  @param[in] s_col
 *    Input. The col number of src image.
 *  @param[in] d_row
 *    Input. The row number of dst image.
 *  @param[in] d_col
 *    Input. The col number of dst image.
 *  @param[in] roi_x
 *    Input. The x-coordinate of top-left corner of roi.
 *  @param[in] roi_y
 *    Input. The y-coordinate of top-left corner of roi.
 *  @param[in] roi_w
 *    Input. The width of roi.
 *  @param[in] roi_h
 *    Input. The height of roi.
 *  @param[in] mode
 *    Input. The color and datatype conversion mode.
 *  @param[in] batchNum
 *    Input. The number of batch of input images. This op regards one image as
 *           one batch.
 *  @param[in] core_version
 *    Input[in]. The hardware core_version.
 *  @retval CNML_STATUS_SUCCESS
 *    The object was set successfully.
 */
cnmlStatus_t cnmlCreatePluginResizeConvert16B16COpParam(
    cnmlPluginResizeAndColorCvtParam_t *param,
    int batchNum,
    int s_row,
    int s_col,
    int d_row,
    int d_col,
    int roi_x,
    int roi_y,
    int roi_w,
    int roi_h,
    ioParams mode,
    cnmlCoreVersion_t core_version);

/*!
 *  @brief A function.
 *
 *  This function frees the PluginResizeConvert16B16COpParam struct, pointed by
 *  the pointer provided by user.
 *
 *  @param[in]  param
 *    Input. A pointer to the address of the struct of computation parameters
 *    for PluginResizeConvert16B16C operator.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Param is a null pointer.
 *    - The pointer content pointed by param is already freed.
 */
cnmlStatus_t cnmlDestroyPluginResizeConvert16B16COpParam(
    cnmlPluginResizeAndColorCvtParam_t* param);

/*!
 *  @brief A function.
 *
 *  This function creates PluginResizeConvert16B16COp with proper param,
 *  input, and output tensors. The current implementation is based on the
 *  official caffe website of weiliu86.
 *
 *  **Supports Caffe on MLU270.**
 *
 *  @param[out] op
 *    Output. A pointer to the base operator address.
 *  @param[in]  param
 *    Input. A PluginResizeAndColorCvt parameter struct pointer.
 *  @param[in] cnml_rank_ptr
 *    Input. An array of 16 src image input address
 *  @param[in] cnml_output_ptr
 *    Input. An array of four-dimensional cnmlTensors for dst image.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - param is not consistant with tensors.
 *    - tensor shapes does not meet reuqirements of YUV420SP.
 */
cnmlStatus_t cnmlCreatePluginResizeConvert16B16COp(
    cnmlBaseOp_t *op,
    cnmlPluginResizeAndColorCvtParam_t param,
    cnmlTensor_t *cnml_rank_ptr,
    cnmlTensor_t *cnml_output_ptr);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginResizeConvert16B16COp on MLU.
 *
 *  **Supports MLU270.**
 *
 *  @param[out]  op
 *    Output. A pointer to the base operator address.
 *  @param[in]  rank_input_addrs
 *    Input. An array stores the address of 16 input tensors.
 *  @param[in]  output_addrs_addrs
 *    Input. An array stores the address of all output tensors.
 *  @param[in]  compute_forw_param
 *    Input. A pointer to the struct address, which records runtime degree of
 *    data parallelism and equipment affinity.
 *  @param[in]  queue
 *    Input. A computation queue pointer.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Base op is nullptr.
 *    - Input and output addrs is nullptr.
 *    - Input and output nums are inconsistent.
 *    - Task type is invalid at runtime.
 */
cnmlStatus_t cnmlComputePluginResizeConvert16B6COpForward(
    cnmlBaseOp_t op,
    cnmlPluginResizeAndColorCvtParam_t param,
    void **rank_input_addrs,
    void **output_addrs_cpu,
    cnrtInvokeFuncParam_t compute_forw_param,
    cnrtQueue_t stream);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginResizeConvert16B16COp on CPU.
 *
 *  @param[out] dst
 *    Output. The pointer of dst image.
 *  @param[in] src
 *    Input. The pointer of src image.
 *  @param[in] batch_num
 *    Input. The batch number of src image.
 *  @param[in] s_row
 *    Input. The row number of src image.
 *  @param[in] s_col
 *    Input. The col number of src image.
 *  @param[in] d_row
 *    Input. The row number of dst image.
 *  @param[in] d_col
 *    Input. The col number of dst image.
 *  @param[in] roi_x
 *    Input. The x-coordinate of top-left corner of roi.
 *  @param[in] roi_y
 *    Input. The y-coordinate of top-left corner of roi.
 *  @param[in] roi_w
 *    Input. The width of roi.
 *  @param[in] roi_h
 *    Input. The height of roi.
 *  @param[in] mode
 *    Input. The color and datatype conversion mode.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Param is nullptr or inconsistent.
 *    - Input and output addrs is nullptr or malloced with wrong sizes.
 */
cnmlStatus_t cnmlCpuComputePluginResizeConvert16B16COpForward(
    unsigned char* dst,
    unsigned char* src,
    int batch_num,
    int s_row,
    int s_col,
    int d_row,
    int d_col,
    int roi_x,
    int roi_y,
    int roi_w,
    int roi_h,
    ioParams mode);
/* ------------------------------------------- */
/* cnmlPluginResizeConvert16B16C operation end */
/* ------------------------------------------- */

/* ================================ */
/* cnmlPluginResize operation start */
/* ================================ */
/*!
 *  @brief A function.
 *
 *  This function creates a PluginResizeOp param object with a pointer
 *  and "user params" provided.
 *
 *  @param[out] param
 *    Output. The returning param descriptor.
 *  @param[in] s_row
 *    Input. The row number of src image.
 *  @param[in] s_col
 *    Input. The col number of src image.
 *  @param[in] d_row
 *    Input. The row number of dst image.
 *  @param[in] d_col
 *    Input. The col number of dst image.
 *  @param[in] mode
 *    Input. The color and datatype conversion mode.
 *  @param[in] core_version
 *    Input[in]. The hardware core_version.
 *  @retval CNML_STATUS_SUCCESS
 *    The object was set successfully.
 */
cnmlStatus_t cnmlCreatePluginResizeOpParam(
    cnmlPluginResizeAndColorCvtParam_t *param,
    int s_row,
    int s_col,
    int d_row,
    int d_col,
    ioParams mode,
    cnmlCoreVersion_t core_version);

/*!
 *  @brief A function.
 *
 *  This function frees the PluginResizeOpParam struct, pointed by
 *  the pointer provided by user.
 *
 *  @param[in]  param
 *    Input. A pointer to the address of the struct of computation parameters
 *    for PluginResize operator.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Param is a null pointer.
 *    - The pointer content pointed by param is already freed.
 */
cnmlStatus_t cnmlDestroyPluginResizeOpParam(
    cnmlPluginResizeAndColorCvtParam_t *param);

/*!
 *  @brief A function.
 *
 *  This function creates PluginResizeOp with proper param,
 *  input, and output tensors. The current implementation is based on the
 *  traditional bi-linear interpolation method in OpenCV.
 *
 *  **Supports MLU220 and MLU270.**
 *
 *  @param[out]  op
 *    Output. A pointer to the base operator address.
 *  @param[in]  param
 *    Input. A PluginResizeAndColorCvt parameter struct pointer.
 *  @param[in]  dst
 *    Input. A four-dimensional tensor for dst image.
 *  @param[in]  src
 *    Input. A four-dimensional tensor for src image.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Base op pointer is nullptr.
 *    - Param is nullptr or not initialized.
 *    - Input and output tensor desps is nullptr or inconsistent with param.
 */
cnmlStatus_t cnmlCreatePluginResizeOp(
    cnmlBaseOp_t *op,
    cnmlPluginResizeAndColorCvtParam_t param,
    cnmlTensor_t dst,
    cnmlTensor_t src);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginResizeOp on MLU.
 *
 *  **Supports MLU220 and MLU270.**
 *
 *  @param[out] op
 *    Output. A pointer to the base operator address.
 *  @param[in] src_tensor
 *    Input. A four-dimensional tensor for src image.
 *  @param[in] dst_tensor
 *    Input. A four-dimensional tensor for dst image.
 *  @param[in] src_mlu_ptr
 *    Input. Address of input tensor.
 *  @param[in] dst_mlu_ptr
 *    Input. Address of output tensor.
 *  @param[in] compute_forw_param
 *    Input. A pointer to the struct address, which records runtime degree of
 *    data parallelism and equipment affinity.
 *  @param[in] queue
 *    Input. A computation queue pointer.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Base op is nullptr.
 *    - Input and output addrs is nullptr.
 *    - Task type is invalid at runtime.
 */
cnmlStatus_t cnmlComputePluginResizeOpForward(
    cnmlBaseOp_t op,
    cnmlTensor_t src_tensor,
    cnmlTensor_t dst_tensor,
    void *src_mlu_ptr,
    void *dst_mlu_ptr,
    cnrtInvokeFuncParam_t compute_forw_param,
    cnrtQueue_t queue);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginResizeOp on CPU.
 *
 *  @param[in]  param
 *    Input. A PluginResizeAndColorCvt parameter struct pointer.
 *  @param[in] src_cpu_ptr
 *    Input. An array stores the address of all input tensors.
 *  @param[in] dst_cpu_ptr
 *    Input. An array stores the address of all output tensors.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - The addrs malloc failed.
 */
cnmlStatus_t cnmlCpuComputePluginResizeOpForward(
    cnmlPluginResizeAndColorCvtParam_t param,
    uint8_t *src_cpu_ptr,
    uint8_t *dst_cpu_ptr);
/* ------------------------------ */
/* cnmlPluginResize operation end */
/* ------------------------------ */

/* =================================================== */
/* cnmlPluginFasterRcnnDetectionOutout operation start */
/* =================================================== */
/*!
 *  @struct cnmlPluginFasterRcnnDetectionOutputOpParam
 *  @brief A struct.
 *
 *  cnmlPluginFasterRcnnDetectionOutputOpParam is a structure describing the
 *  "param" parameter of FasterRcnnDetectionOutput operation.
 *  cnmlCreatePluginFasterRcnnDetectionOutputOpParam() is used to create an
 *  instance of cnmlPluginFasterRcnnDetectionOutputOpParam_t.
 *  cnmlDestroyPluginFasterRcnnDetectionOutputOpParam() is used to destroy an
 *  instance of cnmlPluginFasterRcnnDetectionOutputOpParam_t.
 *  Detailed information of parameters, see cnmlCreatePluginFasterRcnnDetectionOutputOpParam().
 *
 */

struct cnmlPluginFasterRcnnDetectionOutputOpParam;
/*! ``cnmlPluginFasterRcnnDetectionOutputOpParam_t`` is a pointer to a
    structure (cnmlPluginFasterRcnnDetectionOutputOpParam) holding the description of a FasterRcnnDetectionOutput operation param.
*/
typedef cnmlPluginFasterRcnnDetectionOutputOpParam
*cnmlPluginFasterRcnnDetectionOutputOpParam_t;

/*i
 *  @brief A function.
 *
 *  This function creates a PluginFasterRcnnDetectionOutputOp param object with
 *  the pointer and parameters provided by user.
 *
 *  **Supports MLU220 and MLU270.**
 *
 *  @param[out] param
 *    Output. The returning param descriptor.
 *  @param[in] batch_num
 *    Input. The number of batches.
 *  @param[in] box_num
 *    Input. The number of input box.
 *  @param[in] num_class
 *    Input. The number of classes.
 *  @param[in] im_h
 *    Input. Height of input image of backbone network.
 *  @param[in] im_w
 *    Input. Width of input image of backbone network.
 *  @param[in] score_thresh
 *    Input. Score threshold.
 *  @param[in] nms_thresh
 *    Input. Enumerant IOU threshold used in NMS function.
 *  @param[in] fix8
 *    Input. Precision(fix8=1->INT8; fix8=1->FLOAT/HALF).
 *  @param[in] core_version
 *    Input. Supported core version, including MLU220, and MLU270.
 *  @param[in] scale
 *    Input. The scaling of images.
 *  @retval CNML_STATUS_SUCCESS
 *    The object was set successfully.
 */

cnmlStatus_t cnmlCreatePluginFasterRcnnDetectionOutputOpParam(
    cnmlPluginFasterRcnnDetectionOutputOpParam_t *param,
    int batch_num,
    int box_num,
    int num_class,
    int im_h,
    int im_w,
    bool fix8,
    cnmlCoreVersion_t core_version,
    float scale,
    float nms_thresh,
    float score_thresh);

/*!
 *  @brief A function.
 *
 *  This function frees the PluginFasterRcnnDetectionOutputParam struct,
 *  pointed by the pointer provided by user.
 *
 *  **Supports MLU220 and MLU270.**
 *
 *  @param[in]  param
 *    Input. A pointer to the address of the struct of computation parameters
 *           for PluginFasterRcnnDetectionOutput operator.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Param is a null pointer.
 *    - The pointer content pointed by param is already freed.
 */
cnmlStatus_t cnmlDestroyPluginFasterRcnnDetectionOutputOpParam(
    cnmlPluginFasterRcnnDetectionOutputOpParam_t *param);

/*!
 *  @brief A function.
 *
 *  This function creates PluginFasterRcnnDetectionOutputOp with proper param,
 *  input, and output tensors.
 *
 *  **Supports MLU220 and MLU270.**
 *
 *  @param[out]  op
 *    Output. A pointer to the base operator address.
 *  @param[in]  param
 *    Input. A PluginFasterRcnnDetectionOutput parameter struct pointer.
 *  @param[in]  bbox_pred
 *    Input. An array of four-dimensional cnmlTensors with a shape of
 *           [num_class * 4, box_num, 1, 1](NCHW).
 *           Support FLOAT16 and FLOAT32 dataType currently.
 *  @param[in]  scores_pred
 *    Input. An array of four-dimensional cnmlTensors with a shape of
 *           [num_class, box_num, 1, 1](NCHW).
 *           Support FLOAT16 and FLOAT32 dataType currently.
 *  @param[in]  rois_pred
 *    Input. An array of four-dimensional cnmlTensors with a shape of
 *           [1, box_num, 1, 5](NCHW).
 *           Support FLOAT16 and FLOAT32 dataType currently.
 *  @param[out]  new_box
 *    Output. An array of four-dimensional cnmlTensors with a shape of
 *           [1, box_num * num_class, 1, 6](NCHW).
 *           Support FLOAT16/FLOAT32 dataType currently.
 *  @param[out]  tmp
 *    Output. An array of four-dimensional cnmlTensors with a shape of
 *           [1, 64, 1, 1](NCHW).
 *           Support FLOAT16 and FLOAT32 dataType currently.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *
 */

cnmlStatus_t cnmlCreatePluginFasterRcnnDetectionOutputOp(
    cnmlBaseOp_t *op_ptr,
    cnmlTensor_t bbox_pred,
    cnmlTensor_t scores_pred,
    cnmlTensor_t rois_pred,
    cnmlTensor_t new_box,
    cnmlTensor_t tmp,
    cnmlPluginFasterRcnnDetectionOutputOpParam_t param);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginFasterRcnnDetectionOutputOp on MLU.
 *
 *  **Supports MLU220 and MLU270.**
 *
 *  @param[out]  op
 *    Output. A pointer to the base operator address.
 *  @param[in]  inputs
 *    Input. An array stores the address of all input tensors.
 *  @param[in]  num_inputs
 *    Input. Number of input tensors.
 *  @param[in]  outputs
 *    Input. An array stores the address of all output tensors.
 *  @param[in]  num_outputs
 *    Input. Number of output tensors.
 *  @param[in]  compute_forw_param
 *    Input. A pointer to the struct address, which records runtime degree of data parallelism and
 *    equipment affinity.
 *  @param[in]  queue
 *    Input. A computation queue pointer.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *   .
 */

cnmlStatus_t cnmlComputePluginFasterRcnnDetectionOutputOpForward(
    cnmlBaseOp_t op,
    void *inputs[],
    int num_inputs,
    void *outputs[],
    int num_outputs,
    cnrtInvokeFuncParam_t *compute_forw_param,
    cnrtQueue_t queue);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginFasterRcnnDetectionOutputOp on MLU.
 *
 *  **Supports MLU220 and MLU270.**
 *
 *  @param[in]  param
 *    Input. A PluginFasterRcnnDetectionOutput parameter struct pointer.
 *  @param[in]  cls_boxes
 *    Input. An array stores the address of all cpu input box data.
 *  @param[in]  cls_scores
 *    Input. An array stores the address of all cpu input score data.
 *  @param[in]  ori_rois
 *    Input. An array stores the address of all cpu input rois data.
 *  @param[out]  all_decoded_boxes_cpu
 *    Output. An array stores the address of all cpu output data.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *
 */
cnmlStatus_t cnmlCpuComputePluginFasterRcnnDetectionOutputOpForward(
    cnmlPluginFasterRcnnDetectionOutputOpParam_t param,
    float *all_decoded_boxes_cpu,
    float *cls_boxes,
    float *cls_scores,
    float *ori_rois);
/* ------------------------------------------------- */
/* cnmlPluginFasterRcnnDetectionOutout operation end */
/* ------------------------------------------------- */

/* ===================================== */
/* cnmlPluginPsRoipoolOp operation start */
/* ===================================== */
/*!
 *  @struct cnmlPluginPsRoiPoolOpParam
 *  @brief A struct.
 *
 *  cnmlPluginPsRoiPoolOpParam is a structure describing the "param"
 *  parameter of PsRoiPool operation, used to create conv operation.
 *  cnmlCreatePluginPsRoiPoolOpParam() is used to create an instance of
 *  cnmlPluginPsRoiPoolOpParam_t.
 *  cnmlDestroyPluginPsRoiPoolOpParam() is used to destroy an instance of
 *  cnmlPluginPsRoiPoolOpParam_t.
 *  Detailed information of parameters, see cnmlCreatePluginPsRoiPoolOpParam().
 */
struct cnmlPluginPsRoiPoolOpParam;
/*! ``cnmlPluginPsRoiPoolOpParam_t`` is a pointer to a
    structure (cnmlPluginPsRoiPoolOpParam) holding the description of a PsRoiPool operation param.
*/
typedef cnmlPluginPsRoiPoolOpParam *cnmlPluginPsRoiPoolOpParam_t;

/*!
 *  @brief A function.
 *
 *  This function creates a PluginPsroiPoolOpParam param object with
 *  the pointer and parameters provided by user.
 *
 *  **Supports MLU220 and MLU270.**
 *
 *  @param[out] param
 *    Output. The returning param descriptor.
 *  @param[in] batchNum
 *    Input. The number of batches.
 *  @param[in] input8
 *    Input. Int input tensor is or is not int8.
 *  @param[in] outputdim
 *    Input. The size of outputdim.
 *  @param[in] group_size
 *    Input. The size of group.
 *  @param[in] height
 *    Input. The height of feature map.
 *  @param[in] width
 *    Input. The width of feature map.
 *  @param[in] pooled_height
 *    Input. The height of output feature map.
 *  @param[in] pooled_width
 *    Input. The width of output feature map.
 *  @param[in] nums_rois.
 *    Input  The num of rois.
 *  @param[in] rois_offset.
 *    Input  The len of per roi.
 *  @param[in] spatial_scale.
 *    Input  Spatial scale.
 *  @param[in] core_version
 *    Input. Supported core version, including MLU220 and MLU270.
 *  @retval CNML_STATUS_SUCCESS
 *    The object was set successfully.
 */
cnmlStatus_t cnmlCreatePluginPsRoiPoolOpParam(
    cnmlPluginPsRoiPoolOpParam_t *param,
  int batchNum,
  int int8,
  int outputdim,
  int group_size,
  int height,
  int width,
  int pooled_height,
  int pooled_width,
  int num_rois,
  int rois_offset,
  float spatial_scale,
  cnmlCoreVersion_t core_version);


/*!
 *  @brief A function.
 *
 *  This function frees the PluginPsRoiPoolParam struct, pointed by
 *  the pointer provided by user.
 *
 *  **Supports MLU220 and MLU270.**
 *
 *  @param[in]  param
 *    Input. A pointer to the address of the struct of computation parameters for PluginPsRoiPool operator.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Param is a null pointer.
 *    - The pointer content pointed by param is already freed.
 */
cnmlStatus_t cnmlDestroyPluginPsRoiPoolOpParam(
    cnmlPluginPsRoiPoolOpParam_t *param);

/*!
 *  @brief A function.
 *
 *  This function creates PluginPsRoiPoolOp with proper param,
 *  input, and output tensors.
 *
 *  **Supports MLU220 and MLU270.**
 *
 *  @param[out]  op
 *    Output. A pointer to the base operator address.
 *  @param[in]  param
 *    Input. A PluginPsRoiPool parameter struct pointer.
 *  @param[in]  psroipool_input_tensors
 *    Input. An array of four-dimensional cnmlTensors with a shape of
 *           [batchNum, outputdim * group_size * group_size, height, width](NCHW).The other
 *           four-dimensional cnmlTensors width a shape of [batch_num,num_rois,rois_offset,1]
 *           Support only FLOAT16 dataType currently.
 *  @param[in]  outputs
 *    Input. An array of four-dimensional cnmlTensors with a shape of
 *           [batchNum * num_rois, outputdim, pooled_height, pooled_width](NCHW).
 *           Support only FLOAT16 dataType currently.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - base op pointer is nullptr.
 *    - param is nullptr or not initialized.
 *    - input and output tensor desps is nullptr or inconsistent with param.
 */
cnmlStatus_t cnmlCreatePluginPsRoiPoolOp(
    cnmlBaseOp_t *op,
    cnmlPluginPsRoiPoolOpParam_t param,
    cnmlTensor_t *psroipool_input_tensors,
    cnmlTensor_t *psroipool_output_tensors);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginPsRoiPoolOp on MLU.
 *
 *  **Supports MLU220 and MLU270.**
 *
 *  @param[out]  op
 *    Output. A pointer to the base operator address.
 *  @param[in]  inputs
 *    Input. An array stores the address of all input tensors.
 *  @param[in]  num_inputs
 *    Input. Number of input tensors
 *  @param[in]  outputs
 *    Input. An array stores the address of all output tensors.
 *  @param[in]  num_outputs
 *    Input. Number of output tensors.
 *  @param[in]  compute_forw_param
 *    Input. A pointer to the struct address, which records runtime degree of data parallelism and
 *    equipment affinity.
 *  @param[in]  queue
 *    Input. A computation queue pointer.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - op is nullptr or not initialized.
 *    - input and output tensor desps is nullptr or inconsistent with param.
 */
cnmlStatus_t cnmlComputePluginPsroipoolOpForward(
    cnmlBaseOp_t op,
    void *input[],
    int num_inputs,
    void *output[],
    int num_outputs,
    cnrtInvokeFuncParam_t *compute_forw_param,
    cnrtQueue_t queue);

/* --------------------------------- */
/* cnmlPluginPsRoiPool operation end */
/* --------------------------------- */

/* =================================== */
/* cnmlPluginYuv2RgbOp operation start */
/* =================================== */
/*!
 *  @brief A function.
 *
 *  This function creates a PluginYuvToRgbOp param object with
 *  the pointer and parameters provided by user.
 *
 *  **Supports MLU220 and MLU270.**
 *
 *  @param[out] param
 *    Output. The returning param descriptor.
 *  @param[in] s_row
 *    Input. Rows of input image of backbone network.
 *  @param[in] s_col
 *    Input. Cols of input image of backbone network.
 *  @param[in] mode
 *    Input. The model for convert.
 *  @param[in] core_version
 *    Input. Supported core version, including MLU220 and MLU270.
 *  @retval CNML_STATUS_SUCCESS
 *    The object was set successfully.
 */
cnmlStatus_t cnmlCreatePluginYuvToRgbOpParam(
    cnmlPluginResizeAndColorCvtParam_t *param,
    int s_row,
    int s_col,
    ioParams mode,
    cnmlCoreVersion_t core_version,
    bool is_variable = false);

/*!
 *  @brief A function.
 *
 *  This function creates a PluginYuvToRgbOp param object with
 *  the pointer and parameters provided by user.
 *
 *  **Supports MLU220 and MLU270.**
 *
 *  @param[out] param
 *    Output. The returning param descriptor.
 *  @param[in] s_row
 *    Input. Rows of input image of backbone network.
 *  @param[in] s_col
 *    Input. Cols of input image of backbone network.
 *  @param[in] batch_num
 *    Input. The batch number of input tensor.
 *  @param[in] mode
 *    Input. The model for convert.
 *  @param[in] core_version
 *    Input. Supported core version, including MLU220 and MLU270.
 *  @retval CNML_STATUS_SUCCESS
 *    The object was set successfully.
 */
cnmlStatus_t cnmlCreatePluginYuvToRgbOpParam_V2(
    cnmlPluginResizeAndColorCvtParam_t *param,
    int s_row,
    int s_col,
    int batch_num,
    ioParams mode,
    cnmlCoreVersion_t core_version,
    bool is_variable = false);

/*!
 *  @brief A function.
 *
 *  This function creates a PluginYuvToRgbOp param object with
 *  the pointer and parameters provided by user.
 *
 *  **Supports MLU220/MLU270**
 *
 *  @param[out] param
 *    Output. The returning param descriptor.
 *  @param[in] s_row
 *    Input. Rows of input image of backbone network.
 *  @param[in] s_col
 *    Input. Cols of input image of backbone network.
 *  @param[in] batch_num
 *    Input. The batch number of input tensor.
 *  @param[in] mode
 *    Input. The model for convert.
 *  @param[in] core_version
 *    Input. Supported core version, including 220/270.
 *  @param[in] input_series
 *    Input. a flag to judge whether y and uv input address is continuation.
 *  @retval CNML_STATUS_SUCCESS
 *    The object was set successfully.
 */
cnmlStatus_t cnmlCreatePluginYuvToRgbOpParam_V3(
    cnmlPluginResizeAndColorCvtParam_t *param,
    int s_row,
    int s_col,
    int batch_num,
    ioParams mode,
    cnmlCoreVersion_t core_version,
    bool input_series,
    bool is_variable);

/*!
 *  @brief A function.
 *
 *  This function frees the PluginYuvToRgbOp struct, pointed by
 *  the pointer provided by user.
 *
 *  **Supports MLU220 and MLU270.**
 *
 *  @param[in]  param
 *    Input. A pointer to the address of the struct of computation parameters for PluginYuvToRgbOp operator.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Param is a null pointer.
 *    - The pointer content pointed by param is already freed.
 */
cnmlStatus_t cnmlDestroyPluginYuvToRgbOpParam(cnmlPluginResizeAndColorCvtParam_t* param);

/*!
 *  @brief A function.
 *
 *  This function creates PluginYuvToRgbOp with proper param,
 *  input, and output tensors.
 *
 *  **Supports MLU220 and MLU270.**
 *
 *  @param[out]  op
 *    Output. A pointer to the base operator address.
 *  @param[in]  param
 *    Input. A PluginYuvToRgbOp parameter struct pointer.
 *  @param[in]  yuv2rgb_input_tensors
 *    Input. An array of four-dimensional cnmlTensors with a shape of
 *           [batchNum, channelIn, rows, cols](NCHW).
 *           Support FLOAT16 or UINT8 dataType currently.
 *  @param[in]  outputs
 *    Input. An array of four-dimensional cnmlTensors with a shape of
 *           [batchNum, channel, rows, cols](NCHW).
 *           Support FLOAT16 or UINT8 dataType currently.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *
 */
cnmlStatus_t cnmlCreatePluginYuvToRgbOp(
    cnmlBaseOp_t *op,
    cnmlPluginResizeAndColorCvtParam_t param,
    cnmlTensor_t *cnml_input_ptr,
    cnmlTensor_t *cnml_output_ptr);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginYuvToRgbOp on MLU.
 *
 *  **Supports MLU220 and MLU270.**
 *
 *  @param[out]  op
 *    Output. A pointer to the base operator address.
 *  @param[in]  param
 *    Input. A PluginYuvToRgbOp parameter struct pointer.
 *  @param[in]  inputs_addrs
 *    Input. An array stores the address of all input tensors.
 *  @param[in]  outputs_addrs
 *    Input. An array stores the address of all output tensors.
 *  @param[in]  compute_forw_param
 *    Input. A pointer to the struct address, which records runtime degree of data parallelism and
 *    equipment affinity.
 *  @param[in]  queue
 *    Input. A computation queue pointer.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *
 */
cnmlStatus_t cnmlComputePluginYuvToRgbOpForward(
    cnmlBaseOp_t op,
    cnmlPluginResizeAndColorCvtParam_t param,
    void **input_addrs,
    void **output_addrs,
    cnrtInvokeFuncParam_t compute_forw_param,
    cnrtQueue_t queue);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginYuvToRgbOp on MLU.
 *
 *  **Supports MLU220 and MLU270.**
 *
 *  @param[out]  op
 *    Output. A pointer to the base operator address.
 *  @param[in]  param
 *    Input. A PluginYuvToRgbOp parameter struct pointer.
 *  @param[in] cnml_input_ptr
 *    Input. An array of four-dimensional cnmlTensors for src image
 *  @param[in]  input_addrs
 *    Input. An array stores the address of all input tensors
 *  @param[in] cnml_output_ptr
 *    Input. An array of four-dimensional cnmlTensors for dst image
 *  @param[in]  output_addrs
 *    Input. An array stores the address of all output tensors
 *  @param[in]  queue
 *    Input. A computation queue pointer.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Base op is nullptr.
 *    - Input / output addrs is nullptr.
 *    - Input / output nums are inconsistent.
 *    - Task type is invalid at runtime.
 */
cnmlStatus_t cnmlComputePluginYuvToRgbOpForward_V2(cnmlBaseOp_t op,
    cnmlPluginResizeAndColorCvtParam_t param,
    cnmlTensor_t *cnml_input_ptr,
    void **input_addrs,
    cnmlTensor_t *cnml_output_ptr,
    void **output_addrs,
    cnrtQueue_t queue);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginYuvToRgbOp on MLU.
 *
 *  **Supports MLU220/MLU270**
 *
 *  @param[out] dst
 *    Output. The result of yuv2rgb on cpu.
 *  @param[in] src
 *    Input. The YUV image data.
 *  @param[in] s_row
 *    Input. Rows of input image of backbone network.
 *  @param[in] s_col
 *    Input. Cols of input image of backbone network.
 *  @param[in] mode
 *    Input. The model for convert.
 */
void cnmlCpuComputePluginYuvToRgbOpForward(
    unsigned char* dst,
    unsigned char* src,
    int s_row,
    int s_col,
    ioParams mode);
/* --------------------------------- */
/* cnmlPluginYuv2RgbOp operation end */
/* --------------------------------- */

/* ======================================= */
/* cnmlPluginCropAndResize operation start */
/* ======================================= */
/*!
 *  @brief A function.
 *
 *  This function creates a PluginCropAndResizeOp param object with a pointer
 *  and "user params" provided.
 *
 *  @param[out] param
 *    Output. The returning param descriptor.
 *  @param[in] s_row
 *    Input. The row number of src image.
 *  @param[in] s_col
 *    Input. The col number of src image.
 *  @param[in] d_row
 *    Input. The row number of dst image.
 *  @param[in] d_col
 *    Input. The col number of dst image.
 *  @param[in] mode
 *    Input. The color and datatype conversion mode.
 *  @param[in] batchNum
 *    Input. The number of batch of input images. This op regards one image as
 *           one batch.
 *  @param[in] core_version
 *    Input[in]. The hardware core_version.
 *  @retval CNML_STATUS_SUCCESS
 *    The object was set successfully.
 */
cnmlStatus_t cnmlCreatePluginCropAndResizeOpParam(
  cnmlPluginResizeAndColorCvtParam_t* param,
  int s_row,
  int s_col,
  int d_row,
  int d_col,
  ioParams mode,
  int batchNum,
  cnmlCoreVersion_t core_version);

/*!
 *  @brief A function.
 *
 *  This function creates a PluginCropAndResizeOp param object with a pointer
 *  and "user params" provided.
 *
 *  @param[out] param
 *    Output. The returning param descriptor.
 *  @param[in] s_row
 *    Input. The row number of src image.
 *  @param[in] s_col
 *    Input. The col number of src image.
 *  @param[in] d_row
 *    Input. The row number of dst image.
 *  @param[in] d_col
 *    Input. The col number of dst image.
 *  @param[in] mode
 *    Input. The color and datatype conversion mode.
 *  @param[in] batchNum
 *    Input. The number of batch of input images. This op regards one image as
 *           one batch.
 *  @param[in] keepAspectRatio
 *    Input. The flag indicate if this op keeps aspect ratio for resized images
 *  @param[in] core_version
 *    Input[in]. The hardware core_version.
 *  @retval CNML_STATUS_SUCCESS
 *    The object was set successfully.
 */
cnmlStatus_t cnmlCreatePluginCropAndResizeOpParam_V2(
  cnmlPluginResizeAndColorCvtParam_t* param,
  int s_row,
  int s_col,
  int d_row,
  int d_col,
  ioParams mode,
  int batchNum,
  int keepAspectRatio,
  cnmlCoreVersion_t core_version);

/*!
 *  @brief A function.
 *
 *  This function frees the PluginCropAndResizeOpParam struct, pointed by
 *  the pointer provided by user.
 *
 *  @param[in]  param
 *    Input. A pointer to the address of the struct of computation parameters
 *    for PluginCropAndResize operator.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Param is a null pointer.
 *    - The pointer content pointed by param is already freed.
 */
cnmlStatus_t cnmlDestroyPluginCropAndResizeOpParam(
    cnmlPluginResizeAndColorCvtParam_t *param);

/*!
 *  @brief A function.
 *
 *  This function creates PluginCropAndResizeOp with proper param,
 *  input, and output tensors. The current implementation is based on the
 *  traditional bi-linear interpolation method on OpenCV.
 *
 *  **Supports MLU220 and MLU270.**
 *
 *  @param[out] op
 *    Output. A pointer to the base operator address.
 *  @param[in] param
 *    Input. A PluginResizeAndColorCvt parameter struct pointer.
 *  @param[in] dst
 *    Input. A four-dimensional tensor for dst image.
 *  @param[in] src
 *    Input. A four-dimensional tensor for src image.
 *  @param[in] cropParams
 *    Input. A four-dimensional tensor for all cropParams, i.e. roiParams.
 *  @param[in] roiNums
 *    Input. A four-dimensional tensor for the number of rois of each images.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - param is not consistant with tensors.
 *    - shapes of cropParams and roiNums are not consistent.
 */
cnmlStatus_t cnmlCreatePluginCropAndResizeOp(
    cnmlBaseOp_t *op,
    cnmlPluginResizeAndColorCvtParam_t param,
    cnmlTensor_t dst,
    cnmlTensor_t src,
    cnmlTensor_t cropParams,
    cnmlTensor_t roiNums);

/*!
 *  @brief A function.
 *
 *  This function creates PluginCropAndResizeOp with proper param,
 *  input, and output tensors. The current implementation is based on the
 *  traditional bi-linear interpolation method on OpenCV.
 *
 *  **Supports MLU220 and MLU270.**
 *
 *  @param[out] op
 *    Output. A pointer to the base operator address.
 *  @param[in] param
 *    Input. A PluginResizeAndColorCvt parameter struct pointer.
 *  @param[in] output_tensors
 *    Input. A vector of output tensors. The size of vector must be 1.
 *  @param[in] input_tensors
 *    Input. A vector of input tensors. The size of vector must be 4. The order of
 *           input tensors must be:
 *            - src_images: A UINT8-tensor with shape [N H W 4] for input data.
 *            - crop_parameters: A UINT32-tensor with shape [N 1 1 4] for roi_xywh.
 *            - roi_numbers: A UINT32-tensor with shape[N 1 1 1] for roi number for each batch.
 *            - pad_values: A UINT32-tensor with shape[1 1 1 4] for padding value.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - param is not consistant with tensors.
 *    - numbers of input/output tensors are wrong.
 */
cnmlStatus_t cnmlCreatePluginCropAndResizeOp_V2(
    cnmlBaseOp_t *op,
    cnmlPluginResizeAndColorCvtParam_t param,
    cnmlTensor_t *output_tensors,
    cnmlTensor_t *input_tensors);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginCropAndResizeOp on MLU.
 *
 *  **Supports MLU220 and MLU270.**
 *
 *  @param[in] op
 *    Output. A pointer to the base operator address.
 *  @param[in] src_mlu_tensor
 *    Input. A four-dimensional tensor for src image.
 *  @param[in] src_addr
 *    Input. Address of input tensor.
 *  @param[in] cropParams_mlu_tensor
 *    Input. A four-dimensional tensor for cropParams.
 *  @param[in] cropParams_addr
 *    Input. Address of cropParams tensor.
 *  @param[in] roiNums_mlu_tensor
 *    Input. A four-dimensional tensor for roiNums.
 *  @param[in] roiNums_addr
 *    Input. Address of roiNums tensor.
 *  @param[in] dst_tensor
 *    Input. A four-dimensional tensor for dst image.
 *  @param[Out] dst_addr
 *    Output. Address of output tensor.
 *  @param[in] compute_forw_param
 *    Input. A pointer to the struct address, which records runtime degree of
 *    data parallelism and equipment affinity.
 *  @param[in] queue
 *    Input. A computation queue pointer.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Base op is nullptr.
 *    - Input and output addrs is nullptr.
 *    - Task type is invalid at runtime.
 */
cnmlStatus_t cnmlComputePluginCropAndResizeOpForward(
    cnmlBaseOp_t op,
    cnmlTensor_t src_mlu_tensor,
    void *src_addr,
    cnmlTensor_t cropParams_mlu_tensor,
    void *cropParams_addr,
    cnmlTensor_t roiNums_mlu_tensor,
    void *roiNums_addr,
    cnmlTensor_t dst_mlu_tensor,
    void *dst_addr,
    cnrtInvokeFuncParam_t compute_forw_param,
    cnrtQueue_t queue);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginCropAndResizeOp on MLU.
 *
 *  **Supports MLU220 and MLU270.**
 *
 *  @param[in] op
 *    Output. A pointer to the base operator address.
 *  @param[in] param
 *    Output. A PluginResizeAndColorCvtParam struct pointer.
 *  @param[in] input_tensors
 *    Input. An array of four-dimensional tensors for inputs.
 *  @param[in] input_addrs
 *    Input. An array of addresses of input tensors.
 *  @param[in] output_tensors
 *    Input. An array of four-dimensional tensors for outputs.
 *  @param[Out] output_addrs
 *    Output. An array of addresses of output tensors.
 *  @param[in] queue
 *    Input. A computation queue pointer.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Base op is nullptr.
 *    - Input and output addrs is nullptr.
 *    - Task type is invalid at runtime.
 */
cnmlStatus_t cnmlComputePluginCropAndResizeOpForward_V2(
    cnmlBaseOp_t op,
    cnmlPluginResizeAndColorCvtParam_t param,
    cnmlTensor_t *input_tensors,
    void **input_addrs,
    cnmlTensor_t *output_tensors,
    void **output_addrs,
    cnrtQueue_t queue);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginCropAndResizeOp on CPU.
 *
 *  @param[out] dst
 *    Output. The pointer of dst image.
 *  @param[in] src
 *    Input. The pointer of src image.
 *  @param[in] s_row
 *    Input. The row number of src image.
 *  @param[in] s_col
 *    Input. The col number of src image.
 *  @param[in] d_row
 *    Input. The row number of dst image.
 *  @param[in] d_col
 *    Input. The col number of dst image.
 *  @param[in] roi_x
 *    Input. The x-coordinate of top-left corner of roi.
 *  @param[in] roi_y
 *    Input. The y-coordinate of top-left corner of roi.
 *  @param[in] roi_w
 *    Input. The width of roi.
 *  @param[in] roi_h
 *    Input. The height of roi.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Input and output pointer is nullptr.
 *    - Param is not consistent with input and output.
 */
cnmlStatus_t cnmlCpuComputePluginCropAndResizeOpForward(
    unsigned char* dst,
    unsigned char* src,
    int s_row,
    int s_col,
    int d_row,
    int d_col,
    int roi_x,
    int roi_y,
    int roi_w,
    int roi_h);
/* ------------------------------------- */
/* cnmlPluginCropAndResize operation end */
/* ------------------------------------- */

/* ============================================== */
/* cnmlPluginCropFeatureAndResize operation start */
/* ============================================== */
/*!
 *  @brief A function.
 *
 *  This function creates a PluginCropFeatureAndResizeOp param object with a pointer
 *  and "user params" provided.
 *
 *  @param[out] param
 *    Output. The returning param descriptor.
 *  @param[in] s_row
 *    Input. The row number of src image.
 *  @param[in] s_col
 *    Input. The col number of src image.
 *  @param[in] d_row
 *    Input. The row number of dst image.
 *  @param[in] d_col
 *    Input. The col number of dst image.
 *  @param[in] batchNum
 *    Input. The number of batch of input images. This op regards one image as
 *           one batch.
 *  @param[in] depth
 *    Input. The depth or channel of src image.
 *  @param[in] box_number
 *    Input. Detect the number of bbox.
 *  @param[in] pad_size
 *    Input. pad_size.
 *  @param[in] core_version
 *    Input[in]. The hardware core_version.
 *  @retval CNML_STATUS_SUCCESS
 *    The object was set successfully.
 */
cnmlStatus_t cnmlCreatePluginCropFeatureAndResizeOpParam(
  cnmlPluginResizeAndColorCvtParam_t* param,
  int s_row,
  int s_col,
  int d_row,
  int d_col,
  int batchNum,
  int depth,
  int box_number,
  int pad_size,
  cnmlCoreVersion_t core_version);

/*!
 *  @brief A function.
 *
 *  This function creates a PluginCropFeatureAndResizeOp param object with a pointer
 *  and "user params" provided.
 *
 *  @param[out] param
 *    Output. The returning param descriptor.
 *  @param[in] s_row
 *    Input. The row number of src image.
 *  @param[in] s_col
 *    Input. The col number of src image.
 *  @param[in] d_row
 *    Input. The row number of dst image.
 *  @param[in] d_col
 *    Input. The col number of dst image.
 *  @param[in] batchNum
 *    Input. The number of batch of input images. This op regards one image as
 *           one batch.
 *  @param[in] depth
 *    Input. The depth or channel of src image.
 *  @param[in] box_number
 *    Input. detect number of bbox.
 *  @param[in] pad_size
 *    Input. pad_size.
 *  @param[in] extrapolation_value
 *    Input. Value used for extrapolation, when applicable.
 *  @param[in] core_version
 *    Input[in]. The hardware core_version.
 *  @retval CNML_STATUS_SUCCESS
 *    The object was set successfully.
 */
cnmlStatus_t cnmlCreatePluginCropFeatureAndResizeOpParam_V2(
  cnmlPluginResizeAndColorCvtParam_t* param,
  int s_row,
  int s_col,
  int d_row,
  int d_col,
  int batchNum,
  int depth,
  int box_number,
  int pad_size,
  cnmlCoreVersion_t core_version,
  float extrapolation_value);

/*!
 *  @brief A function.
 *
 *  This function frees the PluginCropFeatureAndResizeOpParam struct, pointed by
 *  the pointer provided by user.
 *
 *  @param[in]  param
 *    Input. A pointer to the address of the struct of computation parameters
 *    for PluginCropFeatureAndResize operator.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Param is a null pointer.
 *    - The pointer content pointed by param is already freed.
 */
cnmlStatus_t cnmlDestroyPluginCropFeatureAndResizeOpParam(
    cnmlPluginResizeAndColorCvtParam_t* param);

/*!
 *  @brief A function.
 *
 *  This function creates PluginCropFeatureAndResizeOp with proper param,
 *  input, and output tensors. The current implementation is based on the
 *  traditional bi-linear interpolation method on OpenCV.
 *
 *  **Supports MLU220 and MLU270.**
 *
 *  @param[out] op
 *    Output. A pointer to the base operator address.
 *  @param[in] param
 *    Input. A PluginResizeAndColorCvt parameter struct pointer.
 *  @param[in] input_cnml_tensors
 *    Input. A four-dimensional tensor for dst image.
 *  @param[in] output_cnml_tensors
 *    Input. A four-dimensional tensor for src image.
 *  @param[in] extrapolation_value
 *    Input. Value used for extrapolation, when applicable.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - param is not consistant with tensors.
 *    - shapes of cropParams and roiNums are not consistent.
 */
cnmlStatus_t cnmlCreatePluginCropFeatureAndResizeOp(
    cnmlBaseOp_t* op,
    cnmlPluginResizeAndColorCvtParam_t* param,
    cnmlTensor_t* input_cnml_tensors, // src
    cnmlTensor_t* output_cnml_tensors);  // dst

/*!
 *  @brief A function.
 *
 *  This function forwards PluginResizeCropFeatureAndResizeOp on MLU.
 *
 *  **Supports MLU220 and MLU270.**
 *
 *  @param[in] op
 *    Output. A pointer to the base operator address.
 *  @param[in] input_addr
 *    Input. Address of input tensor.
 *  @param[out] output_addr
 *    Output. Address of output tensor.
 *  @param[in] compute_forw_param
 *    Input. A pointer to the struct address, which records runtime degree of
 *    data parallelism and equipment affinity.
 *  @param[in] queue
 *    Input. A computation queue pointer.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Base op is nullptr.
 *    - Input and output addrs is nullptr.
 *    - Task type is invalid at runtime.
 */
cnmlStatus_t cnmlComputePluginCropFeatureAndResizeOpForward(
    cnmlBaseOp_t op,
    void* input_addr[],
    void* output_addr[],
    cnrtInvokeFuncParam_t compute_forw_param,
    cnrtQueue_t queue);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginCropFeatureAndResizeOp on CPU.
 *
 *  @param[out] dst
 *    Output. The pointer of dst image.
 *  @param[in] src
 *    Input. The pointer of src image.
 *  @param[in] boxes
 *    Input. The pointer to detect bbox.
 *  @param[in] box_index
 *    Input. The pointer to index of bbox.
 *  @param[in] new_box
 *    Input. The pointer to output.
 *  @param[in] batchNum
 *    Input. The batch size.
 *  @param[in] depth
 *    Input. The channel of input feature.
 *  @param[in] image_height
 *    Input. The height of input feature.
 *  @param[in] image_width
 *    Input. The width of input feature.
 *  @param[in] crop_height
 *    Input. The height of the resizing output.
 *  @param[in] crop_width
 *    Input. The width of the resizing output.
 *  @param[in] box_number
 *    Input. The number of detect bbox.
 *  @param[in] extrapolation_value
 *    Input. Value used for extrapolation, when applicable.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - The input and output pointer is nullptr.
 *    - Param is not consistent with input and output.
 */
cnmlStatus_t cnmlCpuComputePluginCropFeatureAndResizeOpForward(
    float* src,
    float* boxes,
    float* box_index,
    float* new_box,
    int batchNum,
    int depth,
    int image_height,
    int image_width,
    int crop_height,
    int crop_width,
    int box_number,
    float extrapolation_value);
/* -------------------------------------------- */
/* cnmlPluginCropFeatureAndResize operation end */
/* -------------------------------------------- */

/* =========================================== */
/* cnmlPluginNonMaxSuppression operation start */
/* =========================================== */
/*!
 *  @struct cnmlPluginNonMaxSuppressionOpParam
 *  @brief A struct.
 *
 *  cnmlPluginNonMaxSuppressionOpParam is a structure describing the "param"
 *  parameter of NonMaxSuppression operation.
 *  cnmlCreatePluginNonMaxSuppressionOpParam() is used to create an instance of
 *  cnmlPluginNonMaxSuppressionOpParam_t.
 *  cnmlDestoryPluginNonMaxSuppressionOpParam() is used to destory an instance of
 *  cnmlPluginNonMaxSuppressionOpParam_t.
 *  Detailed information of parameters, see cnmlCreatePluginNonMaxSuppressionOpParam().
 */
struct cnmlPluginNonMaxSuppressionOpParam;
/*! ``cnmlPluginNonMaxSuppressionOpParam_t`` is a pointer to a
    structure (cnmlPluginNonMaxSuppressionOpParam) holding the description of a NonMaxSuppression operation param.
*/
typedef cnmlPluginNonMaxSuppressionOpParam
*cnmlPluginNonMaxSuppressionOpParam_t;

/*!
 *  @brief A function.
 *
 *  This function creates a PluginNonMaxSuppressionOp param object with a pointer
 *  and "user params" provided.
 *
 *  @param[out] param
 *    Output. The returning param descriptor.
 *  @param[in] len
 *    Input. The number of input boxes.
 *  @param[in] max_num
 *    Input. The max number of output boxes.
 *  @param[in] iou_threshold
 *    Input. The threshold of iou to do nms.
 *  @param[in] score_threshold
 *    Input. The threshold of score to do nms.
 *  @param[in] core_version
 *    Input[in]. The hardware core_version.
 *  @retval CNML_STATUS_SUCCESS
 *    The object was set successfully.
 */
cnmlStatus_t cnmlCreatePluginNonMaxSuppressionOpParam(
  cnmlPluginNonMaxSuppressionOpParam_t *param,
  int len,
  int max_num,
  float iou_threshold,
  float score_threshold,
  cnmlCoreVersion_t core_version=CNML_MLU270);

/*!
 *  @brief A function.
 *
 *  This function frees the PluginNonMaxSuppressionOpParam struct, pointed by
 *  the pointer provided by user.
 *
 *  @param[in]  param
 *    Input. A pointer to the address of the struct of computation parameters
 *    for PluginNonMaxSuppression operator.
 *  @param[in]  static_num
 *    Input. Number of static tensors.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Param is a null pointer.
 *    - The pointer content pointed by param is already freed.
 */
cnmlStatus_t cnmlDestroyPluginNonMaxSuppressionOpParam(
    cnmlPluginNonMaxSuppressionOpParam_t *param,
    int static_num);

/*!
 *  @brief A function.
 *
 *  This function creates PluginNonMaxSuppressionOp with proper param,
 *  input, and output tensors.
 *
 *  **Supports Tensorflow on MLU270.**
 *
 *  @param[out] op
 *    Output. A pointer to the base operator address.
 *  @param[in] param
 *    Input. A PluginNonMaxSuppression parameter struct pointer.
 *  @param[in] nms_input_tensors
 *    Input. This pointer contains two array of four-dimensional cnmlTensors,
 *           first tensor's shape is [4, len, 1, 1], second tensor's shape is [1, len, 1, 1].
 *  @param[in] input_num
 *    Input. Number of input tensors.
 *  @param[out] nms_output_tensors
 *    Output. This pointer contains an array of four-dimensional cnmlTensor,
 *           the tensor's shape is [1, max_num, 1, 1].
 *  @param[in] output_num
 *    Input. Number of output tensors.
 *  @param[in] static_num
 *    Input. Number of static tensors.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - param is not consistant with tensors.
 *    - shapes of cropParams and roiNums are not consistent.
 */
cnmlStatus_t cnmlCreatePluginNonMaxSuppressionOp(
    cnmlBaseOp_t *op,
    cnmlPluginNonMaxSuppressionOpParam_t param,
    cnmlTensor_t *nms_input_tensors,
    int input_num,
    cnmlTensor_t *nms_output_tensors,
    int output_num,
    int static_num);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginNonMaxSuppressionOp on MLU.
 *
 *  **Supports MLU270.**
 *
 *  @param[in] op
 *    Output. A pointer to the base operator address.
 *  @param[in] input_tensors
 *    Input. Void.
 *  @param[in] inputs
 *    Input. An array stores the address of all input tensors.
 *  @param[in] num_inputs
 *    Input. Number of input tensors.
 *  @param[out] output_tensors
 *    Output. Void.
 *  @param[in] outputs
 *    Input. An array stores the address of all output tensors.
 *  @param[in] num_outputs
 *    Input. Number of output tensors.
 *  @param[in] queue
 *    Input. A computation queue pointer.
 *  @param[in] extra
 *    Input. A pointer contains other input params.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Base op is nullptr.
 *    - Input and output addrs is nullptr.
 *    - Task type is invalid at runtime.
 */
cnmlStatus_t cnmlComputePluginNonMaxSuppressionOpForward(
    cnmlBaseOp_t op,
    cnmlTensor_t input_tensors[],
    void *inputs[],
    int num_inputs,
    cnmlTensor_t output_tensors[],
    void *outputs[],
    int num_outputs,
    cnrtQueue_t queue,
    void *extra);

/* ----------------------------------------- */
/* cnmlPluginNonMaxSuppression operation end */
/* ----------------------------------------- */

/* ============================= */
/* cnmlPluginNms operation start */
/* ============================= */
/*!
 *  @struct cnmlPluginNmsOpParam
 *  @brief A struct.
 *
 *  cnmlPluginNmsOpParam is a structure describing the "param"
 *  parameter of cnmlPluginNmsOpParam operation.
 *  cnmlCreatePluginNmsOpParam() is used to create an instance of
 *  cnmlPluginNmsParam_t.
 *  cnmlDestroyPluginNmsOpParam() is used to destroy an instance
 *  of cnmlPluginNmsParam_t.
 *  Detailed information of parameters, see cnmlCreatePluginNmsOp().
 */
struct cnmlPluginNmsOpParam;
/*! ``cnmlPluginNmsOpParam_t`` is a pointer to a
    structure (cnmlPluginNmsOpParam) holding the description of a NmsOp operation param.
*/
typedef cnmlPluginNmsOpParam *cnmlPluginNmsOpParam_t;

/*!
 *  @brief A function.
 *
 *  This function creates a PluginNmsOp param object with
 *  the pointer and parameters provided by user. This implementation is based
 *  on the official MXNet website.
 *
 *  **Supports MXNet on MLU270.**
 *
 *  @param[out] param
 *    Output. The returning param descriptor.
 *  @param[in] n
 *    Input. The number of batches.
 *  @param[in] channels
 *    Input. The number of boxes.
 *  @param[in] h
 *    Input. The number of items.
 *  @param[float] overlap_Thresh
 *    Input. Overlapping(IoU) threshold to suppress object with smaller score.
 *  @param[float] valid_Thresh
 *    Input. Filter input boxes to those whose scores greater than valid_thresh.
 *  @param[in] topk
 *    Input. Apply nms to topk boxes with descending scores.
 *  @param[in] coord_start
 *    Input. Start index of the consecutive 4 coordinates.
 *  @param[in] score_index
 *    Input. Index of the scores/confidence of boxes.
 *  @param[in] id_index.
 *    Input. Index of the class categories.
 *  @param[in] background_id
 *    Input. The id of background.
 *  @param[bool] force_suppress
 *    Input. if set 0 and id_index is provided, nms will only apply to boxes belongs to the same category.
 *  @param[in] in_format
 *    Input. The input box encoding type.1 indicate "center" and 0 indicate "corner".
 *  @param[in] out_format
 *    Input. The output box encoding type.1 indicate "center" and 0 indicate "corner".
 *  @param[in] dtype_flag
 *    Input. The data type of input. 0:float32, 1:float64, 2:float16
 *  @param[cnmlCoreVersion_t] coreVersion
 *    Input. The core version of MLU.
 *  @retval CNML_STATUS_SUCCESS
 *    The object was set successfully.
 */
cnmlStatus_t cnmlCreatePluginNmsOpParam(
  cnmlPluginNmsOpParam_t *param,
  int n,
  int channels,
  int height,
  float overlap_Thresh,
  float valid_Thresh,
  int topk,
  int coord_start,
  int score_index,
  int id_index,
  int background_id,
  bool force_suppress,
  int in_format,
  int out_format,
  int dtype_flag=2,
  cnmlCoreVersion_t coreVersion=CNML_MLU270);

/*!
 *  @brief A function.
 *
 *  This function frees the PluginNmsOpParam struct, pointed by
 *  the pointer provided by user.
 *
 *  @param[in]  param
 *    Input. A pointer to the address of the struct of computation parameters
 *    for PluginNms operator.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Param is a null pointer.
 *    - The pointer content pointed by param is already freed.
 */
cnmlStatus_t cnmlDestroyPluginNmsOpParam(
    cnmlPluginNmsOpParam_t *param);

/*!
 *  @brief A function.
 *
 *  This function creates PluginNmsOp with proper param,
 *  input, and output tensors. The current implementation is based on the
 *  official MXNet NMS op.
 *
 *  **Supports MXNet on MLU270.**
 *
 *  @param[out]  op
 *    Output. A pointer to the base operator address.
 *  @param[in]  param
 *    Input. A PluginNmsOp parameter struct pointer.
 *  @param[in]  nms_input_tensors
 *    Input. An array of four-dimensional cnmlTensors with a shape of
 *           [batchNum, inputC, inputH, inputW](NCHW).
 *           Support only FLOAT16 dataType currently.
 *  @param[in]  outputs
 *    Output. An array of four-dimensional cnmlTensors with a shape of
 *           [batchsize, anchor_num, 4, 1](NCHW).
 *           Support only FLOAT16 dataType currently.
 *           The size is the length of result.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Base op pointer is nullptr.
 *    - Param is nullptr or not initialized.
 *    - Input and output tensor desps is nullptr or inconsistent with param.
 */
cnmlStatus_t cnmlCreatePluginNmsOp(
  cnmlBaseOp_t *op,
  cnmlPluginNmsOpParam_t param,
  cnmlTensor_t *nms_input_tensors,
  cnmlTensor_t *nms_output_tensors
);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginNmsOp on MLU.
 *
 *  **Supports MLU270.**
 *
 *  @param[in] op
 *    Output. A pointer to the base operator address.
 *  @param[in] input_tensors
 *    Input. Void.
 *  @param[in] inputs
 *    Input. An array stores the address of all input tensors.
 *  @param[in] num_inputs
 *    Input. Number of input tensors.
 *  @param[out] output_tensors
 *    Output. Void.
 *  @param[in] outputs
 *    Input. An array stores the address of all output tensors.
 *  @param[in] num_outputs
 *    Input. Number of output tensors.
 *  @param[in] queue
 *    Input. A computation queue pointer.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Base op is nullptr.
 *    - Input and output addrs is nullptr.
 *    - Task type is invalid at runtime.
 */
cnmlStatus_t cnmlComputePluginNmsOpForward(
  cnmlBaseOp_t op,
  void **inputs,
  int input_num,
  void **outputs,
  int output_num,
  cnrtQueue_t queue
);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginNmsOp on CPU.
 *
 *  **Supports MLU270.**
 *
 *  @param[in]  param
 *    Input. A PluginProposal parameter struct pointer.
 *  @param[in]  inputs
 *    Input. Adress of cpu input data.
 *  @param[out]  outputs
 *    Output. Adress of cpu output data.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Param is nullptr or inconsistent.
 *    - Input and output addrs is nullptr or malloced with wrong sizes.
 */
cnmlStatus_t cnmlCpuComputePluginNmsOpForward(
    cnmlPluginNmsOpParam_t param,
    float *input,
    float *output);

/* --------------------------- */
/* cnmlPluginNms operation end */
/* --------------------------- */

/* ================================ */
/* cnmlPluginInitOp operation start */
/* ================================ */
/*!
 *  @struct cnmlPluginInitOpParam
 *  @brief A struct.
 *
 *  cnmlPluginInitOpParam is a structure describing the "param"
 *  parameter of cnmlPluginInitOpParam operation.
 *  cnmlCreatePluginInitOpParam() is used to create an instance of
 *  cnmlPluginInitParam_t.
 *  cnmlDestroyPluginInitOpParam() is used to destroy an instance of
 *  cnmlPluginInitParam_t.
 *  Detailed information of parameters, see cnmlCreatePluginInitOpParam().
 */
struct cnmlPluginInitOpParam;
/*! ``cnmlPluginInitOpParam_t`` is a pointer to a
    structure (cnmlPluginInitOpParam) holding the description of a InitOp operation param.
*/
typedef cnmlPluginInitOpParam *cnmlPluginInitOpParam_t;

/*!
 *  @brief A function.
 *
 *  This function creates a PluginInitOp param object with
 *  the pointer and parameters provided by user. This implementation is based
 *  on the official MXNet website.
 *
 *  **Supports MXNet on MLU270.**
 *
 *  @param[out] param
 *    Output. The returning param descriptor.
 *  @param[in] size
 *    Input. The size of need initialized.
 *  @param[float] value
 *    Input. The value of should be initialized.
 *  @param[in] dtype_flag
 *    Input. The data type of input. 0:float32, 1:float64, 2:float16
 *  @param[cnmlCoreVersion_t] coreVersion
 *    Input. The core version of MLU.
 */
cnmlStatus_t cnmlCreatePluginInitOpParam(
  cnmlPluginInitOpParam_t *param,
  int size,
  float value,
  int dtype_flag=2,
  cnmlCoreVersion_t coreVersion=CNML_MLU270);

/*!
 *  @brief A function.
 *
 *  This function frees the PluginInitOpParam struct, pointed by
 *  the pointer provided by user.
 *
 *  @param[in]  param
 *    Input. A pointer to the address of the struct of computation parameters
 *    for PluginInit operator.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Param is a null pointer.
 *    - The pointer content pointed by param is already freed.
 */
cnmlStatus_t cnmlDestroyPluginInitOpParam(
    cnmlPluginInitOpParam_t *param);

/*!
 *  @brief A function.
 *
 *  This function creates PluginInitOp with proper param,
 *  input, and output tensors. The current implementation is based on the
 *  official MXNet Init op.
 *
 *  **Supports MXNet on MLU270.**
 *
 *  @param[out]  op
 *    Output. A pointer to the base operator address.
 *  @param[in]  param
 *    Input. A PluginInitOp parameter struct pointer.
 *  @param[in]  Init_input_tensors
 *    Input. An array of multi-dimensional cnmlTensors .
 *           Support only FLOAT16 dataType currently.
 *  @param[in]  outputs
 *    Output. An array of multi-dimensional cnmlTensors.
 *           Support only FLOAT16 dataType currently.
 *           The size is the length of result.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Base op pointer is nullptr.
 *    - Param is nullptr or not initialized.
 *    - Input and output tensor desps is nullptr or inconsistent with param.
 */
cnmlStatus_t cnmlCreatePluginInitOp(
  cnmlBaseOp_t *op,
  cnmlPluginInitOpParam_t param,
  cnmlTensor_t *Init_input_tensors,
  cnmlTensor_t *Init_output_tensors
);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginInitOp on MLU.
 *
 *  **Supports MLU270.**
 *
 *  @param[in] op
 *    Output. A pointer to the base operator address.
 *  @param[in] input_tensors
 *    Input. Void.
 *  @param[in] inputs
 *    Input. An array stores the address of all input tensors.
 *  @param[in] num_inputs
 *    Input. Number of input tensors.
 *  @param[out] output_tensors
 *    Output. Void.
 *  @param[in] outputs
 *    Input. An array stores the address of all output tensors.
 *  @param[in] num_outputs
 *    Input. Number of output tensors.
 *  @param[in] queue
 *    Input. A computation queue pointer.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Base op is nullptr.
 *    - Input and output addrs is nullptr.
 *    - Task type is invalid at runtime.
 */
cnmlStatus_t cnmlComputePluginInitOpForward(
  cnmlBaseOp_t op,
  void **inputs,
  int input_num,
  void **outputs,
  int output_num,
  cnrtQueue_t queue
);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginInitOp on CPU.
 *
 *  @param[in] param
 *    Input. Param of Init operator, cnmlPluginInitOpParam.
 *  @param[in] output
 *    Input. An address of output tensor.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 */
cnmlStatus_t cnmlCpuComputePluginInitOpForward(
  cnmlPluginInitOpParam_t param,
  float *output
);

/* ---------------------------- */
/* cnmlPluginInit operation end */
/* ---------------------------- */

/* ================================ */
/* cnmlPluginArange operation start */
/* ================================ */
/*!
 *  @struct cnmlPluginArangeOpParam
 *  @brief A struct.
 *
 *  cnmlPluginArangeOpParam is a structure describing the "param"
 *  parameter of cnmlPluginArangeOpParam operation.
 *  cnmlCreatePlugincnmlPluginArangeOpParam() is used to create an instance of
 *  cnmlPluginArangeParam_t.
 *  cnmlDestroyPlugincnmlPluginArangeOpParam() is used to destroy an instance of
 *  cnmlPluginArangeParam_t.
 *  Detailed information of parameters, see cnmlCreatePluginArangeOpParam().
 */
struct cnmlPluginArangeOpParam;
/*! ``cnmlPluginArangeParam_t`` is a pointer to a
    structure (cnmlPluginArangeParam) holding the description of a ArangeOp operation param.
*/
typedef cnmlPluginArangeOpParam *cnmlPluginArangeParam_t;

/*!
 *  @brief A function.
 *
 *  This function creates a PluginArangeOp param object with
 *  the pointer and parameters provided by user. This implementation is based
 *  on the official MXNet website.
 *
 *  **Supports MXNet on MLU270.**
 *
 *  @param[out] param
 *    Output. The returning param descriptor.
 *  @param[float] start
 *    Input. Start of interval.
 *  @param[float] stop
 *    Input. End of interval.
 *  @param[float] step
 *    Input. Spacing between values.
 *  @param[int] repeat
 *    Input. The repeating time of all elements.
 *  @param[int] size
 *    Input. intput shape size .
 *  @param[int] dtype_flag
 *    Input. The data type of input. only support float16 so far
 *  @param[cnmlCoreVersion_t] coreVersion
 *    Input. The core version of MLU.
 *  @retval CNML_STATUS_SUCCESS
 *    The object was set successfully.
 */
cnmlStatus_t cnmlCreatePluginArangeOpParam(
  cnmlPluginArangeParam_t *param,
  float start,
  float stop,
  float step,
  int repeat,
  int size,
  int dtype_flag,
  cnmlCoreVersion_t coreVersion);

/*!
 *  @brief A function.
 *
 *  This function frees the PluginArangeOpParam struct, pointed by
 *  the pointer provided by user.
 *
 *  @param[in]  param
 *    Input. A pointer to the address of the struct of computation parameters.
 *    for PluginNms operator.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Param is a null pointer.
 *    - The pointer content pointed by param is already freed.
 */
cnmlStatus_t cnmlDestroyPluginArangeOpParam(
    cnmlPluginArangeParam_t *param);
/*!
 *  @brief A function.
 *
 *  This function creates PluginArangeOp with proper param,
 *  input, and output tensors. The current implementation is based on the
 *  official MXNet NMS op.
 *
 *  **Supports MXNet on MLU270.**
 *
 *  @param[out]  op
 *    Output. A pointer to the base operator address.
 *  @param[in]  param
 *    Input. A PluginArangeOp parameter struct pointer.
 *  @param[in]  arange_input_tensors
 *    Input. Support only FLOAT16 dataType currently.
 *  @param[in]  outputs
 *    Output. Support only FLOAT16 dataType currently.
 *           The size is the length of result.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Base op pointer is nullptr.
 *    - Param is nullptr or not initialized.
 *    - Input and output tensor desps is nullptr or inconsistent with param.
 */
cnmlStatus_t cnmlCreatePluginArangeOp(
  cnmlBaseOp_t *op,
  cnmlPluginArangeParam_t param,
  cnmlTensor_t *arange_input_tensors,
  cnmlTensor_t *arange_output_tensors
);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginArangeOp on MLU.
 *
 *  **Supports MLU270.**
 *
 *  @param[in] op
 *    Output. A pointer to the base operator address.
 *  @param[in] input_tensors
 *    Input. Void.
 *  @param[in] inputs
 *    Input. An array stores the address of all input tensors.
 *  @param[in] num_inputs
 *    Input. Number of input tensors.
 *  @param[out] output_tensors
 *    Output. Void.
 *  @param[in] outputs
 *    Input. An array stores the address of all output tensors.
 *  @param[in] num_outputs
 *    Input. Number of output tensors.
 *  @param[in] queue
 *    Input. A computation queue pointer.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Base op is nullptr.
 *    - Input and output addrs is nullptr.
 *    - Task type is invalid at runtime.
 */
cnmlStatus_t cnmlComputePluginArangeOpForward(
  cnmlBaseOp_t op,
  void **inputs,
  int input_num,
  void **outputs,
  int output_num,
  cnrtQueue_t queue
);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginArangeOp on CPU.
 *
 *  @param[in] param
 *    Input. A pointer of cnmlPluginArangeParam, which
 *    supports params needed by this operator.
 *  @param[in] output
 *    Input. An address of output tensor.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 */
cnmlStatus_t cnmlCpuComputePluginArangeOpForward(
  cnmlPluginArangeParam_t param,
  float* output
);

/* ------------------------------ */
/* cnmlPluginArange operation end */
/* ------------------------------ */

/* =============================================== */
/* cnmlPluginYolov2DetectionOutput operation start */
/* =============================================== */
/*!
 *  @struct cnmlPluginYolov2DetectionOutputOpParam
 *  @brief A struct.
 *
 *  cnmlPluginYolov2DetectionOutputOpParam is a structure describing the "param"
 *  parameter of Yolov2DetectionOutput operation.
 *  cnmlCreatePluginYolov2DetectionOutputOpParam() is used to create an instance
 *  of cnmlPluginYolov2DetectionOutputOpParam_t.
 *  cnmlDestroyPluginYolov2DetectionOutputOpParam() is used to destroy an
 *  instance of cnmlPluginYolov2DetectionOutputOpParam_t.
 *  Detailed information of parameters, see cnmlCreatePluginYolov2DetectionOutputOpParam().
 */
struct cnmlPluginYolov2DetectionOutputOpParam;
/*! ``cnmlPluginYolov2DetectionOutputOpParam_t`` is a pointer to a
    structure (cnmlPluginYolov2DetectionOutputOpParam) holding the description of a Yolov2DetectionOutput operation param.
*/
typedef cnmlPluginYolov2DetectionOutputOpParam *cnmlPluginYolov2DetectionOutputOpParam_t;

/*!
 *  @brief A function.
 *
 *  This function creates a PluginYolov2DetectionOutputOp param object with
 *  the pointer and parameters provided by user.
 *
 *  **Supports MLU220 and MLU270.**
 *
 *  @param[out] param
 *    Output. The returning param descriptor.
 *  @param[in] width
 *    Input. The width.
 *  @param[in] height
 *    Input. The height.
 *  @param[in] classNum
 *    Input. The number of classes.
 *  @param[in] anchorNum
 *    Input. The number of anchors.
 *  @param[in] coords
 *    Input. The number of anchor coordinates.
 *  @param[in] batchNum
 *    Input. The number of batch.
 *  @param[in] int8_mode
 *    Input. If the net run in int8 mode.
 *  @param[in] confidence_thresh
 *    Input. Confidence threshold.
 *  @param[in] nms_thresh.
 *    Enumerant IOU threshold used in NMS function.
 *  @param[in] core_version
 *    Input. Supported core version, including MLU220, and MLU270.
 *  @param[in] biases
 *    Input. The bias data.
 *  @retval CNML_STATUS_SUCCESS
 *    The object was set successfully.
 *  @warning
 *    The sum of input tensor HW values should be less than 32768.
 */
cnmlStatus_t cnmlCreatePluginYolov2DetectionOutputOpParam(
    cnmlPluginYolov2DetectionOutputOpParam_t *param,
    int width,
    int height,
    int classNum,
    int anchorNum,
    int coords,
    int paramNum,
    int batchNum,
    int int8_mode,
    float confidence_thresh,
    float nms_thresh,
    cnmlCoreVersion_t core_version,
    float* biases);

/*!
 *  @brief A function.
 *
 *  This function creates a PluginYolov2DetectionOutputOp param object with
 *  the pointer and parameters provided by user.
 *
 *  **Supports MLU220 and MLU270.**
 *
 *  @param[out] param
 *    Output. The returning param descriptor.
 *  @param[in] width
 *    Input. The width.
 *  @param[in] height
 *    Input. The height.
 *  @param[in] classNum
 *    Input. The number of classes.
 *  @param[in] anchorNum
 *    Input. The number of anchors.
 *  @param[in] coords
 *    Input. The number of anchor coordinates.
 *  @param[in] batchNum
 *    Input. The number of batch.
 *  @param[in] int8_mode
 *    Input. If the net run in int8 mode.
 *  @param[in] confidence_thresh
 *    Input. Confidence threshold.
 *  @param[in] nms_thresh.
 *    Enumerant IOU threshold used in NMS function.
 *  @param[in] core_version
 *    Input. Supported core version, including MLU220, and MLU270.
 *  @param[in] data_type
 *    Input. Data type of input data, either CNML_DATA_FLOAT16 or CNML_DATA_FLOAT32.
 *  @param[in] biases
 *    Input. The bias data.
 *  @retval CNML_STATUS_SUCCESS
 *    The object was set successfully.
 *  @warning
 *    The sum of input tensor HW values should be less than 32768.
 */
cnmlStatus_t cnmlCreatePluginYolov2DetectionOutputOpParam_V2(
    cnmlPluginYolov2DetectionOutputOpParam_t *param,
    int width,
    int height,
    int classNum,
    int anchorNum,
    int coords,
    int paramNum,
    int batchNum,
    int int8_mode,
    float confidence_thresh,
    float nms_thresh,
    cnmlCoreVersion_t core_version,
    cnmlDataType_t data_type,
    float* biases);

/*!
 *  @brief A function.
 *
 *  This function frees the PluginYolov2DetectionOutputOpParam struct, pointed
 *  by the pointer provided by user.
 *
 *  **Supports MLU220 and MLU270.**
 *
 *  @param[in]  param
 *    Input. A pointer to the address of the struct of computation parameters
 *    for PluginYolov2DetectionOutput operator.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Param is a null pointer.
 *    - The pointer content pointed by param is already freed.
 */
cnmlStatus_t cnmlDestroyPluginYolov2DetectionOutputOpParam(
    cnmlPluginYolov2DetectionOutputOpParam_t *param);

/*!
 *  @brief A function.
 *
 *  This function creates PluginYolov2DetectionOutputOp with proper param,
 *  input, and output tensors. The current implementation is based on the
 *  official caffe website of weiliu86.
 *
 *  **Supports MLU220 and MLU270.**
 *
 *  @param[out]  op
 *    Output. A pointer to the base operator address.
 *  @param[in]  param
 *    Input. A PluginYolov2DetectionOutput parameter struct pointer.
 *  @param[in]  yolov2_input_tensors
 *    Input. An array of four-dimensional cnmlTensors with a shape of
 *           [batchNum, anchornum * width * height, 1, (paramnum + 5)](NCHW).
 *           Support only FLOAT16 dataType currently.
 *  @param[in]  outputs
 *    Input. An array of four-dimensional cnmlTensors with a shape of
 *           [batchNum, 1, 7, 256](NCHW).
 *           Support only FLOAT16 dataType currently.
 *           The first two numbers of each batch store the number of
 *           detected boxes. The data for each box starts from the 65th number,
 *           with an order of [batchId, classId, score, xc, yc, w, h], where
 *           (xc, yc) is the coordinates of center of the box, w is the width of
 *           the bos, h is the height of the box.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Base op pointer is nullptr
 *    - Param is nullptr or not initialized
 *    - Input and output tensor desps is nullptr or inconsistent with param.
 */
cnmlStatus_t cnmlCreatePluginYolov2DetectionOutputOp(
    cnmlBaseOp_t *op,
    cnmlPluginYolov2DetectionOutputOpParam_t param,
    cnmlTensor_t *yolov2_input_tensors,
    cnmlTensor_t *yolov2_output_tensors);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginYolov2DetectionOutputOp on MLU.
 *
 *  **Supports MLU220 and MLU270.**
 *
 *  @param[out]  op
 *    Input. A pointer to the base operator address.
 *  @param[in]  inputs
 *    Input. An array stores the address of all input tensors.
 *  @param[in]  num_inputs
 *    Input. Number of input tensors.
 *  @param[out]  outputs
 *    Output. An array stores the address of all output tensors.
 *  @param[in]  num_outputs
 *    Input. Number of output tensors.
 *  @param[in]  compute_forw_param
 *    Input. A pointer to the struct address, which records runtime degree of
 *    data parallelism and equipment affinity.
 *  @param[in]  queue
 *    Input. A computation queue pointer.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Base op is nullptr.
 *    - Input and output addrs is nullptr.
 *    - Input and output nums are inconsistent.
 *    - Task type is invalid at runtime.
 */
cnmlStatus_t cnmlComputePluginYolov2DetectionOutputOpForward_V2(
    cnmlBaseOp_t op,
    void *inputs[],
    int num_inputs,
    void *outputs[],
    int num_outputs,
    cnrtInvokeFuncParam_t *compute_forw_param,
    cnrtQueue_t stream);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginYolov2DetectionOutputOp on CPU.
 *
 *  @param[in]  param
 *    Input. A PluginYolov2DetectionOutput parameter struct pointer.
 *  @param[in]  inputs
 *    Input. An array stores the address of all cpu input data.
 *  @param[in]  biases_ori
 *    Input. An array stores the address of bias input data.
 *  @param[out]  outputs
 *    Output. An array stores the address of all cpu output data.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Param is nullptr or inconsistent.
 *    - Input and output addrs is nullptr or malloced with wrong sizes.
 */
cnmlStatus_t cnmlCpuComputePluginYolov2DetectionOutputOpForward(
    cnmlPluginYolov2DetectionOutputOpParam_t param,
    void *inputs,
    void *biases_ori,
    void *outputs);

/* --------------------------------------------- */
/* cnmlPluginYolov2DetectionOutput operation end */
/* --------------------------------------------- */

/* ================================= */
/* cnmlPluginBertPre operation start */
/* ================================= */
/*!
 *  @struct cnmlPluginBertPreParam
 *  @brief A struct.
 *
 *  cnmlPluginBertPreParam is a structure describing the "param"
 *  parameter of BertPre operation.
 *  cnmlCreatePluginBertPreOpParam() is used to create an instance of
 *  cnmlPluginBertPreOpParam_t.
 *  cnmlDestroyPluginBertPreOpParam() is used to destroy an instance of
 *  cnmlPluginBertPreOpParam_t.
 *  Detailed information of parameters, see cnmlCreatePluginBertPreOpParam().
 */
struct cnmlPluginBertPreParam;

/*! ``cnmlPluginBertPreParam_t`` is a pointer to a
    structure (cnmlPluginBertPreParam) holding the description of a BertPre operation param.
*/
typedef cnmlPluginBertPreParam *cnmlPluginBertPreParam_t;

/*!
 *  @brief A function.
 *
 *  This function creates a PluginBertPreOp param object with
 *  the pointer and parameters provided by user.
 *
 *  **Supports MLU270.**
 *
 *  @param[out] param
 *    Output. The returning param descriptor.
 *  @param[in]  word_table_ptr
 *    Input. An array stores the word table.
 *  @param[in]  segment_table_ptr
 *    Input. An array stores the segment table.
 *  @param[in]  position_table_ptr
 *    Input. An array stores the position table.
 *  @param[in]  layernorm_gamma_ptr
 *    Input. An array stores the layernorm gamma.
 *  @param[in]  layernorm_bata_ptr
 *    Input. An array stores the layernorm bata.
 *  @param[in]  vocab_size
 *    Input. The size of vacab embedding table.
 *  @param[in]  segment_size
 *    Input. The size of segment embedding table.
 *  @param[in]  position_size
 *    Input. The size of position embedding table.
 *  @param[in]  batch_num
 *    Input. The number of batch.
 *  @param[in]  seq_len
 *    Input. The length of sequence.
 *  @param[in]  hidden_size
 *    Input. The size of embedding vector.
 *  @retval CNML_STATUS_SUCCESS
 *    The object was set successfully.
 */
cnmlStatus_t cnmlCreatePluginBertPreOpParam(
    cnmlPluginBertPreParam_t *param,
    cnmlCoreVersion_t core_version,
    float* word_table_ptr,
    float* segment_table_ptr,
    float* position_table_ptr,
    float* layernorm_gamma_ptr,
    float* layernorm_beta_ptr,
    int vocab_size,
    int segment_size,
    int position_size,
    int batch_num,
    int seq_len,
    int hidden_size);

/*!
 *  @brief A function.
 *
 *  This function creates a PluginBertPreOp param object with
 *  the pointer and parameters provided by user.
 *
 *  **Supports MLU270.**
 *
 *  @param[out] param
 *    Output. The returning param descriptor.
 *  @param[in]  vocab_size
 *    Input. The size of vacab embedding table.
 *  @param[in]  segment_size
 *    Input. The size of segment embedding table.
 *  @param[in]  position_size
 *    Input. The size of position embedding table.
 *  @param[in]  batch_num
 *    Input. The number of batch.
 *  @param[in]  seq_len
 *    Input. The length of sequence.
 *  @param[in]  hidden_size
 *    Input. The size of embedding vector.
 *  @retval CNML_STATUS_SUCCESS
 *    The object was set successfully.
 */
cnmlStatus_t cnmlCreatePluginBertPreOpParam_V2(
    cnmlPluginBertPreParam_t *param,
    cnmlCoreVersion_t core_version,
    int vocab_size,
    int segment_size,
    int position_size,
    int batch_num,
    int seq_len,
    int hidden_size);

/*!
 *  @brief A function.
 *
 *  This function frees the PluginBertPreParam struct, pointed
 *  by the pointer provided by user.
 *
 *  **Supports MLU270.**
 *
 *  @param[in]  param
 *    Input. A pointer to the address of the struct of computation parameters
 *    for PluginBertPre operator.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Param is a null pointer.
 *    - The pointer content pointed by param is already freed.
 */
cnmlStatus_t cnmlDestroyPluginBertPreOpParam(
    cnmlPluginBertPreParam_t* param);

/*!
 *  @brief A function.
 *
 *  This function frees the PluginBertPreParam struct, pointed
 *  by the pointer provided by user.
 *
 *  **Supports MLU270.**
 *
 *  @param[in]  param
 *    Input. A pointer to the address of the struct of computation parameters
 *    for PluginBertPre operator.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Param is a null pointer.
 *    - The pointer content pointed by param is already freed.
 */
cnmlStatus_t cnmlDestroyPluginBertPreOpParam_V2(
    cnmlPluginBertPreParam_t* param);

/*!
 *  @brief A function.
 *
 *  This function creates PluginBertPreOp with proper param.
 *
 *  **Supports MLU270.**
 *
 *  @param[out] op
 *    Output. A pointer to the base operator address.
 *  @param[in]  param
 *    Input. A PluginBertPre parameter struct pointer.
 *  @param[in] cnml_input_ptr
 *    Input. An array of four-dimensional cnmlTensors for input.
 *  @param[in] cnml_output_ptr
 *    Input. An array of four-dimensional cnmlTensors for output.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - The param is not consistant with tensors.
 */
cnmlStatus_t cnmlCreatePluginBertPreOp(
    cnmlBaseOp_t *op,
    cnmlPluginBertPreParam_t param,
    cnmlTensor_t *cnml_input_ptr,
    cnmlTensor_t *cnml_output_ptr);

/*!
 *  @brief A function.
 *
 *  This function creates PluginBertPreOp with proper param.
 *
 *  **Supports MLU270.**
 *
 *  @param[out] op
 *    Output. A pointer to the base operator address.
 *  @param[in]  param
 *    Input. A PluginBertPre parameter struct pointer.
 *  @param[in] cnml_static_ptr
 *    Input. An array of four-dimensional cnmlTensors for consts.
 *  @param[in] cnml_input_ptr
 *    Input. An array of four-dimensional cnmlTensors for input.
 *  @param[in] cnml_output_ptr
 *    Input. An array of four-dimensional cnmlTensors for output.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - The param is not consistant with tensors.
 */
cnmlStatus_t cnmlCreatePluginBertPreOp_V2(
  cnmlBaseOp_t *op,
  cnmlPluginBertPreParam_t param,
  cnmlTensor_t* cnml_static_ptr,
  cnmlTensor_t *cnml_input_ptr,
  cnmlTensor_t *cnml_output_ptr);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginBertPreOp on MLU.
 *
 *  **Supports MLU270.**
 *
 *  @param[out]  op
 *    Output. A pointer to the base operator address.
 *  @param[in]  param
 *    Input. A PluginBertPre parameter struct pointer.
 *  @param[in]  input_addrs
 *    Input. An array stores the address of all input tensors.
 *  @param[in]  output_addrs
 *    Input. An array stores the address of all output tensors.
 *  @param[in]  compute_forw_param
 *    Input. A pointer to the struct address, which records runtime degree of
 *    data parallelism and equipment affinity.
 *  @param[in]  queue
 *    Input. A computation queue pointer.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Base op is nullptr.
 *    - Input and output addrs is nullptr.
 *    - Input and output nums are inconsistent.
 *    - Task type is invalid at runtime.
 */
cnmlStatus_t cnmlComputePluginBertPreOpForward(
    cnmlBaseOp_t op,
    cnmlPluginBertPreParam_t param,
    void **input_addrs,
    void **output_addrs,
    cnrtInvokeFuncParam_t compute_forw_param,
    cnrtQueue_t queue);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginBertPreOp on MLU.
 *
 *  **Supports MLU270.**
 *
 *  @param[out]  op
 *    Output. A pointer to the base operator address.
 *  @param[in] cnml_input_ptr
 *    Input. An array of four-dimensional cnmlTensors for src image.
 *  @param[in]  input_addrs
 *    Input. An array stores the address of all input tensors.
 *  @param[in] cnml_output_ptr
 *    Input. An array of four-dimensional cnmlTensors for dst image.
 *  @param[in]  output_addrs
 *    Input. An array stores the address of all output tensors.
 *  @param[in]  queue
 *    Input. A computation queue pointer.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Base op is nullptr.
 *    - Input and output addrs is nullptr.
 *    - Input and output nums are inconsistent.
 *    - Task type is invalid at runtime.
 */
cnmlStatus_t cnmlComputePluginBertPreOpForward_V2(
  cnmlBaseOp_t op,
  cnmlTensor_t* cnml_input_ptr,
  void **input_addrs,
  cnmlTensor_t* cnml_output_ptr,
  void **output_addrs,
  cnrtQueue_t queue);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginBertPreOp on CPU.
 *
 *  @param[out]  embedding_result
 *    Input. An array stores the result of embedding result.
 *  @param[out]  attention_index_result
 *    Input. An array stores the result of attention mask index.
 *  @param[in]  word_embedding_table
 *    Input. An array stores the word embedding table.
 *  @param[in]  vocab_size
 *    Input. The size of vacab embedding table.
 *  @param[in]  segment_embedding_table
 *    Input. An array stores the segment embedding table.
 *  @param[in]  segment_size
 *    Input. The size of segment embedding table.
 *  @param[in]  position_embedding_table
 *    Input. An array stores the position embedding table.
 *  @param[in]  position_size
 *    Input. The size of position embedding table.
 *  @param[in]  layernorm_gamma
 *    Input. An array stores the layernorm gamma params.
 *  @param[in]  layernorm_beta
 *    Input. An array stores the layernorm beta params.
 *  @param[in]  input_ids
 *    Input. The word ids input.
 *  @param[in]  token_type_ids
 *    Input. The token type ids input.
 *  @param[in]  attention_mask
 *    Input. The attention mask input.
 *  @param[in]  batch_num
 *    Input. The number of batch.
 *  @param[in]  seq_len
 *    Input. The length of sequence.
 *  @param[in]  hidden_size
 *    Input. The size of embedding vector.
 */
void cnmlCpuComputePluginBertPreOpForward(float* embedding_result,
    int* attention_index_result,
    const float* word_embedding_table,
    int vocab_size,
    const float* segment_embedding_table,
    int segment_size,
    const float* position_embedding_table,
    int position_size,
    const float* layernorm_gamma,
    const float* layernorm_beta,
    const int* input_ids,
    const int* token_type_ids,
    const uint16_t* attention_mask,
    int batch_num,
    int seq_len,
    int hidden_size);
/* ------------------------------- */
/* cnmlPluginBertPre operation end */
/* ------------------------------- */

/* ======================================== */
/* cnmlPluginBertEmbEncoder operation start */
/* ======================================== */

struct cnmlPluginBertEmbEncoderOpParam;

/*! ``cnmlPluginBertEmbEncoderOpParam_t`` is a pointer to a
    structure (cnmlPluginBertEmbEncoderOpParam) holding the description of a BertEmbEncoder operation param.
*/
typedef cnmlPluginBertEmbEncoderOpParam *cnmlPluginBertEmbEncoderOpParam_t;

/*!
 *  @brief A function.
 *
 *  This function creates a PluginBertEmbEncoderOp param object with
 *  the pointer and parameters provided by user.
 *
 *  **Supports MLU270.**
 *
 *  @param[out] param
 *    Output. The returning param descriptor.
 *  @param[in] core_version
 *    Input. Supported core version, only support MLU270.
 *  @param[in]  bert_kernel_case
 *    Input. The kernel case to decide using which mlu kernel.
 *  @param[in]  batch_size
 *    Input. The number of batch.
 *  @param[in]  seq_len
 *    Input. The length of sequence.
 *  @retval CNML_STATUS_SUCCESS
 *    The object was set successfully.
 */
cnmlStatus_t cnmlCreatePluginBertEmbEncoderOpParam(
    cnmlPluginBertEmbEncoderOpParam_t *param,
    cnmlCoreVersion_t core_version,
    int batch_num,
    int seq_len);

/*!
 *  @brief A function.
 *
 *  This function frees the PluginBertEmbEncoderOpParam struct, pointed
 *  by the pointer provided by user.
 *
 *  **Supports MLU270.**
 *
 *  @param[in]  param
 *    Input. A pointer to the address of the struct of computation parameters
 *    for PluginBertEmbEncoder operator.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Param is a null pointer.
 *    - The pointer content pointed by param is already freed.
 */
cnmlStatus_t cnmlDestroyPluginBertEmbEncoderOpParam(
    cnmlPluginBertEmbEncoderOpParam_t* param);
/*!
 *  @brief A function.
 *
 *  This function creates PluginBertEmbEncoderOp with proper param.
 *
 *  **Supports MLU270.**
 *
 *  @param[out] op
 *    Output. A pointer to the base operator address.
 *  @param[in]  param
 *    Input. A PluginBertEmbEncoder parameter struct pointer.
 *  @param[in] input_tensors
 *    Input. An array of four-dimensional cnmlTensors for inputs, it has three
 *    inputs, frist input is input ids for sequence, second input is token type
 *    ids, third input is input mask.
 *  @param[in] output_tensors
 *    Input. An array of four-dimensional cnmlTensors for outputs, it has two
 *    outputs, fisrt is start logits, second is end logits.
 *  @param[in] cnml_static_tensors
 *    Input. An array of four-dimensional cnmlTensors for consts, it has 42
 *    static tensors as follows:
 *    0. word_embedding_table: Vocabulary.
 *    1. segment_embedding_table: Segment table.
 *    2. position_embedding_table: Position table.
 *    3. embedding_layernorm_gamma: Embedding layernorm gamma params.
 *    4. embedding_layernorm_beta: Embedding layernorm beta params.
 *    5~8. attr_kernel_Q_ch: Query weights from channel 0~3.
 *    9. attr_bias_Q: Query bias.
 *    10~13. attr_kernel_K_ch: Key weights from channel 0~3.
 *    14. attr_bias_K: Key bias.
 *    15~18. attr_kernel_V_ch: Value weights from channel 0~3.
 *    19. attr_bias_V: Value_bias.
 *    20~23. attr_output_kernel_ch: Multi-head output weights from channel 0~3.
 *    24. attr_output_bias: Multi-head output bias.
 *    25. attr_layernorm_beta: Attention layernorm beta params.
 *    26. attr_layernorm_gamma: Attention layernorm gamma params.
 *    27~30. inter_kernel_ch: Forward feedback inner layer weights from channel 0~3.
 *    31. inter_bias: Forward feedback inner layer bias.
 *    32~35. output_kernel_ch: Forward feedback output layer weights from channel 0~3.
 *    36. output_bias: Forward feedback output layer bias.
 *    37. output_layernorm_beta: Forward feedback layernorm beta params.
 *    38. output_layernorm_gamma: Forward feedback layernorm gamma params.
 *    39. fix_pos: Fix position for params of int16.
 *  @param[in] static_tensors_num
 *    Input. The number of cnmlTensors for consts.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - The param is not consistant with tensors.
 *    - Tensor shapes does not meet reuqirements.
 */
cnmlStatus_t cnmlCreatePluginBertEmbEncoderOp(
    cnmlBaseOp_t *op,
    cnmlPluginBertEmbEncoderOpParam_t param,
    cnmlTensor_t *input_tensors,
    cnmlTensor_t *output_tensors,
    cnmlTensor_t *cnml_static_tensors,
    int static_tensors_num);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginBertEmbEncoderOp on MLU.
 *
 *  **Supports MLU270.**
 *
 *  @param[out]  op
 *    Output. A pointer to the base operator address.
 *  @param[in]  cnml_input_ptr
 *    Input. An array of four-dimensional cnmlTensors for inputs.
 *  @param[in]  inputs_addrs
 *    Input. An array stores the address of all input tensors.
 *  @param[in]  cnml_output_ptr
 *    Input. An array of four-dimensional cnmlTensors for outputs.
 *  @param[in]  outputs_addrs
 *    Input. An array stores the address of all output tensors.
 *  @param[in]  queue
 *    Input. A computation queue pointer.
 *  @param[in] extra
 *    Input. A pointer contains other input params.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Base op is nullptr.
 *    - Input and output addrs is nullptr.
 *    - Input and output nums are inconsistent.
 *    - Task type is invalid at runtime.
 */
cnmlStatus_t cnmlComputePluginBertEmbEncoderOp(
    cnmlBaseOp_t op,
    cnmlTensor_t *cnml_input_ptr, // default as nullptr
    void **inputs_addrs,
    cnmlTensor_t *cnml_output_ptr, // default as nullptr
    void **outputs_addrs,
    cnrtQueue_t queue,
    void *extra);
/* -------------------------------------- */
/* cnmlPluginBertEmbEncoder operation end */
/* -------------------------------------- */

/* =================================== */
/* cnmlPluginBertSquad operation start */
/* =================================== */

struct cnmlPluginBertSquadOpParam;

/*! ``cnmlPluginBertSquadOpParam_t`` is a pointer to a
    structure (cnmlPluginBertSquadOpParam) holding the description of a BertSquad operation param.
*/
typedef cnmlPluginBertSquadOpParam *cnmlPluginBertSquadOpParam_t;

/*!
 *  @brief A function.
 *
 *  This function creates a PluginBertSquadOp param object with
 *  the pointer and parameters provided by user.
 *
 *  **Supports MLU270.**
 *
 *  @param[out] param
 *    Output. The returning param descriptor.
 *  @param[in]  batch_size
 *    Input. The number of batch.
 *  @param[in]  seq_len
 *    Input. The length of sequence.
 *  @retval CNML_STATUS_SUCCESS
 *    The object was set successfully.
 */
cnmlStatus_t cnmlCreatePluginBertSquadOpParam(
    cnmlPluginBertSquadOpParam_t *param,
    cnmlCoreVersion_t core_version,
    int batch_num,
    int seq_len,
    int hidden_size);

/*!
 *  @brief A function.
 *
 *  This function frees the PluginBertSquadOpParam struct, pointed
 *  by the pointer provided by user.
 *
 *  **Supports MLU270.**
 *
 *  @param[in]  param
 *    Input. A pointer to the address of the struct of computation parameters
 *    for PluginBertSquad operator.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Param is a null pointer.
 *    - The pointer content pointed by param is already freed.
 */
cnmlStatus_t cnmlDestroyPluginBertSquadOpParam(
    cnmlPluginBertSquadOpParam_t *param);
/*!
 *  @brief A function.
 *
 *  Deprecated. This interface will be deleted in next version and
 *  cnmlCreatePluginBertSquadOp_V2 is recommended to use.
 *  This function creates PluginBertSquadOp with proper param.
 *
 *  **Supports MLU270.**
 *
 *  @param[out] op
 *    Output. A pointer to the base operator address.
 *  @param[in]  param
 *    Input. A PluginBertSquad parameter struct pointer.
 *  @param[in] input_tensors
 *    Input. An array of four-dimensional cnmlTensors for inputs, it has three
 *    inputs, frist input is input ids for sequence, second input is token type
 *    ids, third input is input mask.
 *  @param[in] output_tensors
 *    Input. An array of four-dimensional cnmlTensors for outputs, it has two
 *    outputs, fisrt is start logits, second is end logits.
 *  @param[in] cnml_static_tensors
 *    Input. An array of four-dimensional cnmlTensors for consts, it has 42
 *    static tensors as follows:
 *    0. word_embedding_table: Vocabulary.
 *    1. segment_embedding_table: Segment table.
 *    2. position_embedding_table: Position table.
 *    3. embedding_layernorm_beta: Embedding layernorm beta params.
 *    4. embedding_layernorm_gamma: Embedding layernorm gamma params.
 *    5. post_output_kernel: Post processor weights.
 *    6. post_output_bias: Post processsor bias.
 *    7~10. attr_kernel_Q_ch: Query weights from channel 0~3.
 *    11. attr_bias_Q: Query bias.
 *    12~15. attr_kernel_K_ch: Key weights from channel 0~3.
 *    16. attr_bias_K: Key bias.
 *    17~20. attr_kernel_V_ch: Value weights from channel 0~3.
 *    21. attr_bias_V: Value_bias.
 *    22~25. attr_output_kernel_ch: Multi-head output weights from channel 0~3.
 *    26. attr_output_bias: Multi-head output bias.
 *    27. attr_layernorm_beta: Attention layernorm beta params.
 *    28. attr_layernorm_gamma: Attention layernorm gamma params.
 *    29~32. inter_kernel_ch: Forward feedback inner layer weights from channel 0~3.
 *    33. inter_bias: Forward feedback inner layer bias.
 *    34~37. output_kernel_ch: Forward feedback output layer weights from channel 0~3.
 *    38. output_bias: Forward feedback output layer bias.
 *    39. output_layernorm_beta: Forward feedback layernorm beta params.
 *    40. output_layernorm_gamma: Forward feedback layernorm gamma params.
 *    41. fix_pos: Fix position for params of int16.
 *  @param[in] static_tensors_num
 *    Input. The number of cnmlTensors for consts.
 *  @param[in] batch_size
 *    Input. The number of batches for input.
 *  @param[in] seq_num
 *    Input. The number of sequence for every batch.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - The param is not consistant with tensors.
 *    - Tensor shapes does not meet reuqirements.
 */
cnmlStatus_t cnmlCreatePluginBertSquadOp(
    cnmlBaseOp_t *op,
    cnmlTensor_t *input_tensors,
    cnmlTensor_t *output_tensors,
    cnmlTensor_t *cnml_static_tensors,
    int static_tensors_num,
    int batch_size,
    int seq_num);

/*!
 *  @brief A function.
 *
 *  This function creates PluginBertSquadOp with proper param.
 *
 *  **Supports MLU270.**
 *
 *  @param[out] op
 *    Output. A pointer to the base operator address.
 *  @param[in]  param
 *    Input. A PluginBertSquad parameter struct pointer.
 *  @param[in] input_tensors
 *    Input. An array of four-dimensional cnmlTensors for inputs, it has three
 *    inputs, frist input is input ids for sequence, second input is token type
 *    ids, third input is input mask.
 *  @param[in] output_tensors
 *    Input. An array of four-dimensional cnmlTensors for outputs, it has two
 *    outputs, fisrt is start logits, second is end logits.
 *  @param[in] cnml_static_tensors
 *    Input. An array of four-dimensional cnmlTensors for consts, it has 42
 *    static tensors as follows:
 *    0. word_embedding_table: Vocabulary.
 *    1. segment_embedding_table: Segment table.
 *    2. position_embedding_table: Position table.
 *    3. embedding_layernorm_beta: Embedding layernorm beta params.
 *    4. embedding_layernorm_gamma: Embedding layernorm gamma params.
 *    5. post_output_kernel: Post processor weights.
 *    6. post_output_bias: Post processsor bias.
 *    7~10. attr_kernel_Q_ch: Query weights from channel 0~3.
 *    11. attr_bias_Q: Query bias.
 *    12~15. attr_kernel_K_ch: Key weights from channel 0~3.
 *    16. attr_bias_K: Key bias.
 *    17~20. attr_kernel_V_ch: Value weights from channel 0~3.
 *    21. attr_bias_V: Value_bias.
 *    22~25. attr_output_kernel_ch: Multi-head output weights from channel 0~3.
 *    26. attr_output_bias: Multi-head output bias.
 *    27. attr_layernorm_beta: Attention layernorm beta params.
 *    28. attr_layernorm_gamma: Attention layernorm gamma params.
 *    29~32. inter_kernel_ch: Forward feedback inner layer weights from channel 0~3.
 *    33. inter_bias: Forward feedback inner layer bias.
 *    34~37. output_kernel_ch: Forward feedback output layer weights from channel 0~3.
 *    38. output_bias: Forward feedback output layer bias.
 *    39. output_layernorm_beta: Forward feedback layernorm beta params.
 *    40. output_layernorm_gamma: Forward feedback layernorm gamma params.
 *    41. fix_pos: Fix position for params of int16.
 *  @param[in] static_tensors_num
 *    Input. The number of cnmlTensors for consts.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - The param is not consistant with tensors.
 *    - Tensor shapes does not meet reuqirements.
 */
cnmlStatus_t cnmlCreatePluginBertSquadOp_V2(
    cnmlBaseOp_t *op,
    cnmlPluginBertSquadOpParam_t param,
    cnmlTensor_t *input_tensors,
    cnmlTensor_t *output_tensors,
    cnmlTensor_t *cnml_static_tensors,
    int static_tensors_num);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginBertSquadOp on MLU.
 *
 *  **Supports MLU270.**
 *
 *  @param[out]  op
 *    Output. A pointer to the base operator address.
 *  @param[in]  input_tensors
 *    Input. An array of four-dimensional cnmlTensors for inputs.
 *  @param[in]  inputs
 *    Input. An array stores the address of all input tensors.
 *  @param[in]  output_tensors
 *    Input. An array of four-dimensional cnmlTensors for outputs.
 *  @param[in]  outputs
 *    Input. An array stores the address of all output tensors.
 *  @param[in]  queue
 *    Input. A computation queue pointer.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Base op is nullptr.
 *    - Input and output addrs is nullptr.
 *    - Input and output nums are inconsistent.
 *    - Task type is invalid at runtime.
 */
cnmlStatus_t cnmlComputePluginBertSquadOpForward(
    cnmlBaseOp_t op,
    cnmlTensor_t *input_tensors,  // default as nullptr
    void** inputs,
    cnmlTensor_t *output_tensors,  // default as nullptr
    void** outputs,
    cnrtQueue_t queue,
    void *extra);
/* --------------------------------- */
/* cnmlPluginBertSquad operation end */
/* --------------------------------- */

/* ========================================= */
/* cnmlPluginBertBaseEncoder operation start */
/* ========================================= */
/*!
 *  @struct cnmlPluginBertBaseEncoderOpParam
 *  @brief A struct.
 *
 *  cnmlPluginBertBaseEncoderOpParam is a structure describing the "param"
 *  parameter of BertBaseEncoder operation.
 *  cnmlCreatePluginBertBaseEncoderOpParam() is used to create an instance of
 *  cnmlPluginBertBaseEncoderOpParam_t.
 *  cnmlDestroyPluginBertBaseEncoderOpParam() is used to destroy an instance of
 *  cnmlPluginBertBaseEncoderOpParam_t.
 *  Detailed information of parameters, see cnmlCreatePluginBertBaseEncoderOpParam().
 */
struct cnmlPluginBertBaseEncoderOpParam;

/*! ``cnmlPluginBertBaseEncoderOpParam_t`` is a pointer to a
    structure (cnmlPluginBertBaseEncoderOpParam) holding the description of a BertBaseEncoder operation param.
*/
typedef cnmlPluginBertBaseEncoderOpParam *cnmlPluginBertBaseEncoderOpParam_t;

/*!
 *  @brief A function.
 *
 *  This function creates a PluginBertBaseEncoderOp param object with
 *  the pointer and parameters provided by user.
 *
 *  **Supports MLU270.**
 *
 *  @param[out] param
 *    Output. The returning param descriptor.
 *  @param[in]  batch_size
 *    Input. The number of batch.
 *  @param[in]  seq_len
 *    Input. The length of sequence.
 *  @retval CNML_STATUS_SUCCESS
 *    The object was set successfully.
 */
cnmlStatus_t cnmlCreatePluginBertBaseEncoderOpParam(
    cnmlPluginBertBaseEncoderOpParam_t *param,
    cnmlCoreVersion_t core_version,
    int batch_size,
    int seq_len);

/*!
 *  @brief A function.
 *
 *  This function frees the PluginBertBaseEncoderOpParam struct, pointed
 *  by the pointer provided by user.
 *
 *  **Supports MLU270.**
 *
 *  @param[in]  param
 *    Input. A pointer to the address of the struct of computation parameters
 *    for PluginBertBaseEncoder operator.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Param is a null pointer.
 *    - The pointer content pointed by param is already freed.
 */
cnmlStatus_t cnmlDestroyPluginBertBaseEncoderOpParam(
    cnmlPluginBertBaseEncoderOpParam_t* param);

/*!
 *  @brief A function.
 *
 *  This function creates PluginBertBaseEncoderOp with proper param.
 *
 *  **Supports MLU270.**
 *
 *  @param[out] op
 *    Output. A pointer to the base operator address.
 *  @param[in]  param
 *    Input. A PluginBertBaseEncoder parameter struct pointer.
 *  @param[in] cnml_input_ptr
 *    Input. An array of four-dimensional cnmlTensors for inputs.
 *  @param[in] cnml_output_ptr
 *    Input. An array of four-dimensional cnmlTensors for outputs.
 *  @param[in] cnml_static_ptr
 *    Input. An array of four-dimensional cnmlTensors for consts.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - The param is not consistant with tensors.
 *    - Tensor shapes does not meet reuqirements.
 */
cnmlStatus_t cnmlCreatePluginBertBaseEncoderOp(
    cnmlBaseOp_t *op,
    cnmlPluginBertBaseEncoderOpParam_t param,
    cnmlTensor_t *cnml_input_ptr,
    cnmlTensor_t *cnml_output_ptr,
    cnmlTensor_t *cnml_static_ptr);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginBertBaseEncoderOp on MLU.
 *
 *  **Supports MLU270.**
 *
 *  @param[out]  op
 *    Output. A pointer to the base operator address.
 *  @param[in]  param
 *    Input. A PluginBertBaseEncoder parameter struct pointer.
 *  @param[in]  cnml_input_ptr
 *    Input. An array of four-dimensional cnmlTensors for inputs.
 *  @param[in]  input_addrs
 *    Input. An array stores the address of all input tensors.
 *  @param[in]  cnml_output_ptr
 *    Input. An array of four-dimensional cnmlTensors for outputs.
 *  @param[in]  output_addrs
 *    Input. An array stores the address of all output tensors.
 *  @param[in]  queue
 *    Input. A computation queue pointer.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Base op is nullptr.
 *    - Input and output addrs is nullptr.
 *    - Input and output nums are inconsistent.
 *    - Task type is invalid at runtime.
 */
cnmlStatus_t cnmlComputePluginBertBaseEncoderOpForward(
    cnmlBaseOp_t op,
    cnmlPluginBertBaseEncoderOpParam_t param,
    cnmlTensor_t *cnml_input_ptr,
    void **input_addrs,
    cnmlTensor_t *cnml_output_ptr,
    void **output_addrs,
    cnrtQueue_t queue);
/* --------------------------------------- */
/* cnmlPluginBertBaseEncoder operation end */
/* --------------------------------------- */


/* ======================================= */
/* cnmlPluginDepth2SpaceOp operation start */
/* ======================================= */
/*!
 *  @struct cnmlPluginDepth2SpaceOpParam
 *  @brief A struct.
 *
 *  cnmlPluginDepth2SpaceOpParam is a structure describing the "param"
 *  parameter of cnmlPluginDepth2SpaceOp operation.
 *  cnmlCreatePluginDepth2SpaceOpParam() is used to create
 *  an instance of cnmlPluginDepth2SpaceOpParam_t.
 *  cnmlDestroyPluginDepth2SpaceOpParam() is used to destroy
 *  an instance of cnmlPluginDepth2SpaceOpParam_t.
 *  Detailed information of parameters, see cnmlCreatePluginDepth2SpaceOpParam().
 */
struct cnmlPluginDepth2SpaceOpParam;
/*! ``cnmlPluginDepth2SpaceOpParam_t`` is a pointer to a
    structure (cnmlPluginDepth2SpaceOpParam) holding the description
    of a Depth2Space operation param.
*/
typedef cnmlPluginDepth2SpaceOpParam * cnmlPluginDepth2SpaceOpParam_t;

/*!
 *  @brief A function.
 *
 *  This function creates a cnmlPluginDepth2SpaceOp param object with
 *  the pointer and parameters provided by user.
 *
 *  **Supports MLU220 and MLU270.**
 *
 *  @param[out] param_ptr
 *    Output. The returning param descriptor.
 *  @param[in] block_size
 *    Input. The size of the spatial size
 *  @param[in] input_n
 *    Input. number of input tensor.
 *  @param[in] input_h
 *    Input. Height of input tensor.
 *  @param[in] input_w
 *    Input. Width of input tensor.
 *  @param[in] input_c
 *    Input. Depth of input tensor.
 *  @param[in] core_version
 *    Input. Supported core version.
 *           No default value. A valid core_version must be either MLU220 or MLU270.
 *  @param[in] dtype_flag
 *    Input. Data type of input data, either CNML_DATA_FLOAT16 or CNML_DATA_FLOAT32.
 *  @retval CNML_STATUS_SUCCESS
 *    The object was set successfully.
 */
cnmlStatus_t cnmlCreatePluginDepth2SpaceOpParam(
    cnmlPluginDepth2SpaceOpParam_t *param_ptr,
    const int block_size,
    const int input_n,
    const int input_h,
    const int input_w,
    const int input_c,
    const cnmlCoreVersion_t core_version,
    const cnmlDataType_t data_type);

/*!
 *  @brief A function.
 *
 *  This function frees the cnmlPluginDepth2SpaceOpParam struct, pointed by
 *  the pointer provided by user.
 *
 *  **Supports MLU220 and MLU270.**
 *
 *  @param[in]  param_ptr
 *    Input. A pointer to the address of the struct of computation parameters for cnmlPluginDepth2SpaceOp operator.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Param is a null pointer.
 *    - The pointer content pointed by param is already freed.
 */
cnmlStatus_t cnmlDestroyPluginDepth2SpaceOpParam(
    cnmlPluginDepth2SpaceOpParam_t *param_ptr);

/*!
 *  @brief A function.
 *
 *  This function forwards cnmlPluginDepth2SpaceOp on MLU.
 *
 *  **Supports MLU220 and MLU270.**
 *
 *  @param[out]  op
 *    Output. A pointer to the base operator address.
 *  @param[in] input_tensors
 *    Input. Void.
 *  @param[in] inputs
 *    Input. An array stores the address of all input tensors.
 *  @param[in] num_inputs
 *    Input. Number of input tensors.
 *  @param[out] output_tensors
 *    Output. Void.
 *  @param[in] outputs
 *    Input. An array stores the address of all output tensors.
 *  @param[in] num_outputs
 *    Input. Number of output tensors.
 *  @param[in]  queue
 *    Input. A computation queue pointer.
 *  @param[in] extra
 *    Input. A pointer contains other input params.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Base op is nullptr.
 *    - Input and output addrs is nullptr.
 */
cnmlStatus_t cnmlComputePluginDepth2SpaceOpForward(
    cnmlBaseOp_t op,
    cnmlTensor_t input_tensors[],
    void *mlu_input_ptr[],
    int num_inputs,
    cnmlTensor_t output_tensors[],
    void *mlu_output_ptr[],
    int num_outputs,
    cnrtQueue_t queue,
    void *extra);

/*!
 *  @brief A function.
 *
 *  This function forwards cnmlPluginDepth2SpaceOp on CPU.
 *
 *  @param[in]  param
 *    Input. A cnmlPluginDepth2SpaceOp parameter struct pointer.
 *  @param[in]  src_cpu_ptr
 *    Input. CPU address of input image.
 *  @param[in]  padValue_cpu_ptr
 *    Input. CPU address of pad value.
 *  @param[in]  dst_cpu_ptr
 *    Input. CPU address of output image.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Param is nullptr or inconsistent.
 *    - Input and output addrs is nullptr or malloced with wrong sizes.
 */
cnmlStatus_t cnmlCpuComputePluginDepth2SpaceOpForward(
    const cnmlPluginDepth2SpaceOpParam_t param,
    const float *input_cpu_ptr,
    float *output_cpu_ptr);

/*!
 *  @brief A function.
 *
 *  This function creates cnmlPluginDepth2SpaceOp with proper param,
 *  input, and output tensor.
 *
 *  **Supports MLU220 and MLU270.**
 *
 *  @param[out]  op
 *    Output. A pointer to the base operator address.
 *  @param[in]  param_ptr
 *    Input. A cnmlPluginDepth2SpaceOp parameter struct pointer.
 *  @param[in]  input_tensor
 *    Input. A four-dimensional cnmlTensors with a shape of
 *           [channel, input_h, input_w, depth](NHWC).
 *  @param[in]  output_tensor
 *    Input. An array of four-dimensional cnmlTensors with a shape of
 *           [channel, input_h, input_w, depth](NHWC).
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Param is nullptr or not initialized.
 *    - Input and output tensor desps is nullptr or inconsistent with param.
 */
cnmlStatus_t cnmlCreatePluginDepth2SpaceOp(
    cnmlBaseOp_t *op_ptr,
    const cnmlPluginDepth2SpaceOpParam_t param,
    const cnmlTensor_t input_tensor,
    const cnmlTensor_t output_tensor);

/* ------------------------------------- */
/* cnmlPluginDepth2SpaceOp operation end */
/* ------------------------------------- */

/* ===================================== */
/* cnmlPluginCombinedNMS operation start */
/* ===================================== */
/*!
 *  @struct cnmlPluginCombinedNMSOpParam
 *  @brief A struct.
 *
 *  cnmlPluginCombinedNMSOpParam is a structure describing the "param"
 *  parameter of CombinedNMS operation.
 *  cnmlCreatePluginCombinedNMSOpParam() is used to create an instance of
 *  cnmlPluginCombinedNMSOpParam_t.
 *  cnmlDestoryPluginCombinedNMSOpParam() is used to destory an instance of
 *  cnmlPluginCombinedNMSOpParam_t.
 *  Detailed information of parameters, see cnmlCreatePluginCombinedNMSOpParam().
 */
struct cnmlPluginCombinedNMSOpParam;
/*! ``cnmlPluginCombinedNMSOpParam_t`` is a pointer to a
    structure (cnmlPluginCombinedNMSOpParam) holding the description of a CombinedNMS operation param.
*/
typedef cnmlPluginCombinedNMSOpParam
*cnmlPluginCombinedNMSOpParam_t;

/*!
 *  @brief A function.
 *
 *  This function creates a PluginCombinedNMSOp param object with a pointer
 *  and "user params" provided.
 *
 *  @param[out] param
 *    Output. The returning param descriptor.
 *  @param[in] max_size_per_class
 *    Input. The max output size of per class.
 *  @param[in] total_size_per_batch
 *    Input. The max output size of per batch.
 *  @param[in] nms_topk
 *    Input. Do topk before nms.
 *  @param[in] score_threshold
 *    Input. The threshold of score to do nms.
 *  @param[in] iou_threshold
 *    Input. The threshold of iou to do nms.
 *  @param[in] pad_per_class
 *    Input. Whether to pad per class.
 *  @param[in] clip_boxes
 *    Input. Whether to clip boxes.
 *  @param[in] normalized
 *    Input. Whether to do normalize.
 *  @param[in] nms_eta
 *    Input. Dynamic adjust iou_threshold.
 *  @param[in] background_label.
 *    Input. Label of background
 *  @param[in] batch_size
 *    Input. Batch size.
 *  @param[in] num_classes
 *    Input. Num of classes.
 *  @param[in] num_boxes
 *    Input. Num of boxes.
 *  @param[in] box_size
 *    Input. Dimensions of box.
 *  @param[in] data_type
 *    Input. 2 represent float and 4 represent half.
 *  @param[in] core_num
 *    Input. Core number.
 *  @param[in] core_version
 *    Input. Core version, support MLU270 and MLU220.
 *  @retval CNML_STATUS_SUCCESS
 *    The object was set successfully.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    One of the following conditions is met:
 *    Core_version is not CNML_MLU270 or CNML_MLU220
 *    Core_version is CNML_MLU220 but core_num is greater than 4
 */
cnmlStatus_t cnmlCreatePluginCombinedNMSOpParam(
    cnmlPluginCombinedNMSOpParam_t *param,
    int max_size_per_class,
    int total_size_per_batch,
    int nms_top_k,
    float score_threshold,
    float iou_threshold,
    bool pad_per_class,
    bool clip_boxes,
    bool normalized,
    float nms_eta,
    int background_label,
    int batch_size,
    int num_classes,
    int num_boxes,
    int box_size,
    int data_type,
    int core_num,
    cnmlCoreVersion_t core_version);

/*!
 *  @brief A function.
 *
 *  This function frees the PluginCombinedNMSOpParam struct, pointed by
 *  the pointer provided by user.
 *
 *  @param[in]  param
 *    Input. A pointer to the address of the struct of computation parameters
 *    for PluginCombinedNMS operator.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Param is a null pointer.
 *    - The pointer content pointed by param is already freed.
 */
cnmlStatus_t cnmlDestroyPluginCombinedNMSOpParam(
    cnmlPluginCombinedNMSOpParam_t *param);

/*!
 *  @brief A function.
 *
 *  This function creates PluginCombinedNMSOp with proper param,
 *  input, and output tensors.
 *
 *  **Supports Tensorflow on MLU270.**
 *
 *  @param[out] op
 *    Output. A pointer to the base operator address.
 *  @param[in] param
 *    Input. A PluginCombinedNMS parameter struct pointer.
 *  @param[in] combined_nms_input_tensors
 *    Input. This pointer contains two array of cnmlTensors.
 *  @param[out] combined_nms_output_tensors
 *    Output. This pointer contains four array of cnmlTensor.
 *  @param[in] combined_nms_static_tensors
 *    Input. This pointer contains one array of cnmlTensor.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - param is not consistant with tensors.
 *    - shapes of cropParams and roiNums are not consistent.
 */
cnmlStatus_t cnmlCreatePluginCombinedNMSOp(
    cnmlBaseOp_t *op,
    cnmlPluginCombinedNMSOpParam_t param,
    cnmlTensor_t *combined_nms_input_tensors,
    cnmlTensor_t *combined_nms_output_tensors,
    cnmlTensor_t *combined_nms_static_tensors);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginCombinedNMSOp on MLU.
 *
 *  **Supports MLU270.**
 *
 *  @param[in] op
 *    Output. A pointer to the base operator address.
 *  @param[in] input_tensors
 *    Input. Void.
 *  @param[in] inputs
 *    Input. An array stores the address of all input tensors.
 *  @param[in] num_inputs
 *    Input. Number of input tensors.
 *  @param[out] output_tensors
 *    Output. Void.
 *  @param[in] outputs
 *    Input. An array stores the address of all output tensors.
 *  @param[in] num_outputs
 *    Input. Number of output tensors.
 *  @param[in] queue
 *    Input. A computation queue pointer.
 *  @param[in] extra
 *    Input. A pointer contains other input params.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Base op is nullptr.
 *    - Input and output addrs is nullptr.
 *    - Task type is invalid at runtime.
 */
cnmlStatus_t cnmlComputePluginCombinedNMSOpForward(
    cnmlBaseOp_t op,
    cnmlTensor_t input_tensors[],
    void *inputs[],
    int num_inputs,
    cnmlTensor_t output_tensors[],
    void *outputs[],
    int num_outputs,
    cnrtQueue_t queue,
    void *extra);

/* ----------------------------------- */
/* cnmlPluginCombinedNMS operation end */
/* ----------------------------------- */

/* =============================== */
/* cnmlPluginPadOp operation start */
/* =============================== */
/*!
 *  @struct cnmlPluginPadOpParam
 *  @brief A struct.
 *
 *  cnmlPluginPadOpParam is a structure describing the "param"
 *  parameter of cnmlPluginPadOp operation.
 *  cnmlCreatePluginPadOpParam() is used to create
 *  an instance of cnmlPluginPadOpParam_t.
 *  cnmlDestroyPluginPadOpParam() is used to destroy
 *  an instance of cnmlPluginPadOpParam_t.
 *  Detailed information of parameters, see cnmlCreatePluginPadOpParam().
 */
struct cnmlPluginPadOpParam;

/*! ``cnmlPluginPadOpParam_t`` is a pointer to a
    structure (cnmlPluginPadOpParam) holding the description
    of a pad operation param.
*/
typedef cnmlPluginPadOpParam * cnmlPluginPadOpParam_t;

/*!
 *  @brief A function.
 *
 *  This function creates a cnmlPluginPadOp param object with
 *  the pointer and parameters provided by user.
 *
 *  **Supports MLU220 and MLU270.**
 *
 *  @param[out] param_ptr
 *    Output. The returning param descriptor.
 *  @param[in] input_dims
 *    Input. The input shape.
 *  @param[in] padding_value
 *    Input. A pointer to device paddings data. Default 0.
 *  @param[in] core_version
 *    Input. Supported core version.
 *           No default value. A valid core_version must be either MLU220 or MLU270.
 *  @param[in] data_type
 *    Input. Data type of input data, either CNML_DATA_FLOAT16 or CNML_DATA_FLOAT32.
 *  @retval CNML_STATUS_SUCCESS
 *    The object was set successfully.
 */
cnmlStatus_t cnmlCreatePluginPadOpParam(cnmlPluginPadOpParam_t *param_ptr,
                                        std::vector<int> input_dims,
                                        const float *padding_value,
                                        const cnmlCoreVersion_t core_version,
                                        const cnmlDataType_t data_type);

/*!
 *  @brief A function.
 *
 *  This function frees the cnmlPluginPadOpParam struct, pointed by
 *  the pointer provided by user.
 *
 *  **Supports MLU220 and MLU270.**
 *
 *  @param[in]  param_ptr
 *    Input. A pointer to the address of the struct of computation parameters for cnmlPluginPadOp operator.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Param is a null pointer.
 *    - The pointer content pointed by param is already freed.
 */
cnmlStatus_t cnmlDestroyPluginPadOpParam(cnmlPluginPadOpParam_t *param);

/*!
 *  @brief A function.
 *
 *  This function creates cnmlPluginPadOp with proper param,
 *  input, and output tensor.
 *
 *  **Supports MLU220 and MLU270.**
 *
 *  @param[out]  op
 *    Output. A pointer to the base operator address.
 *  @param[in]  param_ptr
 *    Input. A cnmlPluginPadOp parameter struct pointer.
 *  @param[in]  input_tensor
 *    Input. A one-dimensional to four-dimensional cnmlTensors with a shape of
 *           [channel, input_h, input_w, depth](NHWC).
 *  @param[in]  paddins
 *    Input. A two-dimensional cnmlTensors.
 *  @param[in]  output_tensor
 *    Input. An array of one-dimensional to four-dimensional cnmlTensors with a shape of
 *           [channel, input_h, input_w, depth](NHWC).
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Param is nullptr or not initialized.
 *    - Input and output tensor desps is nullptr or inconsistent with param.
 */
cnmlStatus_t cnmlCreatePluginPadOp(cnmlBaseOp_t *op_ptr,
                                   const cnmlPluginPadOpParam_t param_ptr,
                                   const cnmlTensor_t input_tensor,
                                   const cnmlTensor_t pad_tensor,
                                   const cnmlTensor_t output_tensor);

/*!
 *  @brief A function.
 *
 *  This function forwards cnmlPluginPadOp on MLU.
 *
 *  **Supports MLU220 and MLU270.**
 *
 *  @param[out]  op
 *    Output. A pointer to the base operator address.
 *  @param[in] input_tensors
 *    Input. Void.
 *  @param[in] inputs
 *    Input. An array stores the address of all input tensors.
 *  @param[in] num_inputs
 *    Input. Number of input tensors.
 *  @param[out] output_tensors
 *    Output. Void.
 *  @param[in] outputs
 *    Input. An array stores the address of all output tensors.
 *  @param[in] num_outputs
 *    Input. Number of output tensors.
 *  @param[in]  queue
 *    Input. A computation queue pointer.
 *  @param[in] extra
 *    Input. A pointer contains other input params.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Base op is nullptr.
 *    - Input and output addrs is nullptr.
 */
cnmlStatus_t cnmlComputePluginPadOpForward(cnmlBaseOp_t op,
                                           cnmlTensor_t input_tensors[],
                                           void *mlu_input_ptr[],
                                           int num_inputs,
                                           cnmlTensor_t output_tensors[],
                                           void *mlu_output_ptr[],
                                           int num_outputs,
                                           cnrtQueue_t queue,
                                           void *extra);

/*!
 *  @brief A function.
 *
 *  This function forwards cnmlPluginPadOp on CPU.
 *
 *  @param[in]  param
 *    Input. A cnmlPluginPadOp parameter struct pointer.
 *  @param[in]  input_cpu_ptr
 *    Input. CPU address of input data.
 *  @param[in]  paddings_cpu_ptr
 *    Input. CPU address of padding data.
 *  @param[in]  input_dims
 *    Input. CPU address of input shape.
 *  @param[in]  output_cpu_ptr
 *    Input. CPU address of output data.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally.
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Param is nullptr or inconsistent.
 *    - Input and output addrs is nullptr or malloced with wrong sizes.
 */
cnmlStatus_t cnmlCpuComputePluginPadOpForward(const cnmlPluginPadOpParam_t param_ptr,
                                              const float *input_cpu_ptr,
                                              const int *paddings_cpu_ptr,
                                              std::vector<int> input_dims,
                                              float *output_cpu_ptr);
/* ----------------------------- */
/* cnmlPluginPadOp operation end */
/* ----------------------------- */

/* ================================== */
/* cnmlPluginBoxCoder operation start */
/* ================================== */
/*!
 *  @struct cnmlPluginBoxCoderOpParam
 *  @brief A struct.
 *
 *  cnmlPluginBoxCoderOpParam is a structure describing the
 *  parameter of BoxCoder operation.
 *  cnmlCreatePluginBoxCoderOpParam() is used to create
 *  an instance of cnmlPluginBoxCoderOpParam_t.
 *  cnmlDestroyPluginBoxCoderOpParam() is used to destroy
 *  an instance of cnmlPluginBoxCoderOpParam_t.
 */
struct  cnmlPluginBoxCoderOpParam {
    int row;
    int col;
    int len;
    int axis;
    bool normalized;
    bool float_precision;
    cnmlBoxCodeType_t code_type;
    cnmlCoreVersion_t core_version;
};

/*! ``cnmlPluginBoxCoderOpParam_t`` is a pointer to a
    structure (cnmlPluginBoxCoderOpParam) holding the description of a PluginBoxCoder operation param.
*/
typedef cnmlPluginBoxCoderOpParam *cnmlPluginBoxCoderOpParam_t;

/*!
 *  @brief A function.
 *
 *  This function create the cnmlPluginBoxCoderOpParam struct, pointed
 *  by the pointer provided by user.
 *
 *  **Supports MLU270**
 *
 *  @param[in]  param
 *    Input. A pointer to the address of the struct of computation parameters
 *    for PluginBoxCoder operator.
 *  @param[in] row
 *    Input. An int, relevant to tensor shape
 *  @param[in] col
 *    Input. An int, relevant to tensor shape
 *  @param[in] len
 *    Input. An int, len of one box, must be 4
 *  @param[in] axis
 *    Input. An int, axis param relevant to prior box tensor shape, must be 0 or 1
 *  @param[in] normalized
 *    Input. An bool, normalize data in 0 and 1 or not
 *  @param[in] float_precision
 *    Input. An bool, input and output if float or half, true for float, false for half
 *  @param[in] core_version
 *    Input. An cnmlCoreVersion_t, only support CNML_MLU270 for now
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *      One of row and col <=0;
 *      len !=4;
 *      axis !=0 && axis !=1;
 *      core_version != CNML_MLU270;
 */
cnmlStatus_t cnmlCreatePluginBoxCoderOpParam(
   cnmlPluginBoxCoderOpParam_t *param,
    int row,
    int col,
    int len,
    int axis,
    bool normalized,
    bool float_precision,
    cnmlBoxCodeType_t code_type,
    cnmlCoreVersion_t core_version);

/*!
 *  @brief A function.
 *
 *  This function frees the cnmlPluginBoxCoderOpParam struct, pointed
 *  by the pointer provided by user.
 *
 *  **Supports MLU270**
 *
 *  @param[in]  param
 *    Input. A pointer to the address of the struct of computation parameters
 *    for PluginBoxCoder operator.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Param is a null pointer.
 *    - The pointer content pointed by param is already freed.
 */
cnmlStatus_t cnmlDestroyPluginBoxCoderOpParam(
   cnmlPluginBoxCoderOpParam_t *param);

/*!
 *  @brief A function.
 *
 *  This function creates PluginBoxCoder
 *
 *  **Supports MLU270**
 *
 *  **Reference:**
 *
 *    This implementation is based on paddle.fluid.layers.box_coder
 *
 *  **DataType:**
 *    Support float16 and float32 for both input and output tensors, but data type of input/output
 *    tensor must be the same.
 *
 *  @param[in] op
 *    Input. A pointer to the base operator address.
  *  @param[in]  param
 *    Input. A PluginBoxCoder parameter struct pointer.
 *  @param[in] input_tensors
 *    array of cnmlTensor_t, should be [target_box_tesnor,prior_box_tensor,prior_box_var_tensor]
 *    target_box_tesnor:
 *           An 2-dimensional or 3-dimensional cnmlTensors for feat map input
 *           shape = [row,len] when code_type == kEncodeCenterSize,
 *           where row means num of box;
 *           shape = [row,col,len] when code_type == kDncodeCenterSize,
 *           where row means num of box, col means num of classes
 *  prior_box_tensor:
 *           An 2-dimensional cnmlTensors for box coord
 *           shape = [col,len] when axis == 0
 *           shape = [row,len] when axis == 1
 *  prior_box_var_tensor:
 *           An 3-dimensional cnmlTensors for output
 *           shape = [row,col,len],when code_type == kEncodeCenterSize,
 *           shape = [col,row,len],when code_type == kDecodeCenterSize,
 *  @param[out] output_tensors
 *    array of cnmlTensor_t, should be [output_tensor]
 *    Output. output_tensor:
 *            shape [row, col, len]
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - any one of tensors is nullptr;
 *    - tensor shapes does not meet reuqirements;
 */
cnmlStatus_t cnmlCreatePluginBoxCoderOp(
    cnmlBaseOp_t *op_ptr,
    cnmlPluginBoxCoderOpParam_t param,
    cnmlTensor_t * input_tensors,
    cnmlTensor_t * output_tensors);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginBoxCoderOp on MLU.
 *
 *  **Supports MLU270**
 *
 *  @param[out]  op
 *    Output. A pointer to the base operator address.
 *  @param[in]  input_addrs
 *    Input. An array of MLU addrs,
 *           should be [target_box_mlu_ptr,prior_box_mlu_ptr,prior_box_var_mlu_ptr]
 *  @param[in]  input_num
 *    Input. An int, must == 3
 *  @param[in]  output_addrs
 *    Input. An array of MLU addrs,
 *           should be [output_mlu_ptr]
 *  @param[in]  output_num
 *    Input. An int, must == 1
 *  @param[in]  queue
 *    Input. A computation queue pointer.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Base op is nullptr.
 *    - Input / output addrs is nullptr.
 *    - Input / output nums are inconsistent.
 */
cnmlStatus_t cnmlComputePluginBoxCoderOpForward(
    cnmlBaseOp_t op,
    void **input_addrs,
    const int input_num,
    void **output_addrs,
    const int output_num,
    cnrtQueue_t queue);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginBoxCoderOp on CPU.
 *
 *  @param[in]  target_box_data,
 *    Input. Adress of cpu target box data
 *  @param[in]  prior_box_data
 *    Input. Adress of cpu prior box data
 *  @param[in] prior_box_var_data
 *    Input. Adress of cpu prior box variance data
 *  @param[in]  target_box_shape,
 *    Input. shape of cpu target box data
 *  @param[in]  prior_box_shape
 *    Input. shape of cpu prior box data
 *  @param[in] prior_box_var_shape
 *    Input. shape of cpu prior box variance data
 *  @param[in] normalized
 *    Input. An bool,prior_box is normallized or not
 *  @param[in] axis
 *    Input. on which axis to decode, works only when code_type == kDecodeCenterSize
 *  @param[in] variance
 *    Input. prior box variance data if all box have same variance
 *  @param[out]  outputs
 *    Output. Adress of cpu output data
 *  @param[in] code_type
 *    Input. An cnmlBoxCodeType_t, one of Encode, Decode
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Input / output addrs is nullptr or malloced with wrong sizes.
 */
cnmlStatus_t cnmlCpuComputePluginBoxCoderOp(
    float *target_box_data,
    float *prior_box_data,
    float *prior_box_var_data,
    std::vector<int> target_box_shape,
    std::vector<int> prior_box_shape,
    std::vector<int> prior_box_var_shape,
    bool normalized,
    int axis,
    std::vector<float> variance,
    float *output_data,
    cnmlBoxCodeType_t code_type);
/* -------------------------------- */
/* cnmlPluginBoxCoder operation end */
/* -------------------------------- */

/* ========================================= */
/* cnmlPluginDensityPriorBox operation start */
/* ========================================= */
/*!
 *  @struct cnmlPluginDensityPriorBoxOpParam
 *  @brief A struct.
 *
 *  cnmlPluginDensityPriorBoxOpParam is a structure describing the
 *  parameter of PluginDensityPriorBox operation.
 *  cnmlCreatePluginDensityPriorBoxOpParam() is used to create
 *  an instance of cnmlPluginDensityPriorBoxOpParam_t.
 *  cnmlDestroyPluginDensityPriorBoxOpParam() is used to destroy
 *  an instance of cnmlPluginDensityPriorBoxOpParam_t.
 */
struct  cnmlPluginDensityPriorBoxOpParam {
   int feat_width;
   int feat_height;
   int img_width;
   int img_height;
   int variances_num;
   cnmlTensor_t variances_tensor;
   int densities_num;
   cnmlTensor_t densities_tensor;
   int fixed_sizes_num;
   cnmlTensor_t fixed_sizes_tensor;
   int fixed_ratios_num;
   cnmlTensor_t fixed_ratios_tensor;
   bool clip;
   float step_w;
   float step_h;
   float offset;
   bool float_precision;
   int static_num;
   void** static_ptrs;
   cnmlCoreVersion_t core_version;
};

/*! ``cnmlPluginDensityPriorBoxOpParam_t`` is a pointer to a
    structure (cnmlPluginDensityPriorBoxOpParam) holding the description of PluginDensityPriorBox operation param.
*/
typedef cnmlPluginDensityPriorBoxOpParam *cnmlPluginDensityPriorBoxOpParam_t;

/*!
 *  This function create the PluginDensityPriorBoxOpParam struct, pointed
 *  by the pointer provided by user.
 *
 *  **Supports MLU270**
 *
 *  @param[in]  param
 *    Input. A pointer to the address of the struct of computation parameters
 *    for PluginDensityPriorBox operator.
 *  @param[in] feat_width
 *    Input. An int for feature map width
 *  @param[in] feat_height
 *    Input. An int for feature map height
 *  @param[in] image_width
 *    Input. An int for image width
 *  @param[in] image_height
 *    Input. An int for image height
 *  @param[in] variances_ptr
 *    Input. Address for variance data
 *  @param[in] variances_num
 *    Input. len for variance data, must == 4
 *  @param[in] densities_ptr
 *    Input. Address for density data
 *  @param[in] densities_num
 *    Input. len for density data
 *  @param[in] fixed_sizes_ptr
 *    Input. Address for fixed size data
 *  @param[in] fixed_sizes_num
 *    Input. len for fixed sizes data
 *  @param[in] fixed_ratios_ptr
 *    Input. Address for fixed ratio data
 *  @param[in] fixed_ratios_num
 *    Input. len for fixed ratio data
 *  @param[in] clip
 *    Input. clip coord data in 0 and 1 or not
 *  @param[in] step_w
 *    Input. step of prior box width, if step_w == 0 or step_h == 0, then it will be calculated automaticly
 *  @param[in] step_h
 *    Input. step of prior box height, if step_w == 0 or step_h == 0, then it will be calculated automaticly
 *  @param[in] offset
 *    Input. offset of prior box center
 *  @param[in] float_precision
 *    Input. An bool,input and output is float or half, true for float, false for half
 *  @param[in] core_version
 *    Input. An cnmlCoreVersion_t, only support CNML_MLU270 for now
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    one of variances_ptr, fixed_sizes_ptr,fixed_ratios_ptr is nullptr;
 *    one of densities_num, fixed_sizes_num, fixed_ratios_num <=0;
 *    variances_num != 4;
 *    one of feat_width, feat_height, img_width, img_height <=0;
 *    core_version != CNML_MLU270;
 */
cnmlStatus_t cnmlCreatePluginDensityPriorBoxOpParam(
   cnmlPluginDensityPriorBoxOpParam_t *param,
   int feat_width,
   int feat_height,
   int img_width,
   int img_height,
   float* variances_ptr,
   int variances_num,
   int* densities_ptr,
   int densities_num,
   float* fixed_sizes_ptr,
   int fixed_sizes_num,
   float* fixed_ratios_ptr,
   int fixed_ratios_num,
   bool clip,
   float step_w,
   float step_h,
   float offset,
   bool float_precision,
   cnmlCoreVersion_t core_version);
/*!
 *  @brief A function.
 *
 *  This function frees the cnmlPluginDensityPriorBoxOpParam struct, pointed
 *  by the pointer provided by user.
 *
 *  **Supports MLU270**
 *
 *  @param[in]  param
 *    Input. A pointer to the address of the struct of computation parameters
 *    for PluginDensityPriorBox operator.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Param is a null pointer.
 *    - The pointer content pointed by param is already freed.
 */
cnmlStatus_t cnmlDestroyPluginDensityPriorBoxOpParam(
   cnmlPluginDensityPriorBoxOpParam_t *param);

/*!
 *  @brief A function.
 *
 *  This function creates PluginDensityPriorBoxOp
 *
 *  **Supports MLU270**
 *
 *  **Reference:**
 *
 *    This implementation is based on paddle.fluid.layers.density_prior_box
 *
 *  **DataType:**
 *    Support float16 and float32 for both input and output tensors, but data type of input/output
 *    tensor must be the same.
 *
 *  @param[out] op_ptr
 *    Output. A pointer to the base operator address.
 *  @param[in]  param
 *    Input. A PluginDensityPriorBox parameter struct pointer.
 *  @param[in]  input_tensors
 *    Input. array of cnmltensor_t,should be [feat_tensor,img_tensor]
 *           feat_tensor: An 4-dimensional  cnmlTensors for feat input
 *           img_tensor: An 4-dimensional  cnmlTensors for image input
  *  @param[in]  output_tensors
 *    Input. array of cnmltensor_t,should be [boxes_tensor,vars_tensor]
 *           boxes_tensor: An 4-dimensional cnmlTensors for box coord output,shape = [feat_width, feat_height,num_priors,4]
 *           vars_tensor: An 4-dimensional cnmlTensors for box coord variance output, shape = [feat_width, feat_height,num_priors,4]
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - len != 4
 *    - tensor shapes does not meet reuqirements
 */
cnmlStatus_t cnmlCreatePluginDensityPriorBoxOp(
   cnmlBaseOp_t *op_ptr,
   cnmlPluginDensityPriorBoxOpParam_t param,
   cnmlTensor_t* input_tensors,
   cnmlTensor_t* output_tensors);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginDensityPriorBox on MLU.
 *
 *  **Supports MLU270**
 *
 *  @param[out]  op
 *    Output. A pointer to the base operator address.
 *  @param[in]  input_addrs
 *    Input. An array of MLU addrs,
 *           should be [feat_shape_mlu_ptr,img_shape_mlu_ptr]
 *  @param[in]  input_num
 *    Input. An int, must == 2
 *  @param[in]  output_addrs
 *    Input. An array of MLU addrs,
 *           should be [boxes_mlu_ptr,vars_mlu_ptr]
 *  @param[in]  output_num
 *    Input. An int, must == 2
 *  @param[in]  queue
 *    Input. A computation queue pointer.
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Base op is nullptr.
 *    - Input / output addrs is nullptr.
 *    - Input / output nums are inconsistent.
 */
cnmlStatus_t cnmlComputePluginDensityPriorBoxOpForward(
    cnmlBaseOp_t op,
    void **input_addrs,
    const int input_num,
    void **output_addrs,
    const int output_num,
    cnrtQueue_t queue);

/*!
 *  @brief A function.
 *
 *  This function forwards PluginBoxCoderOp on CPU.
 *
 *  @param[in]  boxes,
 *    Output. Adress of cpu box coord data
 *  @param[in]  vars
 *    Output. Adress of cpu box coord variance data
 *  @param[in] variances_ptr
 *    Input. Address for variance data
 *  @param[in] variances_num
 *    Input. len for variance data, must == 4
 *  @param[in] densities_ptr
 *    Input. Address for density data
 *  @param[in] densities_num
 *    Input. len for density  data
 *  @param[in] fixed_sizes_ptr
 *    Input. Address for fixed size data
 *  @param[in] fixed_sizes_num
 *    Input. len for fixed sizes data
 *  @param[in] fixed_ratios_ptr
 *    Input. Address for fixed ratio data
 *  @param[in] fixed_ratios_num
 *    Input. len for fixed ratio data
 *  @param[in] feature_width
 *    Input. An int, width of feature
 *  @param[in] feature_height
 *    Input. An int, height of feature
 *  @param[in] img_width
 *    Input. An int, width of image
 *  @param[in] img_height
 *    Input. An int, height of image
 *  @param[in] clip
 *    Input. clip coord data in 0 and 1 or not
 *  @param[in] step_w
 *    Input. step of prior box width, if step_w == 0 or step_h == 0, then it will be calculate automaticly
 *  @param[in] step_h
 *    Input. step of prior box height, if step_w == 0 or step_h == 0, then it will be calculate automaticly
 *  @param[in] offset
 *    Input. step of prior box center
 *  @retval CNML_STATUS_SUCCESS
 *    The function ends normally
 *  @retval CNML_STATUS_INVALIDPARAM
 *    At least one of the following conditions is not met:
 *    - Input / output addrs is nullptr or malloced with wrong sizes.
 */
cnmlStatus_t cnmlCpuComputePluginDensityPriorBoxOp(
    float * boxes,
    float * vars,
    float *variances_ptr,
    int variances_num,
    int *densities_ptr,
    int densities_num,
    float *fixed_sizes_ptr,
    int fixed_sizes_num,
    float *fixed_ratios_ptr,
    int fixed_ratios_num,
    int feature_width,
    int feature_height,
    int img_width,
    int img_height,
    bool clip,
    float step_w,
    float step_h,
    float offset);
/* --------------------------------------- */
/* cnmlPluginDensityPriorBox operation end */
/* --------------------------------------- */
#endif
