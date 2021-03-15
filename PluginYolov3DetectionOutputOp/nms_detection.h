/*************************************************************************
 * Copyright (C) [2018] by Cambricon, Inc.
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *************************************************************************/

#ifndef _NMS_DETECTION_H_
#define _NMS_DETECTION_H_

//#define TIMER_PROCESS
//#define TIMER_KEEP
//#define TIMER_MODULE

//#define NMS_DEBUG_MLU
#ifdef NMS_DEBUG_MLU
#define LOG_VECTOR(statement, format, arr, len) \
  __bang_printf(statement);                     \
  __bang_printf("\n");                          \
  for (int idx = 0; idx < len; ++idx) {         \
    __bang_printf(format, *(arr + idx));        \
  }                                             \
  __bang_printf("\n\n");

#define LOGIF_VECTOR(condition, statement, format, arr, len) \
  if (condition) {                                           \
    LOG_VECTOR(statement, format, arr, len)                  \
  }

#define LOG_SCALAR(format, ...) __bang_printf(format, ##__VA_ARGS__)

#define LOGIF_SCALAR(condition, format, ...) \
  if (condition)                             \
  __bang_printf(format, ##__VA_ARGS__)

#define LOG_LINE(statement) __bang_printf(statement)
#else
#define LOG_VECTOR(...)
#define LOGIF_VECTOR(...)
#define LOG_SCALAR(...)
#define LOGIF_SCALAR(...)
#define LOG_LINE(...)
#endif

#define NMS_SIZE 64
#define NMS_UP(x, y) (x / y + (int)(x % y > 0)) * y
#define NMS_DOWN(x, y) (x / y) * y

enum Addr { NRAM, SRAM, GDRAM };
enum SplitMode { NMS_BLOCK = 1, NMS_U1 = 4 };

// max(x, y) ~ max(x - y, 0) + y
template <typename NMS_DT>
__mlu_func__ void __svmax_relu(NMS_DT *dst, NMS_DT *src0, NMS_DT *src1, int len) {
  __bang_sub(dst, src0, src1, len);
  __bang_active_relu(dst, dst, len);
  __bang_add(dst, dst, src1, len);
}

// min(x, y) ~ y - max(y - x, 0)
template <typename NMS_DT>
__mlu_func__ void __svmin_relu(NMS_DT *dst, NMS_DT *src0, NMS_DT *src1, int len) {
  __bang_sub(dst, src1, src0, len);
  __bang_active_relu(dst, dst, len);
  __bang_sub(dst, src1, dst, len);
}

/*!
 * @brief A function
 *
 * This function is nms module for detection serious, user can include this file
 * to call this function to select the boxes with nms, we support input from
 * GDRAM/SRAM/NRAM and output to GDRAM/SRAM/NRAM
 *
 * **Support MLU270**
 *
 * @param[out] output_box_num
 *   Output. The number of output box after nms.
 * @param[out] output_data
 *   Output. The output address to store the result.
 * @param[in] dst
 *   Input. The result store position, GDRAM/SRAM/NRAM.
 * @param[in] input_data_score
 *   Input. The input address where store the box score.
 * @param[in] input_data_box
 *   Input. The input address where store the box coordinate, the order must be
 *   x1, y1, x2, y2, same type data is stored together.
 * @param[in] src
 *   Input. The input data store position, GDRAM/SRAM/NRAM.
 * @param[in] buffer
 *   Input. The head pointer of nram for nms to compute.
 * @param[in] buffer_size
 *   Input. The size of nram above, the unit should be Byte, the value meet:
 *   if dst == GDRAM/SRAM && src == GDRAM/SRAM/NRAM
 *   at least: (64 * 9 + 64 + 256 * 5) * sizeof(NMS_DT)
 *   if dst == NRAM && src == GDRAM/SRAM/NRAM
 *   at least: (64 * 9 + 64) * sizeof(NMS_DT)
 *   of course, it's better when the buffer_size is bigger.
 *   note:
 *   if src == NRAM && input_box_num is pad, we can get the best performace when
 *   buffer_size meet:
 *   if dst == GDRAM/SRAM
 *   at least: (64 * 4 + 64 + 256 * 5) * sizeof(NMS_DT)
 *   if dst == NRAM
 *   at least: (64 * 4 + 64) * sizeof(NMS_DT)
 * @param[in] sram
 *   Input. The sram to find max score when U1, the size at least:
 *   30 * sizeof(NMS_DT)
 * @param[in] split_mode
 *   Input. The core number to compute nms, NMS_BLOCK/NMS_U1
 * @param[in] input_box_num
 *   Input. The input box number.
 * @param[in] input_stride
 *   Input. The input stride of two types of data.
 *   |---input_box_num---|
 *   |--------input_stride-------|
 *   score***************000000000
 *   x1******************000000000
 *   y1******************000000000
 *   x2******************000000000
 *   y2******************000000000
 * @param[in] output_stride
 *   Input. The output stride of two types of data.
 * @param[in] keepNum
 *   Input. The loop number of nms, the max number of selected box, if user don't
 *   have keepNum, let keepNum == input_box_num.
 * @param[in] thresh_iou
 *   Input. The threshold of iou to select the box.
 * @param[in] thresh_score
 *   Input. The threshold of score to break the loop.
 * @param[in] save_method
 *   Input. the save method after nms: 0/1/2
 *   0: save selected box score and box coordinate as:
 *   score,x1,y1,x2,y2|score,x1,y1,x2,y2|...
 *   1: save selected box score and box coordinate as:
 *   |--------output_stride-------|
 *   score******0000000000000000000
 *   x1*********0000000000000000000
 *   y1*********0000000000000000000
 *   x2*********0000000000000000000
 *   y2*********0000000000000000000
 *   2: when dst == NRAM, save selected box score only at original location.
 * @warning
 *   please ensure the params meet the demand
 */
template <typename NMS_DT>
__mlu_func__ void nms_detection(int &output_box_num,
                                NMS_DT *output_data,
                                Addr dst,
                                NMS_DT *input_data_score,
                                NMS_DT *input_data_box,
                                Addr src,
                                NMS_DT *buffer,
                                int buffer_size,
                                NMS_DT *sram,
                                SplitMode split_mode,
                                int input_box_num,
                                int input_stride,
                                int output_stride,
                                int keepNum,
                                NMS_DT thresh_iou,
                                NMS_DT thresh_score,
                                int save_method) {
  LOG_LINE("======NMS_MODULE================\n");

  struct timeval tstart;
  struct timeval tend;
  struct timeval loop_start;
  struct timeval loop_end;
  struct timeval pre_start;
  struct timeval pre_end;
  struct timeval findmax_start;
  struct timeval findmax_end;
  struct timeval store_start;
  struct timeval store_end;
  struct timeval load_start;
  struct timeval load_end;
  struct timeval compute_start;
  struct timeval compute_end;
  struct timeval update_start;
  struct timeval update_end;
  struct timeval keep_start;
  struct timeval keep_end;

#ifdef TIMER_MOUDLE
  gettimeofday(&tstart, NULL);
  gettimeofday(&pre_start, NULL);
#endif

  // global value
  int core_limit = split_mode;
  int32_t *loop_end_flag = (int32_t *)(sram + 28);  // for U1
  loop_end_flag[0] = 0;
  int nms_buffer_count1 = 9;
  int nms_buffer_count2 = 4;
  int nram_save_limit_count = 0;
  dst == NRAM ? nram_save_limit_count = 0 : nram_save_limit_count = 256;

  LOG_SCALAR("input_box_num:%d\n", input_box_num);
  LOG_SCALAR("thresh_iou:%hf\n", thresh_iou);
  LOG_SCALAR("thresh_score:%hf\n", thresh_score);
  LOG_SCALAR("core_limit:%d\n", core_limit);
  LOG_SCALAR("buffer_size:%d\n", buffer_size);
  LOG_SCALAR("nram_save_limit_count:%d\n", nram_save_limit_count);

  /******CHECK START*******/
  // 1、if output_box_num == 0
  if (output_box_num != 0)
    output_box_num = 0;
  // 2、if the core_limit is right
  if (core_limit != 1 && core_limit != 4)  // now only support block or U1
    return;
  // 3、ptr chack
  assert(input_data_score != output_data);
  assert(input_data_box != output_data);
  if (src == NRAM) {
    assert(input_data_score != buffer);
    assert(input_data_box != buffer);
  }
  // 4、if the buffer is enough
  assert(buffer_size >=
         (nms_buffer_count1 * 64 + 64 /*max_box*/ + (nram_save_limit_count * 5) * (dst != NRAM)) *
             sizeof(NMS_DT));
  // 5、check thresh
  assert(thresh_iou > 0 && thresh_iou < 1);
  assert(thresh_score >= 0 && thresh_score < 1);
  // 6、check the save_method
  if (save_method == 2) {  // for ops/detection_ssd
    assert(dst == NRAM && core_limit == 1);
  }
  /******CHECK END*******/

  // if the nram space is enough for compute directly when src is NRAM, MODE is 1
  // if not, MODE is 0
  int MODE = 0;  // 0: load data to buffer; 1: not load, compute directly
  if (src == NRAM) {
    int flag1 = (input_box_num == NMS_UP(input_box_num, NMS_SIZE));  // input_box_num must be pad
    int flag2 = (buffer_size > (nms_buffer_count2 * input_box_num + 64 /*max_box*/ +
                                (nram_save_limit_count * 5) * (dst != NRAM)) *
                                   sizeof(NMS_DT));  // buffer is enough
    if (flag1 && flag2)
      MODE = 1;
  }

  // input data ptr
  NMS_DT *input_score_ptr;
  NMS_DT *input_x1_ptr;
  NMS_DT *input_y1_ptr;
  NMS_DT *input_x2_ptr;
  NMS_DT *input_y2_ptr;
  input_score_ptr = input_data_score;
  input_x1_ptr = input_data_box;
  input_y1_ptr = input_x1_ptr + input_stride;
  input_x2_ptr = input_y1_ptr + input_stride;
  input_y2_ptr = input_x2_ptr + input_stride;

  // nram data ptr
  NMS_DT *x1;
  NMS_DT *y1;
  NMS_DT *x2;
  NMS_DT *y2;
  NMS_DT *score;
  NMS_DT *inter_x1;
  NMS_DT *inter_y1;
  NMS_DT *inter_x2;
  NMS_DT *inter_y2;
  NMS_DT *max_box;  // the max score, x1, y1, x2, y2
  NMS_DT *nram_save;

  int limit = 0;        // find limit when GDRAM or SRAM
  int len_core = 0;     // the length deal by every core
  int max_seg_pad = 0;  // the max length every repeat
  int repeat = 0;
  int remain = 0;
  int remain_pad = 0;
  int input_offset = 0;  // offset of input_data for current core
  int nram_save_count = 0;

  // data division
  if (src == NRAM) {
    if (MODE != 0) {
      repeat = 0;
      remain = input_box_num;
      remain_pad = remain;
    } else {
      limit = (buffer_size - 64 /*for max_box*/ * sizeof(NMS_DT) -
               nram_save_limit_count * 5 * sizeof(NMS_DT)) /
              (nms_buffer_count1 * sizeof(NMS_DT));
      len_core = input_box_num;
      input_offset = 0;
      max_seg_pad = NMS_DOWN(limit, NMS_SIZE);
      repeat = len_core / max_seg_pad;
      remain = len_core % max_seg_pad;
      remain_pad = NMS_UP(remain, NMS_SIZE);
    }
  } else {  // src id SRAM or GDRAM
    limit = (buffer_size - 64 /*for max_box*/ * sizeof(NMS_DT) -
             nram_save_limit_count * 5 * sizeof(NMS_DT)) /
            (nms_buffer_count1 * sizeof(NMS_DT));
    if (core_limit == 1) {
      len_core = input_box_num;
      input_offset = 0;
    } else {
      if (input_box_num % core_limit == 0) {
        len_core = input_box_num / core_limit;
        input_offset = coreId * len_core;
      } else {
        // multi core plan
        int avg_core = input_box_num / core_limit;
        int tmp = input_box_num % core_limit;
        coreId < tmp ? len_core = avg_core + 1 : len_core = avg_core;
        input_offset = avg_core * coreId + (coreId <= tmp ? coreId : tmp);
      }
    }
    max_seg_pad = NMS_DOWN(limit, NMS_SIZE);
    repeat = len_core / max_seg_pad;
    remain = len_core % max_seg_pad;
    remain_pad = NMS_UP(remain, NMS_SIZE);
  }

  LOG_SCALAR("MODE: %d\n", MODE);
  LOG_SCALAR("coreId: %d\n", coreId);
  LOG_SCALAR("limit: %d\n", limit);
  LOG_SCALAR("len_core: %d\n", len_core);
  LOG_SCALAR("max_seg_pad: %d\n", max_seg_pad);
  LOG_SCALAR("repeat: %d\n", repeat);
  LOG_SCALAR("remain: %d\n", remain);
  LOG_SCALAR("remain_pad: %d\n", remain_pad);
  LOG_SCALAR("input_offset: %d\n", input_offset);

  // init the data ptr
  if (src == NRAM && MODE != 0) {
    inter_x1 = buffer;
    inter_y1 = inter_x1 + input_box_num;
    inter_x2 = inter_y1 + input_box_num;
    inter_y2 = inter_x2 + input_box_num;
    max_box = inter_y2 + input_box_num;  // the max score, x1, y1, x2, y2
    nram_save = max_box + 64;
  } else {
    score = buffer;
    x1 = score + max_seg_pad;
    y1 = x1 + max_seg_pad;
    x2 = y1 + max_seg_pad;
    y2 = x2 + max_seg_pad;
    inter_x1 = y2 + max_seg_pad;
    inter_y1 = inter_x1 + max_seg_pad;
    inter_x2 = inter_y1 + max_seg_pad;
    inter_y2 = inter_x2 + max_seg_pad;
    max_box = inter_y2 + max_seg_pad;  // the max score, x1, y1, x2, y2
    nram_save = max_box + 64;
  }

#ifdef TIMER_MOUDLE
  gettimeofday(&pre_end, NULL);
  uint32_t pre_time_usec = (uint32_t)pre_end.tv_usec - (uint32_t)pre_start.tv_usec;
  printf("##########NMS MOUDLE pre Time: %u us\n", pre_time_usec);
#endif

#ifdef TIMER_MOUDLE
  gettimeofday(&loop_start, NULL);
#endif

  for (int keep = 0; keep < keepNum; keep++) {  // loop until the max_score <= 0

#ifdef TIMER_KEEP
    gettimeofday(&keep_start, NULL);
#endif

    if (core_limit != 1) {
      __sync_cluster();  // sync before current loop
    }

#ifdef TIMER_PROCESS
    gettimeofday(&findmax_start, NULL);
#endif
    /******FIND MAX START******/
    int max_index = 0;         // the max score index
    int global_max_index = 0;  // for U1
    NMS_DT max_area = 0;       // the max socre area
    max_box[0] = 0;            // init 0

    for (int i = 0; i <= repeat; i++) {
      if (i == repeat && remain == 0)
        break;
      int seg_len = 0;  // the length every nms compute
      int cpy_len = 0;  // the length every nms memcpy
      i == repeat ? seg_len = remain_pad : seg_len = max_seg_pad;
      i == repeat ? cpy_len = remain : cpy_len = max_seg_pad;

      LOGIF_SCALAR(keep == 0, "seg_len: %d\n", seg_len);
      LOGIF_SCALAR(keep == 0, "cpy_len: %d\n", cpy_len);

      /******NMS LOAD START******/
      if (MODE != 0) {
        score = input_score_ptr;
      } else {
        mluMemcpyDirection_t load_dir = SRAM2NRAM;
        if (src == NRAM) {
          load_dir = NRAM2NRAM;
        } else if (src == SRAM) {
          load_dir = SRAM2NRAM;
        } else {
          load_dir = GDRAM2NRAM;
        }
        __nramset(score, seg_len, 0);
        __memcpy(score, input_score_ptr + input_offset + i * max_seg_pad, cpy_len * sizeof(NMS_DT),
                 load_dir, cpy_len * sizeof(NMS_DT), cpy_len * sizeof(NMS_DT), 0);
      }

      LOGIF_VECTOR(keep == 0, "---score for find max---\n", "%hf ", score, 100);
      /******NMS LOAD END******/

      __bang_max(inter_x1, score, seg_len);
      if (inter_x1[0] > max_box[0]) {
        max_box[0] = inter_x1[0];
        max_index = ((unsigned short *)inter_x1)[1] * (sizeof(NMS_DT) == 2) +
                    ((unsigned int *)inter_x1)[1] * (sizeof(NMS_DT) == 4) + input_offset +
                    i * max_seg_pad;  // offset start from head of input_data
      }
    }  // for repeat

    LOGIF_SCALAR(keep == 0, "max_score_this_core: %hf\n", max_box[0]);
    LOGIF_SCALAR(keep == 0, "max_index_this_core: %d\n", max_index);

    if (core_limit == 1) {
      max_box[1] = input_x1_ptr[max_index];
      max_box[2] = input_y1_ptr[max_index];
      max_box[3] = input_x2_ptr[max_index];
      max_box[4] = input_y2_ptr[max_index];
      max_area = (max_box[3] - max_box[1]) * (max_box[4] - max_box[2]);
      input_score_ptr[max_index] = 0;
      global_max_index = max_index;
    } else if (core_limit == 4) {
      // find the max with sram
      // the max box's x1, y1, x2, y2 on every core
      max_box[1] = input_x1_ptr[max_index];
      max_box[2] = input_y1_ptr[max_index];
      max_box[3] = input_x2_ptr[max_index];
      max_box[4] = input_y2_ptr[max_index];
      ((int32_t *)(max_box + 5))[0] = max_index;
      // copy every core's box info to sram, form: score---x1---y1---x2---y2---
      for (int i = 0; i < 5; i++) {
        __memcpy(sram + i * core_limit + coreId, max_box + i, 1 * sizeof(NMS_DT), NRAM2SRAM);
      }
      // copy every core's max_index to sram, use 2 half to store max_index
      __memcpy(sram + 5 * core_limit + coreId * 2, max_box + 5, 2 * sizeof(NMS_DT), NRAM2SRAM);
      __sync_cluster();

      LOGIF_SCALAR(keep == 0, "sram_score:%hf\n", sram[coreId]);
      LOGIF_SCALAR(keep == 0, "sram_x1:%hf\n", sram[1 * core_limit + coreId]);
      LOGIF_SCALAR(keep == 0, "sram_y1:%hf\n", sram[2 * core_limit + coreId]);
      LOGIF_SCALAR(keep == 0, "sram_x2:%hf\n", sram[3 * core_limit + coreId]);
      LOGIF_SCALAR(keep == 0, "sram_y2:%hf\n", sram[4 * core_limit + coreId]);
      LOGIF_SCALAR(keep == 0, "sram_index:%d\n", ((int32_t *)(sram + 5 * core_limit))[coreId]);

      // copy score from sram to nram and find the max
      __nramset(inter_x1, 64, 0);
      __memcpy(inter_x1, sram, core_limit * sizeof(NMS_DT), SRAM2NRAM);
      __bang_max(max_box, inter_x1, 64);

      int max_core = ((unsigned short *)max_box)[1] * (sizeof(NMS_DT) == 2) +
                     ((unsigned int *)max_box)[1] * (sizeof(NMS_DT) == 4);
      // copy the max box to max_box
      __memcpy(max_box + 1, sram + 1 * core_limit + max_core, 1 * sizeof(NMS_DT), SRAM2NRAM);
      __memcpy(max_box + 2, sram + 2 * core_limit + max_core, 1 * sizeof(NMS_DT), SRAM2NRAM);
      __memcpy(max_box + 3, sram + 3 * core_limit + max_core, 1 * sizeof(NMS_DT), SRAM2NRAM);
      __memcpy(max_box + 4, sram + 4 * core_limit + max_core, 1 * sizeof(NMS_DT), SRAM2NRAM);
      __memcpy(max_box + 5, sram + 5 * core_limit + 2 * max_core, 2 * sizeof(NMS_DT), SRAM2NRAM);
      max_area = (max_box[3] - max_box[1]) * (max_box[4] - max_box[2]);
      global_max_index = ((int32_t *)(max_box + 5))[0];
      if (src != NRAM) {
        input_score_ptr[global_max_index] = 0;
      } else {
        if (coreId == max_core) {
          input_score_ptr[global_max_index] = 0;
        }
      }
    }
    // by now, we get: max_score|max_index|max_box|max_area
    LOGIF_SCALAR(keep == 0, "max_score:%hf\n", max_box[0]);
    LOGIF_SCALAR(keep == 0, "max_x1: %hf\n", max_box[1]);
    LOGIF_SCALAR(keep == 0, "max_y1: %hf\n", max_box[2]);
    LOGIF_SCALAR(keep == 0, "max_x2: %hf\n", max_box[3]);
    LOGIF_SCALAR(keep == 0, "max_y2: %hf\n", max_box[4]);
    LOGIF_SCALAR(keep == 0, "max_area: %hf\n", max_area);
    LOGIF_SCALAR(keep == 0, "global_max_index: %d\n", global_max_index);
/******FIND MAX END******/
#ifdef TIMER_PROCESS
    gettimeofday(&findmax_end, NULL);
    uint32_t findmax_time_usec = (uint32_t)findmax_end.tv_usec - (uint32_t)findmax_start.tv_usec;
    printf("------------------\n");
    printf("###nms findmax Time: %u us\n", findmax_time_usec);
#endif

#ifdef TIMER_PROCESS
    gettimeofday(&store_start, NULL);
#endif
    /******NMS STORE START******/
    // store to sram/gdram
    if (dst != NRAM && output_box_num != 0) {
      mluMemcpyDirection_t store_dir = NRAM2GDRAM;
      if (dst == SRAM) {
        store_dir = NRAM2SRAM;
      } else {  // dst == GDRAM
        store_dir = NRAM2GDRAM;
      }

      if ((nram_save_count == nram_save_limit_count) || (max_box[0] <= thresh_score)) {
        if (core_limit == 1) {
          if (save_method == 0) {  // score, x1, y1, x2, y2
            __memcpy(output_data, nram_save, nram_save_count * 5 * sizeof(NMS_DT), store_dir);
            output_data += nram_save_count * 5;
          } else {  // score---, x1---, y1---, x2---, y2---
            __memcpy(output_data, nram_save, nram_save_count * sizeof(NMS_DT), store_dir,
                     output_stride * sizeof(NMS_DT), nram_save_limit_count * sizeof(NMS_DT), 4);
            output_data += nram_save_count;
          }
          nram_save_count = 0;
        } else {
          if (coreId == coreDim - 1) {
            if (save_method == 0) {  // score, x1, y1, x2, y2
              __memcpy(output_data, nram_save, nram_save_count * 5 * sizeof(NMS_DT), store_dir);
              output_data += nram_save_count * 5;
            } else {  // score---, x1---, y1---, x2---, y2---
              __memcpy(output_data, nram_save, nram_save_count * sizeof(NMS_DT), store_dir,
                       output_stride * sizeof(NMS_DT), nram_save_limit_count * sizeof(NMS_DT), 4);
              output_data += nram_save_count;
            }
            nram_save_count = 0;
          }
        }  // if core_limit
      }    // if move data nram->sram/gdram
    }      // if dst

    // if the max score <= 0, end
    if (core_limit == 1) {
      if (max_box[0] <= thresh_score) {
        break;
      }
    } else {
      if (max_box[0] <= thresh_score) {
        if (coreId == coreDim - 1) {
          loop_end_flag[0] = 1;
        }
      }
      __sync_cluster();  // wait for update loop_end_flag
      if (loop_end_flag[0] == 1) {
        break;
      }
    }

    // store to nram
    NMS_DT *save_ptr;
    int save_offset = 0;
    int save_str_num = 0;
    if (dst == NRAM) {
      save_ptr = output_data;
      save_offset = output_box_num;
      save_str_num = input_box_num;
    } else {
      save_ptr = nram_save;
      save_offset = nram_save_count;
      save_str_num = nram_save_limit_count;
    }
    if (core_limit == 1) {
      if (save_method == 0) {  // score, x1, y1, x2, y2
        __memcpy(save_ptr + save_offset * 5, max_box, 5 * sizeof(NMS_DT), NRAM2NRAM,
                 5 * sizeof(NMS_DT), 5 * sizeof(NMS_DT), 0);
      } else if (save_method == 1) {  // score---, x1---, y1---, x2---, y2---
        __memcpy(save_ptr + save_offset, max_box, 1 * sizeof(NMS_DT), NRAM2NRAM,
                 save_str_num * sizeof(NMS_DT), 1 * sizeof(NMS_DT), 4);
      } else if (save_method == 2) {  // for ssd
        save_ptr[max_index] = max_box[0];
      }
    } else {
      if (coreId == coreDim - 1) {
        if (save_method == 0) {  // score, x1, y1, x2, y2
          __memcpy(save_ptr + save_offset * 5, max_box, 5 * sizeof(NMS_DT), NRAM2NRAM,
                   5 * sizeof(NMS_DT), 5 * sizeof(NMS_DT), 0);
        } else {  // score---, x1---, y1---, x2---, y2---
          __memcpy(save_ptr + save_offset, max_box, 1 * sizeof(NMS_DT), NRAM2NRAM,
                   save_str_num * sizeof(NMS_DT), 1 * sizeof(NMS_DT), 4);
        }
      }
    }
    nram_save_count++;
    output_box_num++;

    // store to sram/gdram --if keep == keepNum
    if (dst != NRAM && output_box_num != 0) {
      mluMemcpyDirection_t store_dir = NRAM2GDRAM;
      if (dst == SRAM) {
        store_dir = NRAM2SRAM;
      } else {  // dst == GDRAM
        store_dir = NRAM2GDRAM;
      }

      if (keep == keepNum) {
        if (core_limit == 1) {
          if (save_method == 0) {  // score, x1, y1, x2, y2
            __memcpy(output_data, nram_save, nram_save_count * 5 * sizeof(NMS_DT), store_dir);
          } else {  // score---, x1---, y1---, x2---, y2---
            __memcpy(output_data, nram_save, nram_save_count * sizeof(NMS_DT), store_dir,
                     output_stride * sizeof(NMS_DT), nram_save_limit_count * sizeof(NMS_DT), 4);
          }
        } else {
          if (coreId == coreDim - 1) {
            if (save_method == 0) {  // score, x1, y1, x2, y2
              __memcpy(output_data, nram_save, nram_save_count * 5 * sizeof(NMS_DT), store_dir);
            } else {  // score---, x1---, y1---, x2---, y2---
              __memcpy(output_data, nram_save, nram_save_count * sizeof(NMS_DT), store_dir,
                       output_stride * sizeof(NMS_DT), nram_save_limit_count * sizeof(NMS_DT), 4);
            }
          }
        }  // if core_limit
      }    // if move data nram->sram/gdram
    }      // if dst
           /******NMS STORE END******/
#ifdef TIMER_PROCESS
    gettimeofday(&store_end, NULL);
    uint32_t store_time_usec = (uint32_t)store_end.tv_usec - (uint32_t)store_start.tv_usec;
    printf("###nms store Time: %u us\n", store_time_usec);
#endif

    for (int i = 0; i <= repeat; i++) {
      if (i == repeat && remain == 0)
        break;
      int seg_len = 0;  // the length every nms compute
      int cpy_len = 0;  // the length every nms memcpy
      i == repeat ? seg_len = remain_pad : seg_len = max_seg_pad;
      i == repeat ? cpy_len = remain : cpy_len = max_seg_pad;

      LOGIF_SCALAR(keep == 0, "seg_len: %d\n", seg_len);
      LOGIF_SCALAR(keep == 0, "cpy_len: %d\n", cpy_len);

#ifdef TIMER_PROCESS
      gettimeofday(&load_start, NULL);
#endif
      /******NMS LOAD START******/
      if (MODE != 0) {
        score = input_score_ptr;
        x1 = input_x1_ptr;
        y1 = input_y1_ptr;
        x2 = input_x2_ptr;
        y2 = input_y2_ptr;
      } else {
        mluMemcpyDirection_t load_dir = SRAM2NRAM;
        if (src == NRAM) {
          load_dir = NRAM2NRAM;
        } else if (src == SRAM) {
          load_dir = SRAM2NRAM;
        } else {
          load_dir = GDRAM2NRAM;
        }
        __nramset(score, seg_len, 0);
        __memcpy(score, input_score_ptr + input_offset + i * max_seg_pad, cpy_len * sizeof(NMS_DT),
                 load_dir, cpy_len * sizeof(NMS_DT), cpy_len * sizeof(NMS_DT), 0);
        __memcpy(x1, input_x1_ptr + input_offset + i * max_seg_pad, cpy_len * sizeof(NMS_DT),
                 load_dir, cpy_len * sizeof(NMS_DT), cpy_len * sizeof(NMS_DT), 0);
        __memcpy(y1, input_y1_ptr + input_offset + i * max_seg_pad, cpy_len * sizeof(NMS_DT),
                 load_dir, cpy_len * sizeof(NMS_DT), cpy_len * sizeof(NMS_DT), 0);
        __memcpy(x2, input_x2_ptr + input_offset + i * max_seg_pad, cpy_len * sizeof(NMS_DT),
                 load_dir, cpy_len * sizeof(NMS_DT), cpy_len * sizeof(NMS_DT), 0);
        __memcpy(y2, input_y2_ptr + input_offset + i * max_seg_pad, cpy_len * sizeof(NMS_DT),
                 load_dir, cpy_len * sizeof(NMS_DT), cpy_len * sizeof(NMS_DT), 0);
      }

      LOGIF_VECTOR(keep == 0, "---score for compute---\n", "%hf ", score, 100);
      LOGIF_VECTOR(keep == 0, "---x1 for compute---\n", "%hf ", x1, 100);
      LOGIF_VECTOR(keep == 0, "---y1 for compute---\n", "%hf ", y1, 100);
      LOGIF_VECTOR(keep == 0, "---x2 for compute---\n", "%hf ", x2, 100);
      LOGIF_VECTOR(keep == 0, "---y2 for compute---\n", "%hf ", y2, 100);
/******NMS LOAD END******/
#ifdef TIMER_PROCESS
      gettimeofday(&load_end, NULL);
      uint32_t load_time_usec = (uint32_t)load_end.tv_usec - (uint32_t)load_start.tv_usec;
      printf("###nms load Time: %u us\n", load_time_usec);
#endif

#ifdef TIMER_PROCESS
      gettimeofday(&compute_start, NULL);
#endif
      /******NMS COMPUTE START******/
      // 0、pre, set the tail to zero when MODE == 1
      if (MODE == 1 && input_stride > input_box_num) {
        __nramset(inter_x1, seg_len, 0);
        int tail_len = input_stride - input_box_num;
        __memcpy(input_score_ptr + input_box_num, inter_x1, tail_len * sizeof(NMS_DT), NRAM2NRAM,
                 tail_len * sizeof(NMS_DT), tail_len * sizeof(NMS_DT), 0);
      }
      // 1、 compute IOU
      // get the area_I
      __nramset(inter_y1, seg_len, max_box[1]);       // max_x1
      __svmax_relu(inter_x1, x1, inter_y1, seg_len);  // inter_x1
      __nramset(inter_y2, seg_len, max_box[3]);       // max_x2
      __svmin_relu(inter_x2, x2, inter_y2, seg_len);  // inter_x2
      __bang_sub(inter_x1, inter_x2, inter_x1, seg_len);
      __bang_active_relu(inter_x1, inter_x1, seg_len);  // inter_w
      __nramset(inter_x2, seg_len, max_box[2]);         // max_y1
      __svmax_relu(inter_y1, y1, inter_x2, seg_len);    // inter_y1
      __nramset(inter_x2, seg_len, max_box[4]);         // max_y2
      __svmin_relu(inter_y2, y2, inter_x2, seg_len);    // inter_y2
      __bang_sub(inter_y1, inter_y2, inter_y1, seg_len);
      __bang_active_relu(inter_y1, inter_y1, seg_len);    // inter_h
      __bang_mul(inter_x1, inter_x1, inter_y1, seg_len);  // area_I
      LOGIF_VECTOR(keep == 0, "---area_I in compute---\n", "%hf ", inter_x1, 100);
      // get the area of input_box: area = (x2 - x1) * (y2 - y1);
      __bang_sub(inter_y1, x2, x1, seg_len);
      __bang_sub(inter_y2, y2, y1, seg_len);
      __bang_mul(inter_x2, inter_y1, inter_y2, seg_len);  // area
      LOGIF_VECTOR(keep == 0, "---area in compute---\n", "%hf ", inter_x2, 100);
      // get the area_U: area + max_area - area_I
      __nramset(inter_y1, seg_len, max_area);
      __bang_add(inter_x2, inter_x2, inter_y1, seg_len);
      __bang_sub(inter_x2, inter_x2, inter_x1, seg_len);  // area_U
      LOGIF_VECTOR(keep == 0, "---area_U in compute---\n", "%hf ", inter_x2, 100);
      // 2、 select the box
      // if IOU greater than thres, set the score to zero, abort it: area_U * thresh > area_I?
      __bang_mul_const(inter_x2, inter_x2, thresh_iou, seg_len);
      __bang_gt(inter_x1, inter_x2, inter_x1, seg_len);
      __bang_mul(score, score, inter_x1, seg_len);
/******NMS COMPUTE END******/
#ifdef TIMER_PROCESS
      gettimeofday(&compute_end, NULL);
      uint32_t compute_time_usec = (uint32_t)compute_end.tv_usec - (uint32_t)compute_start.tv_usec;
      printf("###nms compute Time: %u us\n", compute_time_usec);
#endif

#ifdef TIMER_PROCESS
      gettimeofday(&update_start, NULL);
#endif
      // update the score
      if (MODE == 0) {  // do nothing when MODE = 1
        mluMemcpyDirection_t update_dir = NRAM2SRAM;
        if (src == NRAM) {
          update_dir = NRAM2NRAM;
        } else if (src == SRAM) {
          update_dir = NRAM2SRAM;
        } else {
          update_dir = NRAM2GDRAM;
        }
        __memcpy(input_score_ptr + input_offset + i * max_seg_pad, score, cpy_len * sizeof(NMS_DT),
                 update_dir, cpy_len * sizeof(NMS_DT), cpy_len * sizeof(NMS_DT), 0);
      }
#ifdef TIMER_PROCESS
      gettimeofday(&update_end, NULL);
      uint32_t update_time_usec = (uint32_t)update_end.tv_usec - (uint32_t)update_start.tv_usec;
      printf("###nms update Time: %u us\n", update_time_usec);
      printf("------------------\n");
#endif

    }  // for repeat
#ifdef TIMER_KEEP
    gettimeofday(&keep_end, NULL);
    uint32_t keep_time_usec = (uint32_t)keep_end.tv_usec - (uint32_t)keep_start.tv_usec;
    printf("$$$nms keep num: %d --- keep Time: %u us\n", keep, keep_time_usec);
#endif
  }  // for keepNum
#ifdef TIMER_MOUDLE
  gettimeofday(&loop_end, NULL);
  gettimeofday(&tend, NULL);
#endif

#ifdef TIMER_MOUDLE
  uint32_t loop_time_usec = (uint32_t)loop_end.tv_usec - (uint32_t)loop_start.tv_usec;
  printf("##########NMS MOUDLE loop Total Time: %u us\n", loop_time_usec);
  uint32_t time_usec = (uint32_t)tend.tv_usec - (uint32_t)tstart.tv_usec;
  printf("NMS MOUDLE coreId: %d --- Hardware Total Time: %u us\n", coreId, time_usec);
#endif
}

#endif  // _NMS_DETECTION_H_
