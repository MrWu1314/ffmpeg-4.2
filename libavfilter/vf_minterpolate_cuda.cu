/*
 * This file is part of FFmpeg.
 *
 * 编译方法:
 * nvcc -c -o vf_minterpolate_cuda.ptx vf_minterpolate_cuda.cu
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <climits>
#include <cuda_fp16.h>
#include <stdint.h>
#include <stddef.h>

extern "C" {

#define ALPHA_MAX 1024
#define COST_PRED_SCALE 64
#define NB_PIXEL_MVS 32
#define PX_WEIGHT_MAX 255

// OBMC权重表（与CPU版本相同）
static const __constant__ uint8_t obmc_linear16[256] = {
  0,  4,  4,  8,  8, 12, 12, 16, 16, 12, 12,  8,  8,  4,  4,  0,
  4,  8, 16, 20, 28, 32, 40, 44, 44, 40, 32, 28, 20, 16,  8,  4,
  4, 16, 24, 36, 44, 56, 64, 76, 76, 64, 56, 44, 36, 24, 16,  4,
  8, 20, 36, 48, 64, 76, 92,104,104, 92, 76, 64, 48, 36, 20,  8,
  8, 28, 44, 64, 80,100,116,136,136,116,100, 80, 64, 44, 28,  8,
 12, 32, 56, 76,100,120,144,164,164,144,120,100, 76, 56, 32, 12,
 12, 40, 64, 92,116,144,168,196,196,168,144,116, 92, 64, 40, 12,
 16, 44, 76,104,136,164,196,224,224,196,164,136,104, 76, 44, 16,
 16, 44, 76,104,136,164,196,224,224,196,164,136,104, 76, 44, 16,
 12, 40, 64, 92,116,144,168,196,196,168,144,116, 92, 64, 40, 12,
 12, 32, 56, 76,100,120,144,164,164,144,120,100, 76, 56, 32, 12,
  8, 28, 44, 64, 80,100,116,136,136,116,100, 80, 64, 44, 28,  8,
  8, 20, 36, 48, 64, 76, 92,104,104, 92, 76, 64, 48, 36, 20,  8,
  4, 16, 24, 36, 44, 56, 64, 76, 76, 64, 56, 44, 36, 24, 16,  4,
  4,  8, 16, 20, 28, 32, 40, 44, 44, 40, 32, 28, 20, 16,  8,  4,
  0,  4,  4,  8,  8, 12, 12, 16, 16, 12, 12,  8,  8,  4,  4,  0,
};

static __device__ int av_clip(int a, int amin, int amax) {
    if (a < amin) return amin;
    if (a > amax) return amax;
    return a;
}

static __device__ int FFABS(int x) {
    return x < 0 ? -x : x;
}

static __device__ int FFMAX(int a, int b) {
    return a > b ? a : b;
}

static __device__ int FFMIN(int a, int b) {
    return a < b ? a : b;
}

// mid_pred函数，预留供将来使用
// 目前未使用，高级运动估计可能需要
#if 0
static __device__ int mid_pred(int a, int b, int c) {
    if (a > b) {
        if (c > b) {
            if (c > a) return a;
            else return c;
        } else return b;
    } else {
        if (b > c) {
            if (c > a) return c;
            else return a;
        } else return b;
    }
}
#endif

// 六边形搜索模式（与CPU版本相同）
static const __constant__ int8_t hex2[6][2] = {{-2, 0}, {-1,-2}, {-1, 2}, { 1,-2}, { 1, 2}, { 2, 0}};
static const __constant__ int8_t hex4[16][2] = {
    {-4,-2}, {-4,-1}, {-4, 0}, {-4, 1}, {-4, 2},
    { 4,-2}, { 4,-1}, { 4, 0}, { 4, 1}, { 4, 2},
    {-2, 3}, { 0, 4}, { 2, 3}, {-2,-3}, { 0,-4}, { 2,-3}
};

// 计算OBMC的SAD（重叠块），用于双向ME
__device__ uint64_t get_sad_ob_cuda(uint8_t *data_cur, int cur_stride,
                                     uint8_t *data_ref, int ref_stride,
                                     int x, int y, int x_mv, int y_mv,
                                     int mb_size, int x_min, int x_max, int y_min, int y_max,
                                     int pred_x, int pred_y)
{
    int x_min_clip = x_min + mb_size / 2;
    int x_max_clip = x_max - mb_size / 2;
    int y_min_clip = y_min + mb_size / 2;
    int y_max_clip = y_max - mb_size / 2;
    
    int x_clip = av_clip(x, x_min_clip, x_max_clip);
    int y_clip = av_clip(y, y_min_clip, y_max_clip);
    int x_mv_clip = av_clip(x_mv, x_min_clip, x_max_clip);
    int y_mv_clip = av_clip(y_mv, y_min_clip, y_max_clip);
    
    int mv_x = x_mv - x;
    int mv_y = y_mv - y;
    
    uint64_t sad = 0;
    int i, j;
    
    for (j = -mb_size / 2; j < mb_size * 3 / 2; j++) {
        for (i = -mb_size / 2; i < mb_size * 3 / 2; i++) {
            int ref_x = x_mv_clip + i;
            int ref_y = y_mv_clip + j;
            int cur_x = x_clip + i;
            int cur_y = y_clip + j;
            
            int ref_idx = ref_y * ref_stride + ref_x;
            int cur_idx = cur_y * cur_stride + cur_x;
            sad += FFABS((int)data_ref[ref_idx] - (int)data_cur[cur_idx]);
        }
    }
    
    // 加上预测代价
    sad += (FFABS(mv_x - pred_x) + FFABS(mv_y - pred_y)) * COST_PRED_SCALE;
    
    return sad;
}

// 计算OBMC的SAD（重叠块）
__device__ uint64_t get_sbad_ob(uint8_t *data_cur, int cur_stride,
                                 uint8_t *data_next, int next_stride,
                                 int x, int y, int x_mv, int y_mv,
                                 int mb_size, int x_min, int x_max, int y_min, int y_max,
                                 int pred_x, int pred_y)
{
    int x_clip = av_clip(x, x_min + mb_size / 2, x_max - mb_size / 2);
    int y_clip = av_clip(y, y_min + mb_size / 2, y_max - mb_size / 2);
    int mv_x1 = x_mv - x;
    int mv_y1 = y_mv - y;
    int mv_x = av_clip(x_mv - x, -FFMIN(x_clip - x_min, x_max - x_clip), FFMIN(x_clip - x_min, x_max - x_clip));
    int mv_y = av_clip(y_mv - y, -FFMIN(y_clip - y_min, y_max - y_clip), FFMIN(y_clip - y_min, y_max - y_clip));
    uint64_t sbad = 0;
    
    for (int j = -mb_size / 2; j < mb_size * 3 / 2; j++) {
        for (int i = -mb_size / 2; i < mb_size * 3 / 2; i++) {
            int cur_idx = (y_clip + mv_y + j) * cur_stride + (x_clip + mv_x + i);
            int next_idx = (y_clip - mv_y + j) * next_stride + (x_clip - mv_x + i);
            sbad += FFABS((int)data_cur[cur_idx] - (int)data_next[next_idx]);
        }
    }
    
    return sbad + (FFABS(mv_x1 - pred_x) + FFABS(mv_y1 - pred_y)) * COST_PRED_SCALE;
}

// UMH运动估计核函数
__global__ void cuda_umh_motion_estimation(
    uint8_t *data_cur, int cur_stride,
    uint8_t *data_ref, int ref_stride,
    int16_t *mv_x_out, int16_t *mv_y_out,
    int width, int height, int mb_size, int search_param,
    int b_width, int b_height,
    int x_min, int x_max, int y_min, int y_max,
    int16_t *mv_table_prev0, int16_t *mv_table_prev1,
    int preds0_count, int16_t *preds0_mvs,
    int preds1_count, int16_t *preds1_mvs,
    int pred_x, int pred_y)
{
    int mb_x = blockIdx.x * blockDim.x + threadIdx.x;
    int mb_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (mb_x >= b_width || mb_y >= b_height) return;
    
    int x_mb = mb_x << (mb_size == 16 ? 4 : (mb_size == 8 ? 3 : 2));
    int y_mb = mb_y << (mb_size == 16 ? 4 : (mb_size == 8 ? 3 : 2));
    int mb_idx = mb_y * b_width + mb_x;
    
    // 计算搜索边界
    int x_min_search = FFMAX(x_min, x_mb - search_param);
    int y_min_search = FFMAX(y_min, y_mb - search_param);
    int x_max_search = FFMIN(x_mb + search_param, x_max);
    int y_max_search = FFMIN(y_mb + search_param, y_max);
    
    uint64_t cost_min = UINT64_MAX;
    int best_x = x_mb;
    int best_y = y_mb;
    uint64_t cost;
    
    // 计算代价的辅助宏
    #define GET_COST_UMH(x_val, y_val) \
        ((x_val) < x_min_search || (x_val) > x_max_search || \
         (y_val) < y_min_search || (y_val) > y_max_search) ? \
        UINT64_MAX : \
        get_sbad_ob(data_cur, cur_stride, data_ref, ref_stride, \
                   x_mb, y_mb, (x_val), (y_val), mb_size, \
                   x_min, x_max, y_min, y_max, \
                   pred_x, pred_y)
    
    // 先尝试中值预测器
    cost = GET_COST_UMH(x_mb + pred_x, y_mb + pred_y);
    if (cost < cost_min) {
        cost_min = cost;
        best_x = x_mb + pred_x;
        best_y = y_mb + pred_y;
    }
    
    // 尝试集合0的预测器
    for (int i = 0; i < preds0_count; i++) {
        int px = preds0_mvs[i * 2];
        int py = preds0_mvs[i * 2 + 1];
        cost = GET_COST_UMH(x_mb + px, y_mb + py);
        if (cost < cost_min) {
            cost_min = cost;
            best_x = x_mb + px;
            best_y = y_mb + py;
        }
    }
    
    // 尝试集合1的预测器
    for (int i = 0; i < preds1_count; i++) {
        int px = preds1_mvs[i * 2];
        int py = preds1_mvs[i * 2 + 1];
        cost = GET_COST_UMH(x_mb + px, y_mb + py);
        if (cost < cost_min) {
            cost_min = cost;
            best_x = x_mb + px;
            best_y = y_mb + py;
        }
    }
    
    // 非对称十字搜索
    int x = best_x;
    int y = best_y;
    for (int d = 1; d <= search_param; d += 2) {
        cost = GET_COST_UMH(x - d, y);
        if (cost < cost_min) {
            cost_min = cost;
            best_x = x - d;
            best_y = y;
        }
        cost = GET_COST_UMH(x + d, y);
        if (cost < cost_min) {
            cost_min = cost;
            best_x = x + d;
            best_y = y;
        }
        if (d <= search_param / 2) {
            cost = GET_COST_UMH(x, y - d);
            if (cost < cost_min) {
                cost_min = cost;
                best_x = x;
                best_y = y - d;
            }
            cost = GET_COST_UMH(x, y + d);
            if (cost < cost_min) {
                cost_min = cost;
                best_x = x;
                best_y = y + d;
            }
        }
    }
    
    x = best_x;
    y = best_y;
    
    // 非均匀多六边形网格搜索
    int end_x = FFMIN(x + 2, x_max_search);
    int end_y = FFMIN(y + 2, y_max_search);
    for (int y2 = FFMAX(y_min_search, y - 2); y2 <= end_y; y2++) {
        for (int x2 = FFMAX(x_min_search, x - 2); x2 <= end_x; x2++) {
            cost = GET_COST_UMH(x2, y2);
            if (cost < cost_min) {
                cost_min = cost;
                best_x = x2;
                best_y = y2;
            }
        }
    }
    
    x = best_x;
    y = best_y;
    
    // 扩展六边形搜索
    for (int d = 1; d <= search_param / 4; d++) {
        for (int i = 1; i < 16; i++) {
            cost = GET_COST_UMH(x + hex4[i][0] * d, y + hex4[i][1] * d);
            if (cost < cost_min) {
                cost_min = cost;
                best_x = x + hex4[i][0] * d;
                best_y = y + hex4[i][1] * d;
            }
        }
    }
    
    // 扩展六边形搜索（hex2模式）
    int prev_x;
    int prev_y;
    do {
        prev_x = x;
        prev_y = y;
        x = best_x;
        y = best_y;
        for (int i = 0; i < 6; i++) {
            cost = GET_COST_UMH(x + hex2[i][0], y + hex2[i][1]);
            if (cost < cost_min) {
                cost_min = cost;
                best_x = x + hex2[i][0];
                best_y = y + hex2[i][1];
            }
        }
    } while (x != prev_x || y != prev_y);
    
    // 菱形搜索细化
    static const int8_t dia1[4][2] = {{-1, 0}, {0, -1}, {1, 0}, {0, 1}};
    x = best_x;
    y = best_y;
    do {
        prev_x = x;
        prev_y = y;
        x = best_x;
        y = best_y;
        for (int i = 0; i < 4; i++) {
            cost = GET_COST_UMH(x + dia1[i][0], y + dia1[i][1]);
            if (cost < cost_min) {
                cost_min = cost;
                best_x = x + dia1[i][0];
                best_y = y + dia1[i][1];
            }
        }
    } while (x != prev_x || y != prev_y);
    
    #undef GET_COST_UMH
    
    mv_x_out[mb_idx] = best_x - x_mb;
    mv_y_out[mb_idx] = best_y - y_mb;
}

// 双向运动估计核函数（使用get_sad_ob）
__global__ void cuda_bidir_motion_estimation(
    uint8_t *data_cur, int cur_stride,
    uint8_t *data_ref, int ref_stride,
    int16_t *mv_x_out, int16_t *mv_y_out,
    int width, int height, int mb_size, int search_param,
    int b_width, int b_height,
    int x_min, int x_max, int y_min, int y_max,
    int16_t *mv_table_prev0, int16_t *mv_table_prev1,
    int preds0_count, int16_t *preds0_mvs,
    int preds1_count, int16_t *preds1_mvs,
    int pred_x, int pred_y, int method)
{
    int mb_x = blockIdx.x * blockDim.x + threadIdx.x;
    int mb_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (mb_x >= b_width || mb_y >= b_height) return;
    
    int x_mb = mb_x << (mb_size == 16 ? 4 : (mb_size == 8 ? 3 : 2));
    int y_mb = mb_y << (mb_size == 16 ? 4 : (mb_size == 8 ? 3 : 2));
    int mb_idx = mb_y * b_width + mb_x;
    
    // 计算搜索边界
    int x_min_search = FFMAX(x_min, x_mb - search_param);
    int y_min_search = FFMAX(y_min, y_mb - search_param);
    int x_max_search = FFMIN(x_mb + search_param, x_max);
    int y_max_search = FFMIN(y_mb + search_param, y_max);
    
    uint64_t cost_min = UINT64_MAX;
    int best_x = x_mb;
    int best_y = y_mb;
    uint64_t cost;
    
    // 使用get_sad_ob计算代价的辅助宏
    #define GET_COST_BIDIR(x_val, y_val) \
        ((x_val) < x_min_search || (x_val) > x_max_search || \
         (y_val) < y_min_search || (y_val) > y_max_search) ? \
        UINT64_MAX : \
        get_sad_ob_cuda(data_cur, cur_stride, data_ref, ref_stride, \
                       x_mb, y_mb, (x_val), (y_val), mb_size, \
                       x_min, x_max, y_min, y_max, \
                       pred_x, pred_y)
    
    // 先尝试中值预测器
    cost = GET_COST_BIDIR(x_mb + pred_x, y_mb + pred_y);
    if (cost < cost_min) {
        cost_min = cost;
        best_x = x_mb + pred_x;
        best_y = y_mb + pred_y;
    }
    
    // 尝试集合0的预测器
    for (int i = 0; i < preds0_count; i++) {
        int px = preds0_mvs[i * 2];
        int py = preds0_mvs[i * 2 + 1];
        cost = GET_COST_BIDIR(x_mb + px, y_mb + py);
        if (cost < cost_min) {
            cost_min = cost;
            best_x = x_mb + px;
            best_y = y_mb + py;
        }
    }
    
    // 尝试集合1的预测器
    for (int i = 0; i < preds1_count; i++) {
        int px = preds1_mvs[i * 2];
        int py = preds1_mvs[i * 2 + 1];
        cost = GET_COST_BIDIR(x_mb + px, y_mb + py);
        if (cost < cost_min) {
            cost_min = cost;
            best_x = x_mb + px;
            best_y = y_mb + py;
        }
    }
    
    // 应用搜索方法
    if (method == 9) { // AV_ME_METHOD_UMH（UMH方法）
        // 非对称十字搜索
        int x = best_x;
        int y = best_y;
        for (int d = 1; d <= search_param; d += 2) {
            cost = GET_COST_BIDIR(x - d, y);
            if (cost < cost_min) {
                cost_min = cost;
                best_x = x - d;
                best_y = y;
            }
            cost = GET_COST_BIDIR(x + d, y);
            if (cost < cost_min) {
                cost_min = cost;
                best_x = x + d;
                best_y = y;
            }
            if (d <= search_param / 2) {
                cost = GET_COST_BIDIR(x, y - d);
                if (cost < cost_min) {
                    cost_min = cost;
                    best_x = x;
                    best_y = y - d;
                }
                cost = GET_COST_BIDIR(x, y + d);
                if (cost < cost_min) {
                    cost_min = cost;
                    best_x = x;
                    best_y = y + d;
                }
            }
        }
        
        x = best_x;
        y = best_y;
        
        // 非均匀多六边形网格搜索
        int end_x = FFMIN(x + 2, x_max_search);
        int end_y = FFMIN(y + 2, y_max_search);
        for (int y2 = FFMAX(y_min_search, y - 2); y2 <= end_y; y2++) {
            for (int x2 = FFMAX(x_min_search, x - 2); x2 <= end_x; x2++) {
                cost = GET_COST_BIDIR(x2, y2);
                if (cost < cost_min) {
                    cost_min = cost;
                    best_x = x2;
                    best_y = y2;
                }
            }
        }
        
        x = best_x;
        y = best_y;
        
        // 扩展六边形搜索
        for (int d = 1; d <= search_param / 4; d++) {
            for (int i = 1; i < 16; i++) {
                cost = GET_COST_BIDIR(x + hex4[i][0] * d, y + hex4[i][1] * d);
                if (cost < cost_min) {
                    cost_min = cost;
                    best_x = x + hex4[i][0] * d;
                    best_y = y + hex4[i][1] * d;
                }
            }
        }
        
        // 扩展六边形搜索（hex2模式）
        int prev_x;
        int prev_y;
        do {
            prev_x = x;
            prev_y = y;
            x = best_x;
            y = best_y;
            for (int i = 0; i < 6; i++) {
                cost = GET_COST_BIDIR(x + hex2[i][0], y + hex2[i][1]);
                if (cost < cost_min) {
                    cost_min = cost;
                    best_x = x + hex2[i][0];
                    best_y = y + hex2[i][1];
                }
            }
        } while (x != prev_x || y != prev_y);
        
        // 菱形搜索细化
        static const int8_t dia1[4][2] = {{-1, 0}, {0, -1}, {1, 0}, {0, 1}};
        x = best_x;
        y = best_y;
        do {
            prev_x = x;
            prev_y = y;
            x = best_x;
            y = best_y;
            for (int i = 0; i < 4; i++) {
                cost = GET_COST_BIDIR(x + dia1[i][0], y + dia1[i][1]);
                if (cost < cost_min) {
                    cost_min = cost;
                    best_x = x + dia1[i][0];
                    best_y = y + dia1[i][1];
                }
            }
        } while (x != prev_x || y != prev_y);
    } else {
        // 其他方法使用菱形搜索细化
        static const int8_t dia1[4][2] = {{-1, 0}, {0, -1}, {1, 0}, {0, 1}};
        int x = best_x;
        int y = best_y;
        int prev_x;
        int prev_y;
        do {
            prev_x = x;
            prev_y = y;
            x = best_x;
            y = best_y;
            for (int i = 0; i < 4; i++) {
                cost = GET_COST_BIDIR(x + dia1[i][0], y + dia1[i][1]);
                if (cost < cost_min) {
                    cost_min = cost;
                    best_x = x + dia1[i][0];
                    best_y = y + dia1[i][1];
                }
            }
        } while (x != prev_x || y != prev_y);
    }
    
    #undef GET_COST_BIDIR
    
    mv_x_out[mb_idx] = best_x - x_mb;
    mv_y_out[mb_idx] = best_y - y_mb;
}

// EPZS运动估计核函数
__global__ void cuda_epzs_motion_estimation(
    uint8_t *data_cur, int cur_stride,
    uint8_t *data_ref, int ref_stride,
    int16_t *mv_x_out, int16_t *mv_y_out,
    int width, int height, int mb_size, int search_param,
    int b_width, int b_height,
    int x_min, int x_max, int y_min, int y_max,
    int16_t *mv_table_prev0, int16_t *mv_table_prev1,
    int preds0_count, int16_t *preds0_mvs,
    int preds1_count, int16_t *preds1_mvs,
    int pred_x, int pred_y)
{
    int mb_x = blockIdx.x * blockDim.x + threadIdx.x;
    int mb_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (mb_x >= b_width || mb_y >= b_height) return;
    
    int x_mb = mb_x << (mb_size == 16 ? 4 : (mb_size == 8 ? 3 : 2));
    int y_mb = mb_y << (mb_size == 16 ? 4 : (mb_size == 8 ? 3 : 2));
    int mb_idx = mb_y * b_width + mb_x;
    
    // 计算搜索边界
    int x_min_search = FFMAX(x_min, x_mb - search_param);
    int y_min_search = FFMAX(y_min, y_mb - search_param);
    int x_max_search = FFMIN(x_mb + search_param, x_max);
    int y_max_search = FFMIN(y_mb + search_param, y_max);
    
    uint64_t cost_min = UINT64_MAX;
    int best_x = x_mb;
    int best_y = y_mb;
    uint64_t cost;
    
    // 计算代价的辅助宏（旧版CUDA对lambda支持不好）
    #define GET_COST(x_val, y_val) \
        ((x_val) < x_min_search || (x_val) > x_max_search || \
         (y_val) < y_min_search || (y_val) > y_max_search) ? \
        UINT64_MAX : \
        get_sbad_ob(data_cur, cur_stride, data_ref, ref_stride, \
                   x_mb, y_mb, (x_val), (y_val), mb_size, \
                   x_min, x_max, y_min, y_max, \
                   pred_x, pred_y)
    
    // 先尝试中值预测器
    cost = GET_COST(x_mb + pred_x, y_mb + pred_y);
    if (cost < cost_min) {
        cost_min = cost;
        best_x = x_mb + pred_x;
        best_y = y_mb + pred_y;
    }
    
    // 尝试集合0的预测器
    for (int i = 0; i < preds0_count; i++) {
        int px = preds0_mvs[i * 2];
        int py = preds0_mvs[i * 2 + 1];
        cost = GET_COST(x_mb + px, y_mb + py);
        if (cost < cost_min) {
            cost_min = cost;
            best_x = x_mb + px;
            best_y = y_mb + py;
        }
    }
    
    // 尝试集合1的预测器
    for (int i = 0; i < preds1_count; i++) {
        int px = preds1_mvs[i * 2];
        int py = preds1_mvs[i * 2 + 1];
        cost = GET_COST(x_mb + px, y_mb + py);
        if (cost < cost_min) {
            cost_min = cost;
            best_x = x_mb + px;
            best_y = y_mb + py;
        }
    }
    
    // 菱形搜索细化
    static const int8_t dia1[4][2] = {{-1, 0}, {0, -1}, {1, 0}, {0, 1}};
    int x = best_x;
    int y = best_y;
    int prev_x, prev_y;
    do {
        prev_x = x;
        prev_y = y;
        x = best_x;
        y = best_y;
        for (int i = 0; i < 4; i++) {
            cost = GET_COST(x + dia1[i][0], y + dia1[i][1]);
            if (cost < cost_min) {
                cost_min = cost;
                best_x = x + dia1[i][0];
                best_y = y + dia1[i][1];
            }
        }
    } while (x != prev_x || y != prev_y);
    
    #undef GET_COST
    
    mv_x_out[mb_idx] = best_x - x_mb;
    mv_y_out[mb_idx] = best_y - y_mb;
}

// 计算场景切换检测的SAD
__global__ void cuda_scene_sad(
    uint8_t *frame1, int stride1,
    uint8_t *frame2, int stride2,
    uint64_t *sad_out,
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int diff = FFABS((int)frame1[y * stride1 + x] - (int)frame2[y * stride2 + x]);
    
    // 使用原子加累加
    atomicAdd((unsigned long long*)sad_out, (unsigned long long)diff);
}

// BLEND模式：简单帧混合
__global__ void cuda_blend_frames(
    uint8_t *output, int out_stride,
    uint8_t *frame1, int stride1,
    uint8_t *frame2, int stride2,
    int width, int height, int alpha)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * out_stride + x;
    int val = (alpha * frame2[idx] + (ALPHA_MAX - alpha) * frame1[idx] + 512) >> 10;
    output[idx] = av_clip(val, 0, 255);
}

// 双向OBMC：添加带权重的像素用于运动补偿
__global__ void cuda_bilateral_obmc(
    int16_t *mv_x, int16_t *mv_y,
    uint32_t *pixel_weights_out, int8_t *pixel_refs_out, int *pixel_nb_out,
    int16_t *pixel_mvs_x_out, int16_t *pixel_mvs_y_out,
    int width, int height, int mb_size, int b_width, int log2_mb_size,
    int mb_x_start, int mb_y_start, int mb_count_x, int mb_count_y,
    int alpha)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= mb_count_x * mb_count_y) return;
    
    int mb_idx = tid;
    int mb_x = mb_x_start + (mb_idx % mb_count_x);
    int mb_y = mb_y_start + (mb_idx / mb_count_x);
    int b_height = mb_count_y;
    
    if (mb_x >= b_width || mb_y >= b_height) return;
    
    int mv_idx = mb_y * b_width + mb_x;
    int mv_x_val = mv_x[mv_idx] * 2;
    int mv_y_val = mv_y[mv_idx] * 2;
    
    int x_mb = mb_x << log2_mb_size;
    int y_mb = mb_y << log2_mb_size;
    
    int start_x = x_mb - mb_size / 2;
    int start_y = y_mb - mb_size / 2;
    int startc_x = av_clip(start_x, 0, width - 1);
    int startc_y = av_clip(start_y, 0, height - 1);
    int endc_x = av_clip(start_x + (2 << log2_mb_size), 0, width);
    int endc_y = av_clip(start_y + (2 << log2_mb_size), 0, height);
    
    // mb_size=16时，log2_mb_size=4，使用obmc_linear16表
    // obmc_table_idx计算：x_in_block + (y_in_block << (log2_mb_size + 1))
    // 16x16块时，OBMC窗口为2*mb_size = 32x32
    // 表索引：x_offset + (y_offset << 5)，offset相对于块起始位置
    
    for (int y = startc_y; y < endc_y; y++) {
        for (int x = startc_x; x < endc_x; x++) {
            int pixel_idx = y * width + x;
            int x_in_block = x - start_x;
            int y_in_block = y - start_y;
            
            // 从表获取OBMC权重
            // mb_size=16时使用obmc_linear16，16x16=256项
            // 表覆盖2*mb_size x 2*mb_size区域（mb_size=16时为32x32）
            int obmc_table_idx = x_in_block + (y_in_block << (log2_mb_size + 1));
            int obmc_weight = 0;
            if (log2_mb_size == 4 && obmc_table_idx >= 0 && obmc_table_idx < 256) {  // mb_size=16
                obmc_weight = obmc_linear16[obmc_table_idx];
            }
            
            if (!obmc_weight) continue;
            
            // 获取当前像素引用数
            int nb = pixel_nb_out[pixel_idx];
            if (nb + 2 >= NB_PIXEL_MVS) continue;
            
            int x_min = -x;
            int x_max = width - x - 1;
            int y_min = -y;
            int y_max = height - y - 1;
            
            // 添加对frame 1的引用
            pixel_refs_out[pixel_idx * NB_PIXEL_MVS + nb] = 1;
            pixel_weights_out[pixel_idx * NB_PIXEL_MVS + nb] = obmc_weight * (ALPHA_MAX - alpha);
            pixel_mvs_x_out[pixel_idx * NB_PIXEL_MVS + nb] = av_clip((mv_x_val * alpha) / ALPHA_MAX, x_min, x_max);
            pixel_mvs_y_out[pixel_idx * NB_PIXEL_MVS + nb] = av_clip((mv_y_val * alpha) / ALPHA_MAX, y_min, y_max);
            nb++;
            
            // 添加对frame 2的引用
            pixel_refs_out[pixel_idx * NB_PIXEL_MVS + nb] = 2;
            pixel_weights_out[pixel_idx * NB_PIXEL_MVS + nb] = obmc_weight * alpha;
            pixel_mvs_x_out[pixel_idx * NB_PIXEL_MVS + nb] = av_clip(-mv_x_val * (ALPHA_MAX - alpha) / ALPHA_MAX, x_min, x_max);
            pixel_mvs_y_out[pixel_idx * NB_PIXEL_MVS + nb] = av_clip(-mv_y_val * (ALPHA_MAX - alpha) / ALPHA_MAX, y_min, y_max);
            nb++;
            
            pixel_nb_out[pixel_idx] = nb;
        }
    }
}

// 设置帧数据：带权重引用的像素级插值
__global__ void cuda_set_frame_data(
    uint8_t *output, int out_stride,
    uint8_t *frame1, int stride1,
    uint8_t *frame2, int stride2,
    uint32_t *pixel_weights, int8_t *pixel_refs, int *pixel_nb,
    int16_t *pixel_mvs_x, int16_t *pixel_mvs_y,
    int width, int height, int alpha,
    int chroma, int log2_chroma_w, int log2_chroma_h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int pixel_idx = y * width + x;
    int nb = pixel_nb[pixel_idx];
    
    int weight_sum = 0;
    int val = 0;
    
    // 计算权重和
    for (int i = 0; i < nb; i++) {
        weight_sum += pixel_weights[pixel_idx * NB_PIXEL_MVS + i];
    }
    
    // 无引用时使用默认混合
    if (!weight_sum || !nb) {
        int x_coord = chroma ? (x >> log2_chroma_w) : x;
        int y_coord = chroma ? (y >> log2_chroma_h) : y;
        int stride1_val = chroma ? stride1 : stride1;
        int stride2_val = chroma ? stride2 : stride2;
        
        int idx1 = x_coord + y_coord * stride1_val;
        int idx2 = x_coord + y_coord * stride2_val;
        
        val = (alpha * frame2[idx2] + (ALPHA_MAX - alpha) * frame1[idx1] + 512) >> 10;
    } else {
        // 加权插值
        for (int i = 0; i < nb; i++) {
            int ref = pixel_refs[pixel_idx * NB_PIXEL_MVS + i];
            int weight = pixel_weights[pixel_idx * NB_PIXEL_MVS + i];
            int mv_x_val = pixel_mvs_x[pixel_idx * NB_PIXEL_MVS + i];
            int mv_y_val = pixel_mvs_y[pixel_idx * NB_PIXEL_MVS + i];
            
            uint8_t *ref_frame = (ref == 1) ? frame1 : frame2;
            int ref_stride = (ref == 1) ? stride1 : stride2;
            
            int x_mv, y_mv;
            if (chroma) {
                x_mv = (x >> log2_chroma_w) + mv_x_val / (1 << log2_chroma_w);
                y_mv = (y >> log2_chroma_h) + mv_y_val / (1 << log2_chroma_h);
            } else {
                x_mv = x + mv_x_val;
                y_mv = y + mv_y_val;
            }
            
            int ref_idx = x_mv + y_mv * ref_stride;
            val += weight * ref_frame[ref_idx];
        }
        
        val = (val + weight_sum / 2) / weight_sum;
    }
    
    val = av_clip(val, 0, 255);
    
    int out_x = chroma ? (x >> log2_chroma_w) : x;
    int out_y = chroma ? (y >> log2_chroma_h) : y;
    output[out_x + out_y * out_stride] = val;
}

// 双向OBMC：添加带权重的像素用于运动补偿（双向模式）
__global__ void cuda_bidirectional_obmc(
    int16_t *mv_x_dir0, int16_t *mv_y_dir0,
    int16_t *mv_x_dir1, int16_t *mv_y_dir1,
    uint32_t *pixel_weights_out, int8_t *pixel_refs_out, int *pixel_nb_out,
    int16_t *pixel_mvs_x_out, int16_t *pixel_mvs_y_out,
    int width, int height, int mb_size, int b_width, int log2_mb_size,
    int mb_x_start, int mb_y_start, int mb_count_x, int mb_count_y,
    int alpha)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= mb_count_x * mb_count_y) return;
    
    int mb_idx = tid;
    int mb_x = mb_x_start + (mb_idx % mb_count_x);
    int mb_y = mb_y_start + (mb_idx / mb_count_x);
    int b_height = mb_count_y;
    
    if (mb_x >= b_width || mb_y >= b_height) return;
    
    // 处理两个方向
    // dir=0：frames[2]到frames[1]（前向），使用mv_x_dir0/mv_y_dir0
    // dir=1：frames[2]到frames[3]（后向），使用mv_x_dir1/mv_y_dir1
    for (int dir = 0; dir < 2; dir++) {
        int a = dir ? alpha : (ALPHA_MAX - alpha);
        int mv_idx = mb_y * b_width + mb_x;
        int mv_x_val, mv_y_val;
        
        // 获取该方向的运动矢量
        // CPU版本中frames[2-dir]表示：
        // dir=0：frames[2]（当前帧），mvs[0]指向frames[1]
        // dir=1：frames[1]（前一帧），mvs[1]指向frames[3]？
        // 实际上，CPU代码中frames[2].blocks存储两个方向
        // CUDA版本中dir0和dir1分开存储
        if (dir == 0) {
            mv_x_val = mv_x_dir0[mv_idx];
            mv_y_val = mv_y_dir0[mv_idx];
        } else {
            mv_x_val = mv_x_dir1[mv_idx];
            mv_y_val = mv_y_dir1[mv_idx];
        }
        
        int x_mb = mb_x << log2_mb_size;
        int y_mb = mb_y << log2_mb_size;
        
        // 计算运动补偿后的起始位置
        int start_x = x_mb - mb_size / 2 + mv_x_val * a / ALPHA_MAX;
        int start_y = y_mb - mb_size / 2 + mv_y_val * a / ALPHA_MAX;
        
        int startc_x = av_clip(start_x, 0, width - 1);
        int startc_y = av_clip(start_y, 0, height - 1);
        int endc_x = av_clip(start_x + (2 << log2_mb_size), 0, width);
        int endc_y = av_clip(start_y + (2 << log2_mb_size), 0, height);
        
        // dir=1时反转运动矢量方向
        if (dir == 1) {
            mv_x_val = -mv_x_val;
            mv_y_val = -mv_y_val;
        }
        
        // mb_size=16时使用obmc_linear16，16x16=256项
        for (int y = startc_y; y < endc_y; y++) {
            int y_min = -y;
            int y_max = height - y - 1;
            for (int x = startc_x; x < endc_x; x++) {
                int x_min = -x;
                int x_max = width - x - 1;
                int pixel_idx = y * width + x;
                int x_in_block = x - start_x;
                int y_in_block = y - start_y;
                
                // 从表获取OBMC权重
                // 类似CPU版本使用obmc_tab_linear[4 - log2_mb_size][...]
                // log2_mb_size=4（mb_size=16）时使用索引0
                int obmc_table_idx = x_in_block + (y_in_block << (log2_mb_size + 1));
                int obmc_weight = 0;
                if (log2_mb_size == 4 && obmc_table_idx >= 0 && obmc_table_idx < 256) {
                    obmc_weight = obmc_linear16[obmc_table_idx];
                }
                
                if (!obmc_weight) continue;
                
                // 获取当前像素引用数（需要原子操作保证线程安全）
                int nb = pixel_nb_out[pixel_idx];
                if (nb + 2 >= NB_PIXEL_MVS) continue;
                
                // 添加对frame 1的引用（前向引用）
                pixel_refs_out[pixel_idx * NB_PIXEL_MVS + nb] = 1;
                pixel_weights_out[pixel_idx * NB_PIXEL_MVS + nb] = obmc_weight * (ALPHA_MAX - alpha);
                pixel_mvs_x_out[pixel_idx * NB_PIXEL_MVS + nb] = av_clip((mv_x_val * alpha) / ALPHA_MAX, x_min, x_max);
                pixel_mvs_y_out[pixel_idx * NB_PIXEL_MVS + nb] = av_clip((mv_y_val * alpha) / ALPHA_MAX, y_min, y_max);
                nb++;
                
                // 添加对frame 2的引用（后向引用）
                pixel_refs_out[pixel_idx * NB_PIXEL_MVS + nb] = 2;
                pixel_weights_out[pixel_idx * NB_PIXEL_MVS + nb] = obmc_weight * alpha;
                pixel_mvs_x_out[pixel_idx * NB_PIXEL_MVS + nb] = av_clip(-mv_x_val * (ALPHA_MAX - alpha) / ALPHA_MAX, x_min, x_max);
                pixel_mvs_y_out[pixel_idx * NB_PIXEL_MVS + nb] = av_clip(-mv_y_val * (ALPHA_MAX - alpha) / ALPHA_MAX, y_min, y_max);
                nb++;
                
                pixel_nb_out[pixel_idx] = nb;
            }
        }
    }
}

} // extern "C"