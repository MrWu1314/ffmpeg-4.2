/*
 * Copyright (c) 2024
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#include <stdio.h>
#include <string.h>

#include "libavutil/avstring.h"
#include "libavutil/common.h"
#include "libavutil/hwcontext.h"
#include "libavutil/hwcontext_cuda_internal.h"
#include "libavutil/internal.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"
#include "libavutil/imgutils.h"

#include "avfilter.h"
#include "formats.h"
#include "internal.h"
#include "video.h"
#include "motion_estimation.h"
#include "libavcodec/mathops.h"
#include "libavutil/mathematics.h"

static const enum AVPixelFormat supported_formats[] = {
    AV_PIX_FMT_YUV420P,
    AV_PIX_FMT_NV12,
    AV_PIX_FMT_YUV420P10,
};

#define DIV_UP(a, b) ( ((a) + (b) - 1) / (b) )
#define ALIGN_UP(a, b) (((a) + (b) - 1) & ~((b) - 1))
#define BLOCKX 32
#define BLOCKY 16

#define CHECK_CU(x) x

#define ME_MODE_BIDIR 0
#define ME_MODE_BILAT 1

#define MC_MODE_OBMC 0

#define SCD_METHOD_FDIFF 1

#define NB_FRAMES 4
#define NB_PIXEL_MVS 32
#define ALPHA_MAX 1024
#define COST_PRED_SCALE 64
#define PX_WEIGHT_MAX 255

enum MIMode {
    MI_MODE_DUP         = 0,
    MI_MODE_BLEND       = 1,
    MI_MODE_MCI         = 2,
};

typedef struct CUDAMInterpolateContext {
    const AVClass *class;

    AVCUDADeviceContext *hwctx;

    AVBufferRef *frames_ctx;
    AVFrame     *frame;
    AVFrame *tmp_frame;

    enum AVPixelFormat format;
    enum AVPixelFormat pix_fmt;

    AVRational frame_rate;
    enum MIMode mi_mode;
    int mc_mode;
    int me_mode;
    int me_method;
    int mb_size;
    int search_param;
    int scd_method;
    double scd_threshold;
    
    AVFrame *frames[NB_FRAMES];
    int64_t out_pts;
    
    int width;
    int height;
    int log2_chroma_w;
    int log2_chroma_h;
    int nb_planes;

    CUcontext   cu_ctx;
    CUmodule    cu_module;
    CUfunction  cu_func_epzs_me;
    CUfunction  cu_func_umh_me;
    CUfunction  cu_func_bidir_me;
    CUfunction  cu_func_scene_sad;
    CUfunction  cu_func_blend;
    CUfunction  cu_func_bilateral_obmc;
    CUfunction  cu_func_bidirectional_obmc;
    CUfunction  cu_func_set_frame_data;

    // 运动矢量和像素数据的设备内存
    CUdeviceptr cu_mv_x;
    CUdeviceptr cu_mv_y;
    CUdeviceptr cu_mv_x_dir0;  // 双向运动估计方向0
    CUdeviceptr cu_mv_y_dir0;
    CUdeviceptr cu_mv_x_dir1;  // 双向运动估计方向1
    CUdeviceptr cu_mv_y_dir1;
    CUdeviceptr cu_mv_table_prev0;
    CUdeviceptr cu_mv_table_prev1;
    CUdeviceptr cu_pixel_weights;
    CUdeviceptr cu_pixel_refs;
    CUdeviceptr cu_pixel_nb;
    CUdeviceptr cu_pixel_mvs_x;
    CUdeviceptr cu_pixel_mvs_y;
    CUdeviceptr cu_scene_sad;
    int mv_width;
    int mv_height;
    int b_width;
    int b_height;
    int b_count;
    int log2_mb_size;
    
    // EPZS预测器
    int16_t *preds0_mvs;
    int16_t *preds1_mvs;
    int preds0_count;
    int preds1_count;
    
    // 场景切换检测
    double prev_mafd;
    int scene_changed;
} CUDAMInterpolateContext;


static av_cold int cudaminterpolate_init(AVFilterContext *ctx)
{
    CUDAMInterpolateContext *s = ctx->priv;
    int i;

    s->frame = av_frame_alloc();
    if (!s->frame)
        return AVERROR(ENOMEM);

    s->tmp_frame = av_frame_alloc();
    if (!s->tmp_frame)
        return AVERROR(ENOMEM);

    for (i = 0; i < NB_FRAMES; i++) {
        s->frames[i] = NULL;
    }

    s->out_pts = AV_NOPTS_VALUE;

    return 0;
}

static av_cold void cudaminterpolate_uninit(AVFilterContext *ctx)
{
    CUDAMInterpolateContext *s = ctx->priv;
    int i;
    
    if (s->hwctx && s->cu_module) {
        CudaFunctions *cu = s->hwctx->internal->cuda_dl;
        CUcontext dummy;
        
        CHECK_CU(cu->cuCtxPushCurrent(s->hwctx->cuda_ctx));
        
        if (s->cu_mv_x) CHECK_CU(cu->cuMemFree(s->cu_mv_x));
        if (s->cu_mv_y) CHECK_CU(cu->cuMemFree(s->cu_mv_y));
        if (s->cu_mv_x_dir0) CHECK_CU(cu->cuMemFree(s->cu_mv_x_dir0));
        if (s->cu_mv_y_dir0) CHECK_CU(cu->cuMemFree(s->cu_mv_y_dir0));
        if (s->cu_mv_x_dir1) CHECK_CU(cu->cuMemFree(s->cu_mv_x_dir1));
        if (s->cu_mv_y_dir1) CHECK_CU(cu->cuMemFree(s->cu_mv_y_dir1));
        if (s->cu_mv_table_prev0) CHECK_CU(cu->cuMemFree(s->cu_mv_table_prev0));
        if (s->cu_mv_table_prev1) CHECK_CU(cu->cuMemFree(s->cu_mv_table_prev1));
        if (s->cu_pixel_weights) CHECK_CU(cu->cuMemFree(s->cu_pixel_weights));
        if (s->cu_pixel_refs) CHECK_CU(cu->cuMemFree(s->cu_pixel_refs));
        if (s->cu_pixel_nb) CHECK_CU(cu->cuMemFree(s->cu_pixel_nb));
        if (s->cu_pixel_mvs_x) CHECK_CU(cu->cuMemFree(s->cu_pixel_mvs_x));
        if (s->cu_pixel_mvs_y) CHECK_CU(cu->cuMemFree(s->cu_pixel_mvs_y));
        if (s->cu_scene_sad) CHECK_CU(cu->cuMemFree(s->cu_scene_sad));
        
        CHECK_CU(cu->cuCtxPopCurrent(&dummy));
        
        if (s->cu_module) {
            CHECK_CU(cu->cuModuleUnload(s->cu_module));
        }
    }
    
    av_freep(&s->preds0_mvs);
    av_freep(&s->preds1_mvs);
    
    for (i = 0; i < NB_FRAMES; i++) {
        av_frame_free(&s->frames[i]);
    }
    
    av_frame_free(&s->frame);
    av_frame_free(&s->tmp_frame);
    av_buffer_unref(&s->frames_ctx);
}

static int cudaminterpolate_query_formats(AVFilterContext *ctx)
{
    static const enum AVPixelFormat pixel_formats[] = {
        AV_PIX_FMT_CUDA, AV_PIX_FMT_NONE,
    };
    AVFilterFormats *pix_fmts = ff_make_format_list(pixel_formats);

    return ff_set_common_formats(ctx, pix_fmts);
}

static av_cold int init_hwframe_ctx(CUDAMInterpolateContext *s, AVBufferRef *device_ctx, int width, int height)
{
    AVBufferRef *out_ref = NULL;
    AVHWFramesContext *out_ctx;
    int ret;

    out_ref = av_hwframe_ctx_alloc(device_ctx);
    if (!out_ref)
        return AVERROR(ENOMEM);
    out_ctx = (AVHWFramesContext*)out_ref->data;

    out_ctx->format    = AV_PIX_FMT_CUDA;
    out_ctx->sw_format = s->pix_fmt;
    out_ctx->width     = FFALIGN(width,  BLOCKX);
    out_ctx->height    = FFALIGN(height, BLOCKY);

    ret = av_hwframe_ctx_init(out_ref);
    if (ret < 0)
        goto fail;

    av_frame_unref(s->frame);
    ret = av_hwframe_get_buffer(out_ref, s->frame, 0);
    if (ret < 0)
        goto fail;

    s->frame->width  = width;
    s->frame->height = height;

    av_buffer_unref(&s->frames_ctx);
    s->frames_ctx = out_ref;

    return 0;
fail:
    av_buffer_unref(&out_ref);
    return ret;
}

static int format_is_supported(enum AVPixelFormat fmt)
{
    int i;

    for (i = 0; i < FF_ARRAY_ELEMS(supported_formats); i++)
        if (supported_formats[i] == fmt)
            return 1;
    return 0;
}

static av_cold int init_processing_chain(AVFilterContext *ctx, int width, int height)
{
    CUDAMInterpolateContext *s = ctx->priv;
    AVHWFramesContext *in_frames_ctx;
    int ret;

    if (!ctx->inputs[0]->hw_frames_ctx) {
        av_log(ctx, AV_LOG_ERROR, "No hw context provided on input\n");
        return AVERROR(EINVAL);
    }
    in_frames_ctx = (AVHWFramesContext*)ctx->inputs[0]->hw_frames_ctx->data;
    s->pix_fmt     = in_frames_ctx->sw_format;
    
    if (!format_is_supported(s->pix_fmt)) {
        av_log(ctx, AV_LOG_ERROR, "Unsupported input format: %s\n",
               av_get_pix_fmt_name(s->pix_fmt));
        return AVERROR(ENOSYS);
    }

    ret = init_hwframe_ctx(s, in_frames_ctx->device_ref, width, height);
    if (ret < 0)
        return ret;

    ctx->outputs[0]->hw_frames_ctx = av_buffer_ref(s->frames_ctx);
    if (!ctx->outputs[0]->hw_frames_ctx)
        return AVERROR(ENOMEM);

    return 0;
}

static int cudaminterpolate_config_input(AVFilterLink *inlink)
{
    CUDAMInterpolateContext *s = inlink->dst->priv;
    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(inlink->format);

    s->log2_chroma_h = desc->log2_chroma_h;
    s->log2_chroma_w = desc->log2_chroma_w;
    s->nb_planes = av_pix_fmt_count_planes(inlink->format);

    return 0;
}

static av_cold int cudaminterpolate_config_props(AVFilterLink *outlink)
{
    AVFilterContext *ctx = outlink->src;
    AVFilterLink *inlink = outlink->src->inputs[0];
    CUDAMInterpolateContext *s  = ctx->priv;
    AVHWFramesContext     *frames_ctx = (AVHWFramesContext*)inlink->hw_frames_ctx->data;
    AVCUDADeviceContext *device_hwctx = frames_ctx->device_ctx->hwctx;
    CUcontext dummy;
    CUcontext cuda_ctx = device_hwctx->cuda_ctx;
    CudaFunctions *cu = device_hwctx->internal->cuda_dl;
    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(frames_ctx->sw_format);
    int ret;
    size_t mv_size;
    size_t pixel_size;

    extern char vf_minterpolate_cuda_ptx[];

    s->hwctx = device_hwctx;
    s->width = inlink->w;
    s->height = inlink->h;
    s->log2_chroma_w = desc->log2_chroma_w;
    s->log2_chroma_h = desc->log2_chroma_h;
    s->nb_planes = av_pix_fmt_count_planes(frames_ctx->sw_format);

    outlink->w = inlink->w;
    outlink->h = inlink->h;
    outlink->frame_rate = s->frame_rate;
    outlink->time_base  = av_inv_q(s->frame_rate);

    // 计算块尺寸
    s->log2_mb_size = av_ceil_log2_c(s->mb_size);
    s->mb_size = 1 << s->log2_mb_size;
    s->b_width = s->width >> s->log2_mb_size;
    s->b_height = s->height >> s->log2_mb_size;
    s->b_count = s->b_width * s->b_height;
    s->mv_width = s->b_width;
    s->mv_height = s->b_height;

    ret = CHECK_CU(cu->cuCtxPushCurrent(cuda_ctx));
    if (ret < 0)
        goto fail;

    ret = CHECK_CU(cu->cuModuleLoadData(&s->cu_module, vf_minterpolate_cuda_ptx));
    if (ret < 0) {
        av_log(ctx, AV_LOG_ERROR, "Error loading module data\n");
        goto fail;
    }

    ret = CHECK_CU(cu->cuModuleGetFunction(&s->cu_func_epzs_me, s->cu_module, "cuda_epzs_motion_estimation"));
    if (ret < 0)
        goto fail;

    ret = CHECK_CU(cu->cuModuleGetFunction(&s->cu_func_umh_me, s->cu_module, "cuda_umh_motion_estimation"));
    if (ret < 0)
        goto fail;

    ret = CHECK_CU(cu->cuModuleGetFunction(&s->cu_func_bidir_me, s->cu_module, "cuda_bidir_motion_estimation"));
    if (ret < 0)
        goto fail;

    ret = CHECK_CU(cu->cuModuleGetFunction(&s->cu_func_scene_sad, s->cu_module, "cuda_scene_sad"));
    if (ret < 0)
        goto fail;

    ret = CHECK_CU(cu->cuModuleGetFunction(&s->cu_func_blend, s->cu_module, "cuda_blend_frames"));
    if (ret < 0)
        goto fail;

    ret = CHECK_CU(cu->cuModuleGetFunction(&s->cu_func_bilateral_obmc, s->cu_module, "cuda_bilateral_obmc"));
    if (ret < 0)
        goto fail;

    ret = CHECK_CU(cu->cuModuleGetFunction(&s->cu_func_bidirectional_obmc, s->cu_module, "cuda_bidirectional_obmc"));
    if (ret < 0)
        goto fail;

    ret = CHECK_CU(cu->cuModuleGetFunction(&s->cu_func_set_frame_data, s->cu_module, "cuda_set_frame_data"));
    if (ret < 0)
        goto fail;

    // 分配运动矢量设备内存
    mv_size = s->b_count * sizeof(int16_t);
    ret = CHECK_CU(cu->cuMemAlloc(&s->cu_mv_x, mv_size));
    if (ret < 0)
        goto fail;
    
    ret = CHECK_CU(cu->cuMemAlloc(&s->cu_mv_y, mv_size));
    if (ret < 0)
        goto fail;

    ret = CHECK_CU(cu->cuMemAlloc(&s->cu_mv_table_prev0, mv_size));
    if (ret < 0)
        goto fail;

    ret = CHECK_CU(cu->cuMemAlloc(&s->cu_mv_table_prev1, mv_size));
    if (ret < 0)
        goto fail;

    // 分配双向运动估计设备内存
    ret = CHECK_CU(cu->cuMemAlloc(&s->cu_mv_x_dir0, mv_size));
    if (ret < 0)
        goto fail;
    
    ret = CHECK_CU(cu->cuMemAlloc(&s->cu_mv_y_dir0, mv_size));
    if (ret < 0)
        goto fail;
    
    ret = CHECK_CU(cu->cuMemAlloc(&s->cu_mv_x_dir1, mv_size));
    if (ret < 0)
        goto fail;
    
    ret = CHECK_CU(cu->cuMemAlloc(&s->cu_mv_y_dir1, mv_size));
    if (ret < 0)
        goto fail;

    // 分配像素级数据设备内存
    pixel_size = (size_t)s->width * s->height;
    ret = CHECK_CU(cu->cuMemAlloc(&s->cu_pixel_weights, pixel_size * NB_PIXEL_MVS * sizeof(uint32_t)));
    if (ret < 0)
        goto fail;

    ret = CHECK_CU(cu->cuMemAlloc(&s->cu_pixel_refs, pixel_size * NB_PIXEL_MVS * sizeof(int8_t)));
    if (ret < 0)
        goto fail;

    ret = CHECK_CU(cu->cuMemAlloc(&s->cu_pixel_nb, pixel_size * sizeof(int)));
    if (ret < 0)
        goto fail;

    ret = CHECK_CU(cu->cuMemAlloc(&s->cu_pixel_mvs_x, pixel_size * NB_PIXEL_MVS * sizeof(int16_t)));
    if (ret < 0)
        goto fail;

    ret = CHECK_CU(cu->cuMemAlloc(&s->cu_pixel_mvs_y, pixel_size * NB_PIXEL_MVS * sizeof(int16_t)));
    if (ret < 0)
        goto fail;

    ret = CHECK_CU(cu->cuMemAlloc(&s->cu_scene_sad, sizeof(uint64_t)));
    if (ret < 0)
        goto fail;

    // 分配预测器主机内存
    s->preds0_mvs = av_mallocz(10 * 2 * sizeof(int16_t));
    s->preds1_mvs = av_mallocz(10 * 2 * sizeof(int16_t));
    if (!s->preds0_mvs || !s->preds1_mvs) {
        ret = AVERROR(ENOMEM);
        goto fail;
    }

    s->prev_mafd = 0.0;
    s->scene_changed = 0;
    
    CHECK_CU(cu->cuCtxPopCurrent(&dummy));

    ret = init_processing_chain(ctx, inlink->w, inlink->h);
    if (ret < 0)
        return ret;

    av_log(ctx, AV_LOG_INFO, "Initialized minterpolate_cuda: %dx%d, mb_size:%d, search_param:%d\n",
           s->width, s->height, s->mb_size, s->search_param);
           
    if (inlink->sample_aspect_ratio.num) {
        outlink->sample_aspect_ratio = av_mul_q((AVRational){outlink->h*inlink->w,
                                                             outlink->w*inlink->h},
                                                inlink->sample_aspect_ratio);
    } else {
        outlink->sample_aspect_ratio = inlink->sample_aspect_ratio;
    }

    return 0;

fail:
    return ret;
}

// 已移除未使用的函数

static int bilateral_me_cuda(AVFilterContext *ctx)
{
    CUDAMInterpolateContext *s = ctx->priv;
    CudaFunctions *cu = s->hwctx->internal->cuda_dl;
    CUcontext dummy;
    int ret;
    CUdeviceptr cur_data;
    CUdeviceptr ref_data;
    int cur_stride;
    int ref_stride;
    int x_min;
    int y_min;
    int x_max;
    int y_max;
    
    if (!s->frames[0] || !s->frames[1] || !s->frames[2])
        return 0;

    ret = CHECK_CU(cu->cuCtxPushCurrent(s->hwctx->cuda_ctx));
    if (ret < 0)
        return ret;

    // 双向ME：frames[1]为当前帧，frames[2]为参考帧
    cur_data = (CUdeviceptr)s->frames[1]->data[0];
    ref_data = (CUdeviceptr)s->frames[2]->data[0];
    cur_stride = s->frames[1]->linesize[0];
    ref_stride = s->frames[2]->linesize[0];
    
    x_min = 0;
    y_min = 0;
    x_max = (s->b_width - 1) << s->log2_mb_size;
    y_max = (s->b_height - 1) << s->log2_mb_size;
    
    // 重置MV表：prev1复制到prev2，prev0复制到prev1，然后清零prev0
    if (s->cu_mv_table_prev1 && s->cu_mv_table_prev0) {
        CUDA_MEMCPY2D cpy = {
            .srcMemoryType = CU_MEMORYTYPE_DEVICE,
            .dstMemoryType = CU_MEMORYTYPE_DEVICE,
            .srcDevice = s->cu_mv_table_prev0,
            .dstDevice = s->cu_mv_table_prev1,
            .srcPitch = s->b_count * sizeof(int16_t),
            .dstPitch = s->b_count * sizeof(int16_t),
            .WidthInBytes = s->b_count * sizeof(int16_t),
            .Height = 1,
        };
        CHECK_CU(cu->cuMemcpy2DAsync(&cpy, s->hwctx->stream));
        CHECK_CU(cu->cuStreamSynchronize(s->hwctx->stream));
    }
    if (s->cu_mv_table_prev0) {
        CHECK_CU(cu->cuMemsetD8Async(s->cu_mv_table_prev0, 0, s->b_count * sizeof(int16_t) * 2, s->hwctx->stream));
        CHECK_CU(cu->cuStreamSynchronize(s->hwctx->stream));
    }
    
    // 对所有宏块启动EPZS运动估计
    // 暂时使用简单预测器，每个宏块用(0,0)
    s->preds0_count = 1;
    s->preds0_mvs[0] = 0;
    s->preds0_mvs[1] = 0;
    s->preds1_count = 0;
    
    {
        CUdeviceptr cu_preds0 = 0;
        CUdeviceptr cu_preds1 = 0;
        int pred_x;
        int pred_y;
        int grid_x, grid_y, block_x, block_y;
        
        if (s->preds0_count > 0) {
            CHECK_CU(cu->cuMemAlloc(&cu_preds0, s->preds0_count * 2 * sizeof(int16_t)));
            {
                CUDA_MEMCPY2D cpy = {
                    .srcMemoryType = CU_MEMORYTYPE_HOST,
                    .dstMemoryType = CU_MEMORYTYPE_DEVICE,
                    .srcHost = s->preds0_mvs,
                    .dstDevice = cu_preds0,
                    .srcPitch = s->preds0_count * 2 * sizeof(int16_t),
                    .dstPitch = s->preds0_count * 2 * sizeof(int16_t),
                    .WidthInBytes = s->preds0_count * 2 * sizeof(int16_t),
                    .Height = 1,
                };
                CHECK_CU(cu->cuMemcpy2DAsync(&cpy, s->hwctx->stream));
                CHECK_CU(cu->cuStreamSynchronize(s->hwctx->stream));
            }
        }
        
        pred_x = 0;
        pred_y = 0;
        
        // 对所有宏块启动核函数
        block_x = 16;
        block_y = 16;
        grid_x = DIV_UP(s->b_width, block_x);
        grid_y = DIV_UP(s->b_height, block_y);
    
        {
            void *args[] = {
                &cur_data, &cur_stride,
                &ref_data, &ref_stride,
                &s->cu_mv_x, &s->cu_mv_y,
                &s->width, &s->height, &s->mb_size, &s->search_param,
                &s->b_width, &s->b_height,
                &x_min, &x_max, &y_min, &y_max,
                &s->cu_mv_table_prev0, &s->cu_mv_table_prev1,
                &s->preds0_count, &cu_preds0,
                &s->preds1_count, &cu_preds1,
                &pred_x, &pred_y
            };
            
            // 根据ME方法选择核函数
            CUfunction me_kernel = s->cu_func_epzs_me;
            if (s->me_method == AV_ME_METHOD_UMH) {
                me_kernel = s->cu_func_umh_me;
            }
            
            ret = CHECK_CU(cu->cuLaunchKernel(me_kernel,
                                              grid_x, grid_y, 1,
                                              block_x, block_y, 1,
                                              0, s->hwctx->stream, args, NULL));
            
            if (ret < 0) {
                if (cu_preds0) CHECK_CU(cu->cuMemFree(cu_preds0));
                if (cu_preds1) CHECK_CU(cu->cuMemFree(cu_preds1));
                CHECK_CU(cu->cuCtxPopCurrent(&dummy));
                return ret;
            }
        }
        
        CHECK_CU(cu->cuStreamSynchronize(s->hwctx->stream));
        
        if (cu_preds0) CHECK_CU(cu->cuMemFree(cu_preds0));
        if (cu_preds1) CHECK_CU(cu->cuMemFree(cu_preds1));
    }
    
    CHECK_CU(cu->cuCtxPopCurrent(&dummy));
    return 0;
}

// 双向运动估计
static int bidir_me_cuda(AVFilterContext *ctx)
{
    CUDAMInterpolateContext *s = ctx->priv;
    CudaFunctions *cu = s->hwctx->internal->cuda_dl;
    CUcontext dummy;
    int ret;
    int dir;
    CUdeviceptr cur_data;
    CUdeviceptr ref_data;
    int cur_stride;
    int ref_stride;
    int x_min;
    int y_min;
    int x_max;
    int y_max;
    CUdeviceptr mv_x_out;
    CUdeviceptr mv_y_out;
    CUdeviceptr cu_preds0;
    CUdeviceptr cu_preds1;
    int pred_x;
    int pred_y;
    int grid_x, grid_y, block_x, block_y;
    
    if (!s->frames[1] || !s->frames[2] || !s->frames[3])
        return 0;

    ret = CHECK_CU(cu->cuCtxPushCurrent(s->hwctx->cuda_ctx));
    if (ret < 0)
        return ret;

    // 双向ME：frames[2]为当前帧，在frames[1](dir=0)和frames[3](dir=1)中搜索
    for (dir = 0; dir < 2; dir++) {
        cur_data = (CUdeviceptr)s->frames[2]->data[0];
        ref_data = (CUdeviceptr)s->frames[dir ? 3 : 1]->data[0];
        cur_stride = s->frames[2]->linesize[0];
        ref_stride = s->frames[dir ? 3 : 1]->linesize[0];
        
        x_min = 0;
        y_min = 0;
        x_max = (s->b_width - 1) << s->log2_mb_size;
        y_max = (s->b_height - 1) << s->log2_mb_size;
        
        // 根据方向选择输出MV缓冲区
        mv_x_out = dir ? s->cu_mv_x_dir1 : s->cu_mv_x_dir0;
        mv_y_out = dir ? s->cu_mv_y_dir1 : s->cu_mv_y_dir0;
        
        // 准备预测器
        s->preds0_count = 1;
        s->preds0_mvs[0] = 0;
        s->preds0_mvs[1] = 0;
        s->preds1_count = 0;
        
        cu_preds0 = 0;
        cu_preds1 = 0;
        pred_x = 0;
        pred_y = 0;
        
        if (s->preds0_count > 0) {
            CHECK_CU(cu->cuMemAlloc(&cu_preds0, s->preds0_count * 2 * sizeof(int16_t)));
            {
                CUDA_MEMCPY2D cpy = {
                    .srcMemoryType = CU_MEMORYTYPE_HOST,
                    .dstMemoryType = CU_MEMORYTYPE_DEVICE,
                    .srcHost = s->preds0_mvs,
                    .dstDevice = cu_preds0,
                    .srcPitch = s->preds0_count * 2 * sizeof(int16_t),
                    .dstPitch = s->preds0_count * 2 * sizeof(int16_t),
                    .WidthInBytes = s->preds0_count * 2 * sizeof(int16_t),
                    .Height = 1,
                };
                CHECK_CU(cu->cuMemcpy2DAsync(&cpy, s->hwctx->stream));
                CHECK_CU(cu->cuStreamSynchronize(s->hwctx->stream));
            }
        }
        
        block_x = 16;
        block_y = 16;
        grid_x = DIV_UP(s->b_width, block_x);
        grid_y = DIV_UP(s->b_height, block_y);
        
        {
            void *args[] = {
                &cur_data, &cur_stride,
                &ref_data, &ref_stride,
                &mv_x_out, &mv_y_out,
                &s->width, &s->height, &s->mb_size, &s->search_param,
                &s->b_width, &s->b_height,
                &x_min, &x_max, &y_min, &y_max,
                &s->cu_mv_table_prev0, &s->cu_mv_table_prev1,
                &s->preds0_count, &cu_preds0,
                &s->preds1_count, &cu_preds1,
                &pred_x, &pred_y, &s->me_method
            };
            
            // 使用双向ME核函数
            ret = CHECK_CU(cu->cuLaunchKernel(s->cu_func_bidir_me,
                                              grid_x, grid_y, 1,
                                              block_x, block_y, 1,
                                              0, s->hwctx->stream, args, NULL));
            
            if (ret < 0) {
                if (cu_preds0) CHECK_CU(cu->cuMemFree(cu_preds0));
                if (cu_preds1) CHECK_CU(cu->cuMemFree(cu_preds1));
                CHECK_CU(cu->cuCtxPopCurrent(&dummy));
                return ret;
            }
        }
        
        CHECK_CU(cu->cuStreamSynchronize(s->hwctx->stream));
        
        if (cu_preds0) CHECK_CU(cu->cuMemFree(cu_preds0));
        if (cu_preds1) CHECK_CU(cu->cuMemFree(cu_preds1));
    }
    
    CHECK_CU(cu->cuCtxPopCurrent(&dummy));
    return 0;
}

static int inject_frame_cuda(AVFilterLink *inlink, AVFrame *avf_in)
{
    AVFilterContext *ctx = inlink->dst;
    CUDAMInterpolateContext *s = ctx->priv;
    int i;

    // 帧移位：frames[0]释放，frames[1]->frames[0]，frames[2]->frames[1]，frames[3]->frames[2]，新帧->frames[3]
    av_frame_free(&s->frames[0]);
    for (i = 0; i < NB_FRAMES - 1; i++) {
        s->frames[i] = s->frames[i + 1];
    }
    s->frames[NB_FRAMES - 1] = av_frame_clone(avf_in);
    if (!s->frames[NB_FRAMES - 1])
        return AVERROR(ENOMEM);

    // 根据模式执行运动估计
    if (s->mi_mode == MI_MODE_MCI) {
        if (s->me_mode == ME_MODE_BIDIR) {
            // 双向运动估计
            if (s->frames[1] && s->frames[2] && s->frames[3]) {
                int ret = bidir_me_cuda(ctx);
                if (ret < 0)
                    return ret;
            }
        } else if (s->me_mode == ME_MODE_BILAT) {
            // 双向运动估计
            if (s->frames[0] && s->frames[1] && s->frames[2]) {
                int ret = bilateral_me_cuda(ctx);
                if (ret < 0)
                    return ret;
            }
        }
    }

    return 0;
}

static int detect_scene_change_cuda(AVFilterContext *ctx)
{
    CUDAMInterpolateContext *s = ctx->priv;
    CudaFunctions *cu = s->hwctx->internal->cuda_dl;
    CUcontext dummy;
    int ret;
    uint64_t sad_host = 0;
    double mafd, diff;
    
    if (s->scd_method != SCD_METHOD_FDIFF || !s->frames[1] || !s->frames[2])
        return 0;
    
    ret = CHECK_CU(cu->cuCtxPushCurrent(s->hwctx->cuda_ctx));
    if (ret < 0)
        return ret;
    
    // 重置SAD累加器
    CHECK_CU(cu->cuMemsetD8Async(s->cu_scene_sad, 0, sizeof(uint64_t), s->hwctx->stream));
    
    {
        CUdeviceptr frame1_data = (CUdeviceptr)s->frames[1]->data[0];
        CUdeviceptr frame2_data = (CUdeviceptr)s->frames[2]->data[0];
        int stride1 = s->frames[1]->linesize[0];
        int stride2 = s->frames[2]->linesize[0];
        int grid_x, grid_y, block_x, block_y;
        
        // 启动场景SAD核函数
        block_x = 32;
        block_y = 16;
        grid_x = DIV_UP(s->width, block_x);
        grid_y = DIV_UP(s->height, block_y);
    
        {
            void *args[] = { &frame1_data, &stride1, &frame2_data, &stride2,
                             &s->cu_scene_sad, &s->width, &s->height };
            
            ret = CHECK_CU(cu->cuLaunchKernel(s->cu_func_scene_sad,
                                              grid_x, grid_y, 1,
                                              block_x, block_y, 1,
                                              0, s->hwctx->stream, args, NULL));
            
            if (ret < 0) {
                CHECK_CU(cu->cuCtxPopCurrent(&dummy));
                return ret;
            }
        }
    }
    
    CHECK_CU(cu->cuStreamSynchronize(s->hwctx->stream));
    
    // 将SAD结果复制到主机
    {
        CUDA_MEMCPY2D cpy = {
            .srcMemoryType = CU_MEMORYTYPE_DEVICE,
            .dstMemoryType = CU_MEMORYTYPE_HOST,
            .srcDevice = s->cu_scene_sad,
            .dstHost = &sad_host,
            .srcPitch = sizeof(uint64_t),
            .dstPitch = sizeof(uint64_t),
            .WidthInBytes = sizeof(uint64_t),
            .Height = 1,
        };
        CHECK_CU(cu->cuMemcpy2DAsync(&cpy, s->hwctx->stream));
        CHECK_CU(cu->cuStreamSynchronize(s->hwctx->stream));
    }
    
    CHECK_CU(cu->cuCtxPopCurrent(&dummy));
    
    // 计算场景切换阈值
    {
        double ret_val;
        mafd = (double)sad_host / (s->height * s->width * 3);
        diff = fabs(mafd - s->prev_mafd);
        ret_val = av_clipd(FFMIN(mafd, diff), 0.0, 100.0);
        s->prev_mafd = mafd;
        s->scene_changed = (ret_val >= s->scd_threshold);
    }
    
    return 0;
}

static int interpolate_cuda(AVFilterLink *inlink, AVFrame *avf_out, int alpha)
{
    AVFilterContext *ctx = inlink->dst;
    CUDAMInterpolateContext *s = ctx->priv;
    CudaFunctions *cu = s->hwctx->internal->cuda_dl;
    CUcontext dummy;
    int ret;
    int plane;
    
    if (alpha == 0 || alpha == ALPHA_MAX) {
        // 简单帧复制
        AVFrame *src = alpha ? s->frames[2] : s->frames[1];
        ret = av_hwframe_transfer_data(avf_out, src, 0);
        return ret;
    }
    
    if (s->scene_changed) {
        // 场景切换时复制帧
        AVFrame *src = alpha > ALPHA_MAX / 2 ? s->frames[2] : s->frames[1];
        ret = av_hwframe_transfer_data(avf_out, src, 0);
        return ret;
    }
    
    ret = CHECK_CU(cu->cuCtxPushCurrent(s->hwctx->cuda_ctx));
    if (ret < 0)
        return ret;
    
    switch(s->mi_mode) {
        case MI_MODE_DUP:
            {
                AVFrame *src = alpha > ALPHA_MAX / 2 ? s->frames[2] : s->frames[1];
                CHECK_CU(cu->cuCtxPopCurrent(&dummy));
                ret = av_hwframe_transfer_data(avf_out, src, 0);
                return ret;
            }
            break;
        case MI_MODE_BLEND:
            // 使用CUDA核函数混合帧
            for (plane = 0; plane < s->nb_planes; plane++) {
                int width = avf_out->width;
                int height = avf_out->height;
                int grid_x, grid_y, block_x, block_y;
                CUdeviceptr out_data;
                CUdeviceptr frame1_data;
                CUdeviceptr frame2_data;
                int out_stride;
                int stride1;
                int stride2;
                
                if (plane == 1 || plane == 2) {
                    width = AV_CEIL_RSHIFT(width, s->log2_chroma_w);
                    height = AV_CEIL_RSHIFT(height, s->log2_chroma_h);
                }
                
                out_data = (CUdeviceptr)avf_out->data[plane];
                frame1_data = (CUdeviceptr)s->frames[1]->data[plane];
                frame2_data = (CUdeviceptr)s->frames[2]->data[plane];
                out_stride = avf_out->linesize[plane];
                stride1 = s->frames[1]->linesize[plane];
                stride2 = s->frames[2]->linesize[plane];
                
                block_x = 32;
                block_y = 16;
                grid_x = DIV_UP(width, block_x);
                grid_y = DIV_UP(height, block_y);
                
                {
                    void *args[] = { &out_data, &out_stride,
                                     &frame1_data, &stride1,
                                     &frame2_data, &stride2,
                                     &width, &height, &alpha };
                    
                    ret = CHECK_CU(cu->cuLaunchKernel(s->cu_func_blend,
                                                      grid_x, grid_y, 1,
                                                      block_x, block_y, 1,
                                                      0, s->hwctx->stream, args, NULL));
                    if (ret < 0) {
                        CHECK_CU(cu->cuCtxPopCurrent(&dummy));
                        return ret;
                    }
                }
                CHECK_CU(cu->cuStreamSynchronize(s->hwctx->stream));
            }
            break;
        case MI_MODE_MCI:
            if (s->me_mode == ME_MODE_BIDIR) {
                // 双向OBMC
                int grid_x, grid_y, block_x, block_y;
                int mb_x_start;
                int mb_y_start;
                
                // Reset pixel references
                CHECK_CU(cu->cuMemsetD8Async(s->cu_pixel_nb, 0, s->width * s->height * sizeof(int), s->hwctx->stream));
                CHECK_CU(cu->cuStreamSynchronize(s->hwctx->stream));
                
                // 对每个宏块执行双向OBMC
                block_x = 256;
                block_y = 1;
                grid_x = DIV_UP(s->b_count, block_x);
                grid_y = 1;
                
                mb_x_start = 0;
                mb_y_start = 0;
                
                {
                    void *args_obmc[] = {
                        &s->cu_mv_x_dir0, &s->cu_mv_y_dir0,
                        &s->cu_mv_x_dir1, &s->cu_mv_y_dir1,
                        &s->cu_pixel_weights, &s->cu_pixel_refs, &s->cu_pixel_nb,
                        &s->cu_pixel_mvs_x, &s->cu_pixel_mvs_y,
                        &s->width, &s->height, &s->mb_size, &s->b_width, &s->log2_mb_size,
                        &mb_x_start, &mb_y_start, &s->b_width, &s->b_height,
                        &alpha
                    };
                    
                    // 对所有宏块启动双向OBMC核函数
                    ret = CHECK_CU(cu->cuLaunchKernel(s->cu_func_bidirectional_obmc,
                                                      grid_x, grid_y, 1,
                                                      block_x, block_y, 1,
                                                      0, s->hwctx->stream, args_obmc, NULL));
                    if (ret < 0) {
                        CHECK_CU(cu->cuCtxPopCurrent(&dummy));
                        return ret;
                    }
                    CHECK_CU(cu->cuStreamSynchronize(s->hwctx->stream));
                }
                
                // Set frame data using weighted interpolation
                for (plane = 0; plane < s->nb_planes; plane++) {
                    int width = avf_out->width;
                    int height = avf_out->height;
                    int chroma;
                    CUdeviceptr out_data;
                    CUdeviceptr frame1_data;
                    CUdeviceptr frame2_data;
                    int out_stride;
                    int stride1;
                    int stride2;
                    int grid_x2, grid_y2, block_x2, block_y2;
                    
                    chroma = (plane == 1 || plane == 2);
                    
                    if (chroma) {
                        width = AV_CEIL_RSHIFT(width, s->log2_chroma_w);
                        height = AV_CEIL_RSHIFT(height, s->log2_chroma_h);
                    }
                    
                    out_data = (CUdeviceptr)avf_out->data[plane];
                    frame1_data = (CUdeviceptr)s->frames[1]->data[plane];
                    frame2_data = (CUdeviceptr)s->frames[2]->data[plane];
                    out_stride = avf_out->linesize[plane];
                    stride1 = s->frames[1]->linesize[plane];
                    stride2 = s->frames[2]->linesize[plane];
                    
                    block_x2 = 32;
                    block_y2 = 16;
                    grid_x2 = DIV_UP(width, block_x2);
                    grid_y2 = DIV_UP(height, block_y2);
                    
                    {
                        void *args_set[] = {
                            &out_data, &out_stride,
                            &frame1_data, &stride1,
                            &frame2_data, &stride2,
                            &s->cu_pixel_weights, &s->cu_pixel_refs, &s->cu_pixel_nb,
                            &s->cu_pixel_mvs_x, &s->cu_pixel_mvs_y,
                            &width, &height, &alpha,
                            &chroma, &s->log2_chroma_w, &s->log2_chroma_h
                        };
                        
                        ret = CHECK_CU(cu->cuLaunchKernel(s->cu_func_set_frame_data,
                                                          grid_x2, grid_y2, 1,
                                                          block_x2, block_y2, 1,
                                                          0, s->hwctx->stream, args_set, NULL));
                        if (ret < 0) {
                            CHECK_CU(cu->cuCtxPopCurrent(&dummy));
                            return ret;
                        }
                    }
                    CHECK_CU(cu->cuStreamSynchronize(s->hwctx->stream));
                }
            } else if (s->me_mode == ME_MODE_BILAT) {
                CUdeviceptr mv_x;
                CUdeviceptr mv_y;
                int grid_x, grid_y, block_x, block_y;
                int mb_x_start;
                int mb_y_start;
                
                // 重置像素引用
                CHECK_CU(cu->cuMemsetD8Async(s->cu_pixel_nb, 0, s->width * s->height * sizeof(int), s->hwctx->stream));
                CHECK_CU(cu->cuStreamSynchronize(s->hwctx->stream));
                
                // 对每个宏块执行双向OBMC
                mv_x = s->cu_mv_x;
                mv_y = s->cu_mv_y;
                
                block_x = 256;
                block_y = 1;
                grid_x = DIV_UP(s->b_count, block_x);
                grid_y = 1;
                
                mb_x_start = 0;
                mb_y_start = 0;
                
                {
                    void *args_obmc[] = {
                        &mv_x, &mv_y,
                        &s->cu_pixel_weights, &s->cu_pixel_refs, &s->cu_pixel_nb,
                        &s->cu_pixel_mvs_x, &s->cu_pixel_mvs_y,
                        &s->width, &s->height, &s->mb_size, &s->b_width, &s->log2_mb_size,
                        &mb_x_start, &mb_y_start, &s->b_width, &s->b_height,
                        &alpha
                    };
                    
                    // 对所有宏块启动OBMC核函数
                    ret = CHECK_CU(cu->cuLaunchKernel(s->cu_func_bilateral_obmc,
                                                      grid_x, grid_y, 1,
                                                      block_x, block_y, 1,
                                                      0, s->hwctx->stream, args_obmc, NULL));
                    if (ret < 0) {
                        CHECK_CU(cu->cuCtxPopCurrent(&dummy));
                        return ret;
                    }
                    CHECK_CU(cu->cuStreamSynchronize(s->hwctx->stream));
                }
                
                // 使用加权插值设置帧数据
                for (plane = 0; plane < s->nb_planes; plane++) {
                    int width = avf_out->width;
                    int height = avf_out->height;
                    int chroma;
                    CUdeviceptr out_data;
                    CUdeviceptr frame1_data;
                    CUdeviceptr frame2_data;
                    int out_stride;
                    int stride1;
                    int stride2;
                    int grid_x2, grid_y2, block_x2, block_y2;
                    
                    chroma = (plane == 1 || plane == 2);
                    
                    if (chroma) {
                        width = AV_CEIL_RSHIFT(width, s->log2_chroma_w);
                        height = AV_CEIL_RSHIFT(height, s->log2_chroma_h);
                    }
                    
                    out_data = (CUdeviceptr)avf_out->data[plane];
                    frame1_data = (CUdeviceptr)s->frames[1]->data[plane];
                    frame2_data = (CUdeviceptr)s->frames[2]->data[plane];
                    out_stride = avf_out->linesize[plane];
                    stride1 = s->frames[1]->linesize[plane];
                    stride2 = s->frames[2]->linesize[plane];
                    
                    block_x2 = 32;
                    block_y2 = 16;
                    grid_x2 = DIV_UP(width, block_x2);
                    grid_y2 = DIV_UP(height, block_y2);
                    
                    {
                        void *args_set[] = {
                            &out_data, &out_stride,
                            &frame1_data, &stride1,
                            &frame2_data, &stride2,
                            &s->cu_pixel_weights, &s->cu_pixel_refs, &s->cu_pixel_nb,
                            &s->cu_pixel_mvs_x, &s->cu_pixel_mvs_y,
                            &width, &height, &alpha,
                            &chroma, &s->log2_chroma_w, &s->log2_chroma_h
                        };
                        
                        ret = CHECK_CU(cu->cuLaunchKernel(s->cu_func_set_frame_data,
                                                          grid_x2, grid_y2, 1,
                                                          block_x2, block_y2, 1,
                                                          0, s->hwctx->stream, args_set, NULL));
                        if (ret < 0) {
                            CHECK_CU(cu->cuCtxPopCurrent(&dummy));
                            return ret;
                        }
                    }
                    CHECK_CU(cu->cuStreamSynchronize(s->hwctx->stream));
                }
            }
            break;
    }
    
    CHECK_CU(cu->cuCtxPopCurrent(&dummy));
    return 0;
}

static int cudaminterpolate_filter_frame(AVFilterLink *inlink, AVFrame *avf_in)
{
    AVFilterContext *ctx = inlink->dst;
    AVFilterLink *outlink = ctx->outputs[0];
    CUDAMInterpolateContext *s = ctx->priv;
    int ret;

    if (avf_in->pts == AV_NOPTS_VALUE) {
        ret = ff_filter_frame(ctx->outputs[0], avf_in);
        return ret;
    }

    if (!s->frames[NB_FRAMES - 1] || avf_in->pts < s->frames[NB_FRAMES - 1]->pts) {
        av_log(ctx, AV_LOG_VERBOSE, "Initializing out pts from input pts %"PRId64"\n", avf_in->pts);
        s->out_pts = av_rescale_q(avf_in->pts, inlink->time_base, outlink->time_base);
    }

    if (!s->frames[NB_FRAMES - 1]) {
        // 首帧，克隆以初始化缓冲区
        s->frames[NB_FRAMES - 1] = av_frame_clone(avf_in);
        if (!s->frames[NB_FRAMES - 1])
            return AVERROR(ENOMEM);
    }

    if ((ret = inject_frame_cuda(inlink, avf_in)) < 0)
        return ret;

    if (!s->frames[0])
        return 0;

    // 检测场景切换
    detect_scene_change_cuda(ctx);

    // 根据时间戳生成插值帧
    for (;;) {
        AVFrame *avf_out;
        int alpha;
        int64_t pts;

        if (!s->frames[1] || !s->frames[2])
            break;

        // 检查是否需要生成更多输出帧
        if (av_compare_ts(s->out_pts, outlink->time_base, s->frames[2]->pts, inlink->time_base) > 0)
            break;

        if (!(avf_out = ff_get_video_buffer(ctx->outputs[0], inlink->w, inlink->h)))
            return AVERROR(ENOMEM);

        av_frame_copy_props(avf_out, s->frames[NB_FRAMES - 1] ? s->frames[NB_FRAMES - 1] : s->frames[2]);
        avf_out->pts = s->out_pts++;

        // 计算插值alpha（与原始minterpolate.c逻辑相同）
        pts = av_rescale(avf_out->pts, (int64_t)ALPHA_MAX * outlink->time_base.num * inlink->time_base.den,
                                   (int64_t)outlink->time_base.den * inlink->time_base.num);
        
        if (s->frames[2]->pts > s->frames[1]->pts) {
            alpha = (pts - s->frames[1]->pts * ALPHA_MAX) / (s->frames[2]->pts - s->frames[1]->pts);
            alpha = av_clip(alpha, 0, ALPHA_MAX);
        } else {
            av_log(ctx, AV_LOG_DEBUG, "duplicate input PTS detected\n");
            alpha = 0;
        }

        // 使用CUDA核函数生成插值帧
        ret = interpolate_cuda(inlink, avf_out, alpha);
        if (ret < 0) {
            av_frame_free(&avf_out);
            return ret;
        }

        if ((ret = ff_filter_frame(ctx->outputs[0], avf_out)) < 0)
            return ret;
    }

    return 0;
}

#define OFFSET(x) offsetof(CUDAMInterpolateContext, x)
#define FLAGS AV_OPT_FLAG_VIDEO_PARAM|AV_OPT_FLAG_FILTERING_PARAM
#define CONST(name, help, val, unit) { name, help, 0, AV_OPT_TYPE_CONST, {.i64=val}, 0, 0, FLAGS, unit }

static const AVOption minterpolate_cuda_options[] = {
    { "fps", "output's frame rate", OFFSET(frame_rate), AV_OPT_TYPE_VIDEO_RATE, {.str = "60"}, 0, INT_MAX, FLAGS },
    { "mi_mode", "motion interpolation mode", OFFSET(mi_mode), AV_OPT_TYPE_INT, {.i64 = MI_MODE_MCI}, MI_MODE_DUP, MI_MODE_MCI, FLAGS, "mi_mode" },
        CONST("dup",    "duplicate frames",                     MI_MODE_DUP,            "mi_mode"),
        CONST("blend",  "blend frames",                         MI_MODE_BLEND,          "mi_mode"),
        CONST("mci",    "motion compensated interpolation",     MI_MODE_MCI,            "mi_mode"),
    { "mc_mode", "motion compensation mode", OFFSET(mc_mode), AV_OPT_TYPE_INT, {.i64 = MC_MODE_OBMC}, MC_MODE_OBMC, MC_MODE_OBMC, FLAGS, "mc_mode" },
        CONST("obmc",   "overlapped block motion compensation", MC_MODE_OBMC,           "mc_mode"),
    { "me_mode", "motion estimation mode", OFFSET(me_mode), AV_OPT_TYPE_INT, {.i64 = ME_MODE_BILAT}, ME_MODE_BIDIR, ME_MODE_BILAT, FLAGS, "me_mode" },
        CONST("bidir",  "bidirectional motion estimation",      ME_MODE_BIDIR,          "me_mode"),
        CONST("bilat",  "bilateral motion estimation",          ME_MODE_BILAT,          "me_mode"),
    { "me", "motion estimation method", OFFSET(me_method), AV_OPT_TYPE_INT, {.i64 = AV_ME_METHOD_EPZS}, AV_ME_METHOD_EPZS, AV_ME_METHOD_UMH, FLAGS, "me" },
        CONST("epzs",   "enhanced predictive zonal search",     AV_ME_METHOD_EPZS,      "me"),
        CONST("umh",    "uneven multi-hexagon search",          AV_ME_METHOD_UMH,       "me"),
    { "mb_size", "macroblock size", OFFSET(mb_size), AV_OPT_TYPE_INT, {.i64 = 16}, 4, 16, FLAGS },
    { "search_param", "search parameter", OFFSET(search_param), AV_OPT_TYPE_INT, {.i64 = 32}, 4, INT_MAX, FLAGS },
    { "scd", "scene change detection method", OFFSET(scd_method), AV_OPT_TYPE_INT, {.i64 = SCD_METHOD_FDIFF}, SCD_METHOD_FDIFF, SCD_METHOD_FDIFF, FLAGS, "scene" },
        CONST("fdiff",  "frame difference",                     SCD_METHOD_FDIFF,       "scene"),
    { "scd_threshold", "scene change threshold", OFFSET(scd_threshold), AV_OPT_TYPE_DOUBLE, {.dbl = 5.0}, 0, 100.0, FLAGS },
    { NULL }
};

AVFILTER_DEFINE_CLASS(minterpolate_cuda);


static const AVFilterPad cudaminterpolate_inputs[] = {
    {
        .name        = "default",
        .type        = AVMEDIA_TYPE_VIDEO,
        .config_props = cudaminterpolate_config_input,
        .filter_frame = cudaminterpolate_filter_frame,
    },
};

static const AVFilterPad cudaminterpolate_outputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .config_props = cudaminterpolate_config_props,
    },
};

AVFilter ff_vf_minterpolate_cuda = {
    .name      = "minterpolate_cuda",
    .description = NULL_IF_CONFIG_SMALL("GPU accelerated motion interpolation"),

    .init          = cudaminterpolate_init,
    .uninit        = cudaminterpolate_uninit,

    .priv_size = sizeof(CUDAMInterpolateContext),
    .priv_class = &minterpolate_cuda_class,
    .query_formats = cudaminterpolate_query_formats,
    .inputs    = cudaminterpolate_inputs,
    .outputs   = cudaminterpolate_outputs,

    .flags_internal = FF_FILTER_FLAG_HWFRAME_AWARE,
};
