/*
* Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a
* copy of this software and associated documentation files (the "Software"),
* to deal in the Software without restriction, including without limitation
* the rights to use, copy, modify, merge, publish, distribute, sublicense,
* and/or sell copies of the Software, and to permit persons to whom the
* Software is furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
* THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
* DEALINGS IN THE SOFTWARE.
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
#include "guided.h"

static const enum AVPixelFormat supported_formats[] = {
    AV_PIX_FMT_YUV420P,
    AV_PIX_FMT_NV12,
    AV_PIX_FMT_YUV420P10,
};

#define DIV_UP(a, b) ( ((a) + (b) - 1) / (b) )
#define ALIGN_UP(a, b) (((a) + (b) - 1) & ~((b) - 1))
#define NUM_BUFFERS 2
#define BLOCKX 32
#define BLOCKY 16

#define CHECK_CU(x) x//FF_CUDA_CHECK_DL(ctx, s->hwctx->internal->cuda_dl, x)

typedef struct CUDAGuidedContext {
    const AVClass *class;

    AVCUDADeviceContext *hwctx;

    AVBufferRef *frames_ctx;
    AVFrame     *frame;

    AVFrame *tmp_frame;

    /**
     * Output sw format. AV_PIX_FMT_NONE for no conversion.
     */
    enum AVPixelFormat format;
    enum AVPixelFormat pix_fmt;

    QNGuidedFilterParam luma_param;
    QNGuidedFilterParam chroma_param;
    int hsub, vsub;
    int radius[3];
    float eps[3];

    uint8_t  useTexture;
    CUcontext   cu_ctx;
    CUmodule    cu_module;
    CUfunction  cu_func_char2float;
    CUfunction  cu_func_meanfilter;
    CUfunction  cu_func_floatsquare;
    CUfunction  cu_func_computeab;
    CUfunction  cu_func_float2char;
    CUfunction  cu_func_copy;
    CUdeviceptr cu_temp_data;
    CUdeviceptr cu_temp_meanIP;
    CUdeviceptr cu_temp_corrIP;
    CUdeviceptr cu_temp_a;
    CUdeviceptr cu_temp_b;
} CUDAGuidedContext;


static av_cold int cudaguided_init(AVFilterContext *ctx)
{
    CUDAGuidedContext *s = ctx->priv;

    s->frame = av_frame_alloc();
    if (!s->frame)
        return AVERROR(ENOMEM);

    s->tmp_frame = av_frame_alloc();
    if (!s->tmp_frame)
        return AVERROR(ENOMEM);

    return 0;
}

static av_cold void cudaguided_uninit(AVFilterContext *ctx)
{
    CUDAGuidedContext *s = ctx->priv;
    if (s->hwctx && s->cu_module) {
        CudaFunctions *cu = s->hwctx->internal->cuda_dl;
        CUcontext dummy;

        CHECK_CU(cu->cuCtxPushCurrent(s->hwctx->cuda_ctx));
        if (s->cu_temp_data)
            CHECK_CU(cu->cuMemFree(s->cu_temp_data));
        if (s->cu_temp_meanIP)
            CHECK_CU(cu->cuMemFree(s->cu_temp_meanIP));
        if (s->cu_temp_corrIP)
            CHECK_CU(cu->cuMemFree(s->cu_temp_corrIP));
        if (s->cu_temp_a)
            CHECK_CU(cu->cuMemFree(s->cu_temp_a));
        if (s->cu_temp_b)
            CHECK_CU(cu->cuMemFree(s->cu_temp_b));
        CHECK_CU(cu->cuModuleUnload(s->cu_module));
        s->cu_module = NULL;
        CHECK_CU(cu->cuCtxPopCurrent(&dummy));
    }
    av_frame_free(&s->frame);
    av_buffer_unref(&s->frames_ctx);
    av_frame_free(&s->tmp_frame);
}

static int cudaguided_query_formats(AVFilterContext *ctx)
{
    static const enum AVPixelFormat pixel_formats[] = {
        AV_PIX_FMT_CUDA, AV_PIX_FMT_NONE,
    };
    AVFilterFormats *pix_fmts = ff_make_format_list(pixel_formats);

    return ff_set_common_formats(ctx, pix_fmts);
}

static av_cold int init_hwframe_ctx(CUDAGuidedContext *s, AVBufferRef *device_ctx,int width, int height)
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
    CUDAGuidedContext *s = ctx->priv;
    AVHWFramesContext *in_frames_ctx;
    int ret;

    /* check that we have a hw context */
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

    ret = init_hwframe_ctx(s, in_frames_ctx->device_ref,width,height);
    if (ret < 0)
        return ret;

    ctx->outputs[0]->hw_frames_ctx = av_buffer_ref(s->frames_ctx);
    if (!ctx->outputs[0]->hw_frames_ctx)
        return AVERROR(ENOMEM);

    return 0;
}

static av_cold int cudaguided_config_props(AVFilterLink *outlink)
{
    AVFilterContext *ctx = outlink->src;
    AVFilterLink *inlink = outlink->src->inputs[0];
    CUDAGuidedContext *s  = ctx->priv;
    AVHWFramesContext     *frames_ctx = (AVHWFramesContext*)inlink->hw_frames_ctx->data;
    AVCUDADeviceContext *device_hwctx = frames_ctx->device_ctx->hwctx;
    CUcontext dummy, cuda_ctx = device_hwctx->cuda_ctx;
    CudaFunctions *cu = device_hwctx->internal->cuda_dl;

    int ret;
    int max_width = inlink->w;
    int max_height = inlink->h;
    int cw = AV_CEIL_RSHIFT(inlink->w, s->hsub);
    int ch = AV_CEIL_RSHIFT(inlink->h, s->vsub);
    max_width = FFMAX(max_width, cw);
    max_height = FFMAX(max_height, ch);
    size_t temp_size = max_width * max_height * sizeof(float);

    extern char vf_guided_cuda_ptx[];

    s->hwctx = device_hwctx;

    outlink->w = inlink->w;
    outlink->h = inlink->h;

    ret = CHECK_CU(cu->cuCtxPushCurrent(cuda_ctx));
    if (ret < 0)
        goto fail;

    ret = CHECK_CU(cu->cuModuleLoadData(&s->cu_module, vf_guided_cuda_ptx));
    if (ret < 0) {
        av_log(ctx, AV_LOG_ERROR, "Error loading module data\n");
        goto fail;
    }

    ret = CHECK_CU(cu->cuModuleGetFunction(&s->cu_func_char2float, s->cu_module, "Guided_char2float"));
    if (ret < 0)
        goto fail;
        
    ret = CHECK_CU(cu->cuModuleGetFunction(&s->cu_func_meanfilter, s->cu_module, "Guided_meanfilter"));
    if (ret < 0)
        goto fail;
    
    ret = CHECK_CU(cu->cuModuleGetFunction(&s->cu_func_floatsquare, s->cu_module, "Guided_floatsquare"));
    if (ret < 0)
        goto fail;

    ret = CHECK_CU(cu->cuModuleGetFunction(&s->cu_func_computeab, s->cu_module, "Guided_computeab"));
    if (ret < 0)
        goto fail;

    ret = CHECK_CU(cu->cuModuleGetFunction(&s->cu_func_float2char, s->cu_module, "Guided_float2char"));
    if (ret < 0)
        goto fail;

    ret = CHECK_CU(cu->cuModuleGetFunction(&s->cu_func_copy, s->cu_module, "Guided_copy"));
    if (ret < 0)
        goto fail;

    // Allocate temporary buffers
    ret = CHECK_CU(cu->cuMemAlloc(&s->cu_temp_data, temp_size));
    if (ret < 0)
        goto fail;
    ret = CHECK_CU(cu->cuMemAlloc(&s->cu_temp_meanIP, temp_size));
    if (ret < 0)
        goto fail;
    ret = CHECK_CU(cu->cuMemAlloc(&s->cu_temp_corrIP, temp_size));
    if (ret < 0)
        goto fail;
    ret = CHECK_CU(cu->cuMemAlloc(&s->cu_temp_a, temp_size));
    if (ret < 0)
        goto fail;
    ret = CHECK_CU(cu->cuMemAlloc(&s->cu_temp_b, temp_size));
    if (ret < 0)
        goto fail;
    
    CHECK_CU(cu->cuCtxPopCurrent(&dummy));

    ret = init_processing_chain(ctx, inlink->w, inlink->h);
    if (ret < 0)
        return ret;

    av_log(ctx, AV_LOG_INFO, "luma_radius:%d, luma_eps:%f, chroma_radius:%d, chroma_eps:%f, pix_fmt is %d\n",
           s->luma_param.radius, s->luma_param.eps, s->chroma_param.radius, s->chroma_param.eps, s->pix_fmt);
           
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

static int call_char2float_kernel_tex(AVFilterContext *ctx, CUfunction func,
                              uint8_t *src_dptr, int src_width, int src_height, int src_pitch,
                              CUdeviceptr dst_devptr, int dst_width, int dst_height, int dst_pitch,
                              int pixel_size, float radius)
{
    CUDAGuidedContext *s = ctx->priv;
    CudaFunctions *cu = s->hwctx->internal->cuda_dl;
    CUtexObject tex = 0;
    int radiusVal = (int)radius;
    void *args[] = { &tex, &dst_devptr, &dst_width, &dst_height, &dst_pitch, &radiusVal };
    int ret;

    CUDA_TEXTURE_DESC tex_desc = {
        .filterMode = CU_TR_FILTER_MODE_LINEAR,
        .flags = CU_TRSF_READ_AS_INTEGER,
    };

    CUDA_RESOURCE_DESC res_desc = {
        .resType = CU_RESOURCE_TYPE_PITCH2D,
        .res.pitch2D.format = pixel_size == 1 ?
                              CU_AD_FORMAT_UNSIGNED_INT8 :
                              CU_AD_FORMAT_UNSIGNED_INT16,
        .res.pitch2D.numChannels = 1,
        .res.pitch2D.width = src_width,
        .res.pitch2D.height = src_height,
        .res.pitch2D.pitchInBytes = src_pitch * pixel_size,
        .res.pitch2D.devPtr = (CUdeviceptr)src_dptr,
    };

    ret = CHECK_CU(cu->cuTexObjectCreate(&tex, &res_desc, &tex_desc, NULL));
    if (ret < 0)
        goto exit;

    ret = CHECK_CU(cu->cuLaunchKernel(func,
                                      DIV_UP(dst_width, BLOCKX), DIV_UP(dst_height, BLOCKY), 1,
                                      BLOCKX, BLOCKY, 1, 0, 0/*s->cu_stream*/, args, NULL));

exit:
    if (tex)
        CHECK_CU(cu->cuTexObjectDestroy(tex));

    return ret;
}

static int call_meanfilter_kernel(AVFilterContext *ctx, CUfunction func,
                                  float *src_dptr, float *dst_dptr,
                                  int width, int height, int pitch, int radius)
{
    CUDAGuidedContext *s = ctx->priv;
    CudaFunctions *cu = s->hwctx->internal->cuda_dl;
    CUdeviceptr src_devptr = (CUdeviceptr)src_dptr;
    CUdeviceptr dst_devptr = (CUdeviceptr)dst_dptr;
    void *args[] = { &src_devptr, &dst_devptr, &width, &height, &pitch, &radius };
    int ret;

    ret = CHECK_CU(cu->cuLaunchKernel(func,
                                      DIV_UP(width, BLOCKX), DIV_UP(height, BLOCKY), 1,
                                      BLOCKX, BLOCKY, 1, 0, 0, args, NULL));

    return ret;
}

static int call_floatsquare_kernel(AVFilterContext *ctx, CUfunction func,
                                   float *data_dptr, int width, int height, int pitch)
{
    CUDAGuidedContext *s = ctx->priv;
    CudaFunctions *cu = s->hwctx->internal->cuda_dl;
    CUdeviceptr data_devptr = (CUdeviceptr)data_dptr;
    void *args[] = { &data_devptr, &width, &height, &pitch };
    int ret;

    ret = CHECK_CU(cu->cuLaunchKernel(func,
                                      DIV_UP(width, BLOCKX), DIV_UP(height, BLOCKY), 1,
                                      BLOCKX, BLOCKY, 1, 0, 0, args, NULL));

    return ret;
}

static int call_computeab_kernel(AVFilterContext *ctx, CUfunction func,
                                 float *meanIP_dptr, float *corrIP_dptr,
                                 float *a_dptr, float *b_dptr,
                                 int width, int height, int pitch, float delta)
{
    CUDAGuidedContext *s = ctx->priv;
    CudaFunctions *cu = s->hwctx->internal->cuda_dl;
    CUdeviceptr meanIP_devptr = (CUdeviceptr)meanIP_dptr;
    CUdeviceptr corrIP_devptr = (CUdeviceptr)corrIP_dptr;
    CUdeviceptr a_devptr = (CUdeviceptr)a_dptr;
    CUdeviceptr b_devptr = (CUdeviceptr)b_dptr;
    float deltaVal = delta;
    void *args[] = { &meanIP_devptr, &corrIP_devptr, &a_devptr, &b_devptr, 
                     &width, &height, &pitch, &deltaVal };
    int ret;

    ret = CHECK_CU(cu->cuLaunchKernel(func,
                                      DIV_UP(width, BLOCKX), DIV_UP(height, BLOCKY), 1,
                                      BLOCKX, BLOCKY, 1, 0, 0, args, NULL));

    return ret;
}

static int call_float2char_kernel_tex(AVFilterContext *ctx, CUfunction func,
                                      uint8_t *src_dptr, float *mean_dptr, float *corr_dptr,
                                      uint8_t *dst_dptr, int width, int height, int src_pitch, int dst_pitch)
{
    CUDAGuidedContext *s = ctx->priv;
    CudaFunctions *cu = s->hwctx->internal->cuda_dl;
    CUdeviceptr dst_devptr = (CUdeviceptr)dst_dptr;
    CUdeviceptr mean_devptr = (CUdeviceptr)mean_dptr;
    CUdeviceptr corr_devptr = (CUdeviceptr)corr_dptr;
    CUtexObject tex = 0;
    void *args[] = { &tex, &mean_devptr, &corr_devptr, &dst_devptr, 
                     &width, &height, &src_pitch, &dst_pitch };
    int ret;

    CUDA_TEXTURE_DESC tex_desc = {
        .filterMode = CU_TR_FILTER_MODE_LINEAR,
        .flags = CU_TRSF_READ_AS_INTEGER,
    };

    CUDA_RESOURCE_DESC res_desc = {
        .resType = CU_RESOURCE_TYPE_PITCH2D,
        .res.pitch2D.format = CU_AD_FORMAT_UNSIGNED_INT8,
        .res.pitch2D.numChannels = 1,
        .res.pitch2D.width = width,
        .res.pitch2D.height = height,
        .res.pitch2D.pitchInBytes = src_pitch,
        .res.pitch2D.devPtr = (CUdeviceptr)src_dptr,
    };

    ret = CHECK_CU(cu->cuTexObjectCreate(&tex, &res_desc, &tex_desc, NULL));
    if (ret < 0)
        goto exit;

    ret = CHECK_CU(cu->cuLaunchKernel(func,
                                      DIV_UP(width, BLOCKX), DIV_UP(height, BLOCKY), 1,
                                      BLOCKX, BLOCKY, 1, 0, 0, args, NULL));

exit:
    if (tex)
        CHECK_CU(cu->cuTexObjectDestroy(tex));

    return ret;
}

static int call_copy_kernel_tex(AVFilterContext *ctx, CUfunction func,
                                uint8_t *src_dptr, uint8_t *dst_dptr,
                                int width, int height, int src_pitch, int dst_pitch)
{
    CUDAGuidedContext *s = ctx->priv;
    CudaFunctions *cu = s->hwctx->internal->cuda_dl;
    CUdeviceptr dst_devptr = (CUdeviceptr)dst_dptr;
    CUtexObject tex = 0;
    void *args[] = { &tex, &dst_devptr, &width, &height, &dst_pitch };
    int ret;

    CUDA_TEXTURE_DESC tex_desc = {
        .filterMode = CU_TR_FILTER_MODE_LINEAR,
        .flags = CU_TRSF_READ_AS_INTEGER,
    };

    CUDA_RESOURCE_DESC res_desc = {
        .resType = CU_RESOURCE_TYPE_PITCH2D,
        .res.pitch2D.format = CU_AD_FORMAT_UNSIGNED_INT8,
        .res.pitch2D.numChannels = 1,
        .res.pitch2D.width = width,
        .res.pitch2D.height = height,
        .res.pitch2D.pitchInBytes = src_pitch,
        .res.pitch2D.devPtr = (CUdeviceptr)src_dptr,
    };

    ret = CHECK_CU(cu->cuTexObjectCreate(&tex, &res_desc, &tex_desc, NULL));
    if (ret < 0)
        goto exit;

    ret = CHECK_CU(cu->cuLaunchKernel(func,
                                      DIV_UP(width, BLOCKX), DIV_UP(height, BLOCKY), 1,
                                      BLOCKX, BLOCKY, 1, 0, 0, args, NULL));

exit:
    if (tex)
        CHECK_CU(cu->cuTexObjectDestroy(tex));

    return ret;
}

static int cudaguided_filter_plane(AVFilterContext *ctx, AVFrame *out, AVFrame *in, int plane)
{
    CUDAGuidedContext *s = ctx->priv;
    int cw = AV_CEIL_RSHIFT(in->width, s->hsub);
    int ch = AV_CEIL_RSHIFT(in->height, s->vsub);
    int w[3] = { in->width, cw, cw};
    int h[3] = { in->height, ch, ch};
    int width = w[plane];
    int height = h[plane];
    int radius = s->radius[plane];
    float delta = s->eps[plane];
    int ret = 0;

    if (radius == 0) {
        // Just copy if radius is 0
        return call_copy_kernel_tex(ctx, s->cu_func_copy,
                                    in->data[plane], out->data[plane],
                                    width, height, in->linesize[plane], out->linesize[plane]);
    }

    // Step 1: Convert char to float
    ret = call_char2float_kernel_tex(ctx, s->cu_func_char2float,
                                     in->data[plane], width, height, in->linesize[plane],
                                     s->cu_temp_data, width, height, width,
                                     1, radius);
    if (ret < 0)
        return ret;

    // Step 2: Mean filter on data -> meanIP
    ret = call_meanfilter_kernel(ctx, s->cu_func_meanfilter,
                                 (float*)s->cu_temp_data, (float*)s->cu_temp_meanIP,
                                 width, height, width, radius);
    if (ret < 0)
        return ret;

    // Step 3: Square the data
    ret = call_floatsquare_kernel(ctx, s->cu_func_floatsquare,
                                  (float*)s->cu_temp_data, width, height, width);
    if (ret < 0)
        return ret;

    // Step 4: Mean filter on squared data -> corrIP
    ret = call_meanfilter_kernel(ctx, s->cu_func_meanfilter,
                                 (float*)s->cu_temp_data, (float*)s->cu_temp_corrIP,
                                 width, height, width, radius);
    if (ret < 0)
        return ret;

    // Step 5: Compute a and b from meanIP and corrIP
    ret = call_computeab_kernel(ctx, s->cu_func_computeab,
                                (float*)s->cu_temp_meanIP, (float*)s->cu_temp_corrIP,
                                (float*)s->cu_temp_a, (float*)s->cu_temp_b,
                                width, height, width, delta);
    if (ret < 0)
        return ret;

    // Step 6: Mean filter on a -> meanIP (reuse meanIP buffer)
    ret = call_meanfilter_kernel(ctx, s->cu_func_meanfilter,
                                 (float*)s->cu_temp_a, (float*)s->cu_temp_meanIP,
                                 width, height, width, radius);
    if (ret < 0)
        return ret;

    // Step 7: Mean filter on b -> corrIP (reuse corrIP buffer)
    ret = call_meanfilter_kernel(ctx, s->cu_func_meanfilter,
                                 (float*)s->cu_temp_b, (float*)s->cu_temp_corrIP,
                                 width, height, width, radius);
    if (ret < 0)
        return ret;

    // Step 8: Convert float to char using meanIP and corrIP
    ret = call_float2char_kernel_tex(ctx, s->cu_func_float2char,
                                     in->data[plane],
                                     (float*)s->cu_temp_meanIP, (float*)s->cu_temp_corrIP,
                                     out->data[plane],
                                     width, height, in->linesize[plane], out->linesize[plane]);
    if (ret < 0)
        return ret;

    return 0;
}

static int cudaguided_filter(AVFilterContext *ctx, AVFrame *out, AVFrame *in)
{
    CUDAGuidedContext *s = ctx->priv;
    AVFilterLink* outlink = ctx->outputs[0];
    AVFrame *src = in;
    int ret = 0;
    int plane;

    // Process each plane - output directly to out frame
    for (plane = 0; plane < 3; plane++) {
        if (in->data[plane] == NULL || out->data[plane] == NULL)
            continue;

        ret = cudaguided_filter_plane(ctx, out, src, plane);
        if (ret < 0)
            return ret;
    }
    
    ret = av_frame_copy_props(out, in);
    if (ret < 0)
        return ret;

    return 0;
}

static int cudaguided_filter_frame(AVFilterLink *link, AVFrame *in)
{
    AVFilterContext       *ctx = link->dst;
    CUDAGuidedContext        *s = ctx->priv;
    AVFilterLink      *outlink = ctx->outputs[0];
    CudaFunctions          *cu = s->hwctx->internal->cuda_dl;

    AVFrame *out = NULL;
    CUcontext dummy;
    int ret = 0;

    out = ff_get_video_buffer(outlink, outlink->w, outlink->h);
    if (!out) {
        ret = AVERROR(ENOMEM);
        goto fail;
    }

    ret = CHECK_CU(cu->cuCtxPushCurrent(s->hwctx->cuda_ctx));
    if (ret < 0)
        goto fail;

    ret = cudaguided_filter(ctx, out, in);

    CHECK_CU(cu->cuCtxPopCurrent(&dummy));
    if (ret < 0)
        goto fail;

    av_reduce(&out->sample_aspect_ratio.num, &out->sample_aspect_ratio.den,
              (int64_t)in->sample_aspect_ratio.num * outlink->h * link->w,
              (int64_t)in->sample_aspect_ratio.den * outlink->w * link->h,
              INT_MAX);

    av_frame_free(&in);

    return ff_filter_frame(outlink, out);
fail:
    av_frame_free(&in);
    av_frame_free(&out);
    return ret;
}

static int config_input(AVFilterLink *inlink)
{
    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(inlink->format);
    AVFilterContext    *ctx = inlink->dst;
    CUDAGuidedContext *s = ctx->priv;
    int ret;
    
    s->hsub = desc->log2_chroma_w;
    s->vsub = desc->log2_chroma_h;

    if (s->luma_param.radius < 0 || s->luma_param.radius > 32) {
        av_log(ctx, AV_LOG_ERROR,
               "Invalid luma radius %d, should be in [0,32]\n",
               s->luma_param.radius);
        return AVERROR(EINVAL);
    }
    if (s->luma_param.eps < 0 || s->luma_param.eps > 1) {
        av_log(ctx, AV_LOG_ERROR,
               "Invalid luma eps:%f, should be in [0 1]\n",
               s->luma_param.eps);
        return AVERROR(EINVAL);
    }

    if (s->chroma_param.radius < 0 || s->chroma_param.radius > 32) {
        av_log(ctx, AV_LOG_ERROR,
               "Invalid chroma radius %d, should be in [0,32]\n",
               s->chroma_param.radius);
        return AVERROR(EINVAL);
    }
    if (s->chroma_param.eps < 0 || s->chroma_param.eps > 1) {
        av_log(ctx, AV_LOG_ERROR,
               "Invalid chroma eps:%f, should be in [0 1]\n",
               s->chroma_param.eps);
        return AVERROR(EINVAL);
    }
    
    s->radius[Y] = s->luma_param.radius;
    s->radius[U] = s->radius[V] = s->chroma_param.radius;
    
    s->eps[Y] = s->luma_param.eps;
    s->eps[U] = s->eps[V] = s->chroma_param.eps;
    
    return 0;
}

#define OFFSET(x) offsetof(CUDAGuidedContext, x)
#define FLAGS (AV_OPT_FLAG_FILTERING_PARAM|AV_OPT_FLAG_VIDEO_PARAM)

static const AVOption options[] = {
    { "luma_radius", "Radius of the luma guided filter", OFFSET(luma_param.radius), AV_OPT_TYPE_INT, {.i64 = 1}, 0, 32, .flags = FLAGS },
    { "lr",          "Radius of the luma guided filter", OFFSET(luma_param.radius), AV_OPT_TYPE_INT, {.i64 = 1}, 0, 32, .flags = FLAGS },
    { "luma_eps",    "Eps should be applied to luma",    OFFSET(luma_param.eps), AV_OPT_TYPE_FLOAT, {.dbl = 0.0005}, 0, 1, .flags = FLAGS },
    { "le",          "Eps should be applied to luma",    OFFSET(luma_param.eps), AV_OPT_TYPE_FLOAT, {.dbl = 0.0005}, 0, 1, .flags = FLAGS },
    
    { "chroma_radius", "Radius of the chroma guided filter", OFFSET(chroma_param.radius), AV_OPT_TYPE_INT, {.i64 = 0}, 0, 32, .flags = FLAGS },
    { "cr",            "Radius of the chroma guided filter", OFFSET(chroma_param.radius), AV_OPT_TYPE_INT, {.i64 = 0}, 0, 32, .flags = FLAGS },
    { "chroma_eps",    "Eps should be applied to chroma",    OFFSET(chroma_param.eps), AV_OPT_TYPE_FLOAT, {.dbl = 0.0005}, 0, 1, .flags = FLAGS },
    { "ce",            "Eps should be applied to chroma",    OFFSET(chroma_param.eps), AV_OPT_TYPE_FLOAT, {.dbl = 0.0005}, 0, 1, .flags = FLAGS },
    { NULL }
};

static const AVClass cudaguided_class = {
    .class_name = "guided_cuda",
    .item_name  = av_default_item_name,
    .option     = options,
    .version    = LIBAVUTIL_VERSION_INT,
};

static const AVFilterPad cudaguided_inputs[] = {
    {
        .name        = "default",
        .type        = AVMEDIA_TYPE_VIDEO,
        .config_props = config_input,
        .filter_frame = cudaguided_filter_frame,
    },
};

static const AVFilterPad cudaguided_outputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .config_props = cudaguided_config_props,
    },
};

AVFilter ff_vf_guided_cuda = {
    .name      = "guided_cuda",
    .description = NULL_IF_CONFIG_SMALL("GPU accelerated guided filter"),

    .init          = cudaguided_init,
    .uninit        = cudaguided_uninit,

    .priv_size = sizeof(CUDAGuidedContext),
    .priv_class = &cudaguided_class,
    .query_formats = cudaguided_query_formats,
    .inputs    = cudaguided_inputs,
    .outputs   = cudaguided_outputs,

    .flags_internal = FF_FILTER_FLAG_HWFRAME_AWARE,
};
