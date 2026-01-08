/*
* Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
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
//#include "libavutil/cuda_check.h"
#include "libavutil/internal.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"
#include "libavutil/imgutils.h"

#include "avfilter.h"
#include "formats.h"
#include "internal.h"
//#include "unsharp_eval.h"
#include "video.h"
//#include "cuda/load_helper.h"

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

typedef struct CUDAUnsharpContext {
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

    int lmsize_x, lmsize_y, cmsize_x, cmsize_y;
    float lamount, camount;
    uint8_t  useTexture;
    CUcontext   cu_ctx;
    CUmodule    cu_module;
    CUfunction  cu_func_uchar_copy;
    CUfunction  cu_func_uchar3x3;
    CUfunction  cu_func_uchar5x5;
    CUfunction  cu_func_uchar7x7;
    CUfunction  cu_func_ushort3x3;
    CUfunction  cu_func_ushort5x5;
    CUfunction  cu_func_ushort7x7;
    // CUstream    cu_stream;
} CUDAUnsharpContext;


static av_cold int cudaunsharp_init(AVFilterContext *ctx)
{
    CUDAUnsharpContext *s = ctx->priv;

    //s->format = AV_PIX_FMT_NONE;
    s->frame = av_frame_alloc();
    if (!s->frame)
        return AVERROR(ENOMEM);

    s->tmp_frame = av_frame_alloc();
    if (!s->tmp_frame)
        return AVERROR(ENOMEM);

    return 0;
}

static av_cold void cudaunsharp_uninit(AVFilterContext *ctx)
{
    CUDAUnsharpContext *s = ctx->priv;
    if (s->hwctx && s->cu_module) {
        CudaFunctions *cu = s->hwctx->internal->cuda_dl;
        CUcontext dummy;

        CHECK_CU(cu->cuCtxPushCurrent(s->hwctx->cuda_ctx));
        CHECK_CU(cu->cuModuleUnload(s->cu_module));
        s->cu_module = NULL;
        CHECK_CU(cu->cuCtxPopCurrent(&dummy));
    }
    av_frame_free(&s->frame);
    av_buffer_unref(&s->frames_ctx);
    av_frame_free(&s->tmp_frame);
}

static int cudaunsharp_query_formats(AVFilterContext *ctx)
{
    static const enum AVPixelFormat pixel_formats[] = {
        AV_PIX_FMT_CUDA, AV_PIX_FMT_NONE,
    };
    AVFilterFormats *pix_fmts = ff_make_format_list(pixel_formats);

    return ff_set_common_formats(ctx, pix_fmts);
}

static av_cold int init_hwframe_ctx(CUDAUnsharpContext *s, AVBufferRef *device_ctx,int width, int height)
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
    CUDAUnsharpContext *s = ctx->priv;
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

static av_cold int cudaunsharp_config_props(AVFilterLink *outlink)
{
    AVFilterContext *ctx = outlink->src;
    AVFilterLink *inlink = outlink->src->inputs[0];
    CUDAUnsharpContext *s  = ctx->priv;
    AVHWFramesContext     *frames_ctx = (AVHWFramesContext*)inlink->hw_frames_ctx->data;
    AVCUDADeviceContext *device_hwctx = frames_ctx->device_ctx->hwctx;
    CUcontext dummy, cuda_ctx = device_hwctx->cuda_ctx;
    CudaFunctions *cu = device_hwctx->internal->cuda_dl;

    int ret;

    extern char vf_unsharp_cuda_ptx[];
    //extern const unsigned char ff_vf_unsharp_cuda_ptx_data[];
    //extern const unsigned int ff_vf_unsharp_cuda_ptx_len;

    s->hwctx = device_hwctx;
    // s->cu_stream = s->hwctx->stream;

    outlink->w = inlink->w;
    outlink->h = inlink->h;

    ret = CHECK_CU(cu->cuCtxPushCurrent(cuda_ctx));
    if (ret < 0)
        goto fail;

    ret = CHECK_CU(cu->cuModuleLoadData(&s->cu_module, vf_unsharp_cuda_ptx));
    if (ret < 0) {
        av_log(ctx, AV_LOG_ERROR, "Error loading module data\n");
        goto fail;
    }
    // ret = ff_cuda_load_module(ctx, s->hwctx, &s->cu_module,
    //                           ff_vf_unsharp_cuda_ptx_data,
    //                           ff_vf_unsharp_cuda_ptx_len);
    if (ret < 0)
        goto fail;

    ret = CHECK_CU(cu->cuModuleGetFunction(&s->cu_func_uchar3x3, s->cu_module, "Unsharp_3x3_uchar_tex"));
    if (ret < 0)
        goto fail;
        
    ret =CHECK_CU(cu->cuModuleGetFunction(&s->cu_func_uchar_copy, s->cu_module, "Unsharp_copy_tex"));
    if (ret < 0)
        goto fail;
    
    ret =CHECK_CU(cu->cuModuleGetFunction(&s->cu_func_uchar5x5, s->cu_module, "Unsharp_5x5_uchar_tex"));
    if (ret < 0)
        goto fail;
        

    ret = CHECK_CU(cu->cuModuleGetFunction(&s->cu_func_uchar7x7, s->cu_module, "Unsharp_7x7_uchar_tex"));
    if (ret < 0)
        goto fail;
    
    CHECK_CU(cu->cuCtxPopCurrent(&dummy));

    ret = init_processing_chain(ctx, inlink->w, inlink->h);
    if (ret < 0)
        return ret;

    av_log(ctx, AV_LOG_INFO, "lmsize_xy:%dx%d ,lamount %f,pix_fmt is %d\n",
           s->lmsize_x,s->lmsize_y,s->lamount,s->pix_fmt);
           
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

static int call_unsharp_kernel_tex(AVFilterContext *ctx, CUfunction func, int channels,
                              uint8_t *src_dptr, int src_width, int src_height, int src_pitch,
                              uint8_t *dst_dptr, int dst_width, int dst_height, int dst_pitch,
                              int pixel_size,float amount)
{
    CUDAUnsharpContext *s = ctx->priv;
    CudaFunctions *cu = s->hwctx->internal->cuda_dl;
    CUdeviceptr dst_devptr = (CUdeviceptr)dst_dptr;
    CUtexObject tex = 0;
    int amountVal = amount * 65535.0;
    void *args_uchar[] = { &tex, &dst_devptr, &dst_width, &dst_height, &dst_pitch,&amountVal};
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
        .res.pitch2D.numChannels = channels,
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
                                      BLOCKX, BLOCKY, 1, 0, 0/*s->cu_stream*/, args_uchar, NULL));

exit:
    if (tex)
        CHECK_CU(cu->cuTexObjectDestroy(tex));

    return ret;
}


static int unsharpcuda_3x3_tex(AVFilterContext *ctx,
                            AVFrame *out, AVFrame *in)
{
    AVHWFramesContext *in_frames_ctx = (AVHWFramesContext*)in->hw_frames_ctx->data;
    CUDAUnsharpContext *s = ctx->priv;

    int ret = 0;

    switch (in_frames_ctx->sw_format) {
    case AV_PIX_FMT_YUV420P:
        call_unsharp_kernel_tex(ctx, s->cu_func_uchar3x3, 1,
                           in->data[0], in->width, in->height, in->linesize[0],
                           out->data[0], out->width, out->height, out->linesize[0],
                           1,s->lamount);
        if (s->camount)
        {
            call_unsharp_kernel_tex(ctx, s->cu_func_uchar3x3, 1,
                            in->data[1], in->width/2, in->height/2, in->linesize[0]/2,
                            out->data[1], out->width/2, out->height/2, out->linesize[0]/2,
                            1,s->camount);
            call_unsharp_kernel_tex(ctx, s->cu_func_uchar3x3, 1,
                            in->data[2], in->width/2, in->height/2, in->linesize[0]/2,
                            out->data[2], out->width/2, out->height/2, out->linesize[0]/2,
                            1,s->camount);
        }
        else
        {
            call_unsharp_kernel_tex(ctx, s->cu_func_uchar_copy, 1,
                           in->data[1], in->width / 2, in->height / 2, in->linesize[1],
                           out->data[1], out->width / 2, out->height / 2, out->linesize[1],
                           1,s->lamount);
            call_unsharp_kernel_tex(ctx, s->cu_func_uchar_copy, 1,
                           in->data[2], in->width / 2, in->height / 2, in->linesize[2],
                           out->data[2], out->width / 2, out->height / 2, out->linesize[2],
                           1,s->lamount);
        }
        
        break;
    case AV_PIX_FMT_NV12:
        call_unsharp_kernel_tex(ctx, s->cu_func_uchar3x3, 1,
                           in->data[0], in->width, in->height, in->linesize[0],
                           out->data[0], out->width, out->height, out->linesize[0],
                           1,s->lamount);
        call_unsharp_kernel_tex(ctx, s->cu_func_uchar_copy, 1,
                           in->data[1], in->width, in->height / 2, in->linesize[1],
                           out->data[1], out->width, out->height / 2, out->linesize[1],
                           1,s->lamount);
        break;
    default:
        ret = AVERROR_BUG;
    }

    return ret;
}

static int unsharpcuda_5x5_tex(AVFilterContext *ctx,
                            AVFrame *out, AVFrame *in)
{
    AVHWFramesContext *in_frames_ctx = (AVHWFramesContext*)in->hw_frames_ctx->data;
    CUDAUnsharpContext *s = ctx->priv;

    int ret = 0;

    switch (in_frames_ctx->sw_format) {
    case AV_PIX_FMT_YUV420P:
        call_unsharp_kernel_tex(ctx, s->cu_func_uchar5x5, 1,
                           in->data[0], in->width, in->height, in->linesize[0],
                           out->data[0], out->width, out->height, out->linesize[0],
                           1,s->lamount);
        if (s->camount)
        {
            call_unsharp_kernel_tex(ctx, s->cu_func_uchar5x5, 1,
                            in->data[1], in->width/2, in->height/2, in->linesize[0]/2,
                            out->data[1], out->width/2, out->height/2, out->linesize[0]/2,
                            1,s->camount);
            call_unsharp_kernel_tex(ctx, s->cu_func_uchar5x5, 1,
                            in->data[2], in->width/2, in->height/2, in->linesize[0]/2,
                            out->data[2], out->width/2, out->height/2, out->linesize[0]/2,
                            1,s->camount);
        }
        else
        {
            call_unsharp_kernel_tex(ctx, s->cu_func_uchar_copy, 1,
                           in->data[1], in->width / 2, in->height / 2, in->linesize[1],
                           out->data[1], out->width / 2, out->height / 2, out->linesize[1],
                           1,s->lamount);
            call_unsharp_kernel_tex(ctx, s->cu_func_uchar_copy, 1,
                           in->data[2], in->width / 2, in->height / 2, in->linesize[2],
                           out->data[2], out->width / 2, out->height / 2, out->linesize[2],
                           1,s->lamount);
        }
        
        break;
    case AV_PIX_FMT_NV12:
        call_unsharp_kernel_tex(ctx, s->cu_func_uchar5x5, 1,
                           in->data[0], in->width, in->height, in->linesize[0],
                           out->data[0], out->width, out->height, out->linesize[0],
                           1,s->lamount);
        call_unsharp_kernel_tex(ctx, s->cu_func_uchar_copy, 1,
                           in->data[1], in->width, in->height / 2, in->linesize[1],
                           out->data[1], out->width, out->height / 2, out->linesize[1],
                           1,s->lamount);
        break;
    default:
        ret = AVERROR_BUG;
    }

    return ret;
}

static int unsharpcuda_7x7_tex(AVFilterContext *ctx,
                            AVFrame *out, AVFrame *in)
{
    AVHWFramesContext *in_frames_ctx = (AVHWFramesContext*)in->hw_frames_ctx->data;
    CUDAUnsharpContext *s = ctx->priv;

    int ret = 0;

    switch (in_frames_ctx->sw_format) {
    case AV_PIX_FMT_YUV420P:
        call_unsharp_kernel_tex(ctx, s->cu_func_uchar7x7, 1,
                           in->data[0], in->width, in->height, in->linesize[0],
                           out->data[0], out->width, out->height, out->linesize[0],
                           1,s->lamount);
        if (s->camount)
        {
            call_unsharp_kernel_tex(ctx, s->cu_func_uchar7x7, 1,
                            in->data[1], in->width/2, in->height/2, in->linesize[0]/2,
                            out->data[1], out->width/2, out->height/2, out->linesize[0]/2,
                            1,s->camount);
            call_unsharp_kernel_tex(ctx, s->cu_func_uchar7x7, 1,
                            in->data[2], in->width/2, in->height/2, in->linesize[0]/2,
                            out->data[2], out->width/2, out->height/2, out->linesize[0]/2,
                            1,s->camount);
        }
        else
        {
            call_unsharp_kernel_tex(ctx, s->cu_func_uchar_copy, 1,
                           in->data[1], in->width / 2, in->height / 2, in->linesize[1],
                           out->data[1], out->width / 2, out->height / 2, out->linesize[1],
                           1,s->lamount);
            call_unsharp_kernel_tex(ctx, s->cu_func_uchar_copy, 1,
                           in->data[2], in->width / 2, in->height / 2, in->linesize[2],
                           out->data[2], out->width / 2, out->height / 2, out->linesize[2],
                           1,s->lamount);
        }
        
        break;
    case AV_PIX_FMT_NV12:
        call_unsharp_kernel_tex(ctx, s->cu_func_uchar7x7, 1,
                           in->data[0], in->width, in->height, in->linesize[0],
                           out->data[0], out->width, out->height, out->linesize[0],
                           1,s->lamount);
        call_unsharp_kernel_tex(ctx, s->cu_func_uchar_copy, 1,
                           in->data[1], in->width, in->height / 2, in->linesize[1],
                           out->data[1], out->width, out->height / 2, out->linesize[1],
                           1,s->lamount);
        break;
    default:
        ret = AVERROR_BUG;
    }

    return ret;
}

static int cudaunsharp_unsharp(AVFilterContext *ctx, AVFrame *out, AVFrame *in)
{
    CUDAUnsharpContext *s = ctx->priv;
    AVFilterLink* outlink = ctx->outputs[0];
    AVFrame *src = in;
    int ret = 0;
    int size = s->lmsize_x >= s->lmsize_y ? s->lmsize_x : s->lmsize_y;
    if(size < 5)
    {
        ret = unsharpcuda_3x3_tex(ctx, s->frame, src);
    }
    else if(size < 7)
    {
        ret = unsharpcuda_5x5_tex(ctx, s->frame, src);
    }
    else //if(size == 7)
    {
        ret = unsharpcuda_7x7_tex(ctx, s->frame, src);
    } 
    
    if (ret < 0)
        return ret;

    src = s->frame;
    ret = av_hwframe_get_buffer(src->hw_frames_ctx, s->tmp_frame, 0);
    if (ret < 0)
        return ret;

    av_frame_move_ref(out, s->frame);
    av_frame_move_ref(s->frame, s->tmp_frame);

    s->frame->width  = outlink->w;
    s->frame->height = outlink->h;

    ret = av_frame_copy_props(out, in);
    if (ret < 0)
        return ret;

    return 0;
}

static int cudaunsharp_filter_frame(AVFilterLink *link, AVFrame *in)
{
    AVFilterContext       *ctx = link->dst;
    CUDAUnsharpContext        *s = ctx->priv;
    AVFilterLink      *outlink = ctx->outputs[0];
    CudaFunctions          *cu = s->hwctx->internal->cuda_dl;

    AVFrame *out = NULL;
    CUcontext dummy;
    int ret = 0;

    out = av_frame_alloc();
    if (!out) {
        ret = AVERROR(ENOMEM);
        goto fail;
    }

    ret = CHECK_CU(cu->cuCtxPushCurrent(s->hwctx->cuda_ctx));
    if (ret < 0)
        goto fail;

    ret = cudaunsharp_unsharp(ctx, out, in);

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

#define OFFSET(x) offsetof(CUDAUnsharpContext, x)
#define FLAGS (AV_OPT_FLAG_FILTERING_PARAM|AV_OPT_FLAG_VIDEO_PARAM)
#define MIN_SIZE 3
#define MAX_SIZE 7
static const AVOption options[] = {
    { "luma_msize_x",   "set luma matrix horizontal size",   OFFSET(lmsize_x), AV_OPT_TYPE_INT,   { .i64 = 5 }, MIN_SIZE, MAX_SIZE, FLAGS },
    { "lx",             "set luma matrix horizontal size",   OFFSET(lmsize_x), AV_OPT_TYPE_INT,   { .i64 = 5 }, MIN_SIZE, MAX_SIZE, FLAGS },
    { "luma_msize_y",   "set luma matrix vertical size",     OFFSET(lmsize_y), AV_OPT_TYPE_INT,   { .i64 = 5 }, MIN_SIZE, MAX_SIZE, FLAGS },
    { "ly",             "set luma matrix vertical size",     OFFSET(lmsize_y), AV_OPT_TYPE_INT,   { .i64 = 5 }, MIN_SIZE, MAX_SIZE, FLAGS },
    { "luma_amount",    "set luma effect strength",          OFFSET(lamount),  AV_OPT_TYPE_FLOAT, { .dbl = 1.5 },       -2,        5, FLAGS },
    { "la",             "set luma effect strength",          OFFSET(lamount),  AV_OPT_TYPE_FLOAT, { .dbl = 1.5 },       -2,        5, FLAGS },
    { "chroma_msize_x", "set chroma matrix horizontal size", OFFSET(cmsize_x), AV_OPT_TYPE_INT,   { .i64 = 5 }, MIN_SIZE, MAX_SIZE, FLAGS },
    { "cx",             "set chroma matrix horizontal size", OFFSET(cmsize_x), AV_OPT_TYPE_INT,   { .i64 = 5 }, MIN_SIZE, MAX_SIZE, FLAGS },
    { "chroma_msize_y", "set chroma matrix vertical size",   OFFSET(cmsize_y), AV_OPT_TYPE_INT,   { .i64 = 5 }, MIN_SIZE, MAX_SIZE, FLAGS },
    { "cy",             "set chroma matrix vertical size",   OFFSET(cmsize_y), AV_OPT_TYPE_INT,   { .i64 = 5 }, MIN_SIZE, MAX_SIZE, FLAGS },
    { "chroma_amount",  "set chroma effect strength",        OFFSET(camount),  AV_OPT_TYPE_FLOAT, { .dbl = 0 },       -2,        5, FLAGS },
    { "ca",             "set chroma effect strength",        OFFSET(camount),  AV_OPT_TYPE_FLOAT, { .dbl = 0 },       -2,        5, FLAGS },
    //{ "useTex",         "use texture to read pic",           OFFSET(useTexture),   AV_OPT_TYPE_BOOL,  { .i64 = 0 },        0,        1, FLAGS },
    { NULL }
};

static const AVClass cudaunsharp_class = {
    .class_name = "cudaunsharp",
    .item_name  = av_default_item_name,
    .option     = options,
    .version    = LIBAVUTIL_VERSION_INT,
};

static const AVFilterPad cudaunsharp_inputs[] = {
    {
        .name        = "default",
        .type        = AVMEDIA_TYPE_VIDEO,
        .filter_frame = cudaunsharp_filter_frame,
    },
};

static const AVFilterPad cudaunsharp_outputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .config_props = cudaunsharp_config_props,
    },
};

AVFilter ff_vf_unsharp_cuda = {
    .name      = "unsharp_cuda",
    .description = NULL_IF_CONFIG_SMALL("GPU accelerated video unsharp"),

    .init          = cudaunsharp_init,
    .uninit        = cudaunsharp_uninit,

    .priv_size = sizeof(CUDAUnsharpContext),
    .priv_class = &cudaunsharp_class,
    .query_formats = cudaunsharp_query_formats,
    .inputs    = cudaunsharp_inputs,
    .outputs   = cudaunsharp_outputs,

    .flags_internal = FF_FILTER_FLAG_HWFRAME_AWARE,
};

