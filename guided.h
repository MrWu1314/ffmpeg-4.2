/*
 * Image enhancement filter header
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

#ifndef AVFILTER_UNSHARP_H
#define AVFILTER_UNSHARP_H

#include "config.h"
#include "avfilter.h"
#include "libswscale/swscale.h"
#include "libswscale/swscale_internal.h"
typedef struct QNGuidedFilterParam {
    int radius;
    float eps;
} QNGuidedFilterParam;

#define Y 0
#define U 1
#define V 2
//#define A 3

typedef struct QNGuidedContext {
    const AVClass *class;
    int fast;
    QNGuidedFilterParam luma_param;   ///< luma parameters (width, height, amount)
    QNGuidedFilterParam chroma_param; ///< chroma parameters (width, height, amount)
    int hsub, vsub;
    int radius[3];
    float eps[3]; 
    SwsContext *sws_ctx;
    int nb_threads;
    int (* apply_guided)(AVFilterContext *ctx, AVFrame *in, AVFrame *out, enum AVPixelFormat fmt);
} QNGuidedContext;

#endif /* AVFILTER_UNSHARP_H */
