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

#ifndef AVFILTER_IMGENHANCE_H
#define AVFILTER_IMGENHANCE_H

#include "config.h"
#include "avfilter.h"

#define MIN_MATRIX_SIZE 3
#define MAX_MATRIX_SIZE 63

typedef struct EnhanceFilterParam {
    int msize_x;                             ///< matrix width
    int msize_y;                             ///< matrix height
    int amount;                              ///< effect amount
    int steps_x;                             ///< horizontal step count
    int steps_y;                             ///< vertical step count
    int scalebits;                           ///< bits to shift pixel
    int32_t halfscale;                       ///< amount to add to pixel
    uint32_t *sr;                            ///< finite state machine storage within a row
    uint32_t **sc;                           ///< finite state machine storage across rows
    uint32_t tmp[32];
} EnhanceFilterParam;

typedef struct EnhanceContext {
    const AVClass *class;
    int is_guided;
    float eps;
    int lmsize_x, lmsize_y, cmsize_x, cmsize_y;
    float lamount, camount;
    EnhanceFilterParam luma;                 ///< luma parameters (width, height, amount)
    EnhanceFilterParam chroma;               ///< chroma parameters (width, height, amount)
    int hsub, vsub;
    int bitdepth;
    int bps;
    int nb_threads;
    int opencl;
    int (*apply_enhance)(AVFilterContext *ctx, AVFrame *in, AVFrame *out);
    int (*unsharp_slice)(AVFilterContext *ctx, void *arg, int jobnr, int nb_jobs);
    float (*max_pixel)[8];
} EnhanceContext;

#endif /* AVFILTER_IMGENHANCE_H */
