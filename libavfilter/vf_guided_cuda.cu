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

extern "C" {

__device__ float clamp_float(float val, float min_val, float max_val)
{
    if (val < min_val)
        return min_val;
    else if (val > max_val)
        return max_val;
    else
        return val;
}

__device__ int border_interpolate(int p, int len)
{
    if ((unsigned)p < (unsigned)len)
        return p;
    
    int delta = 1;
    if (len == 1)
        return 0;
    do
    {
        if (p < 0)
            p = -p - 1 + delta;
        else
            p = len - 1 - (p - len) - delta;
    }
    while ((unsigned)p >= (unsigned)len);
    return p;
}

// Kernel to convert unsigned char to float (normalized to [0,1])
__global__ void Guided_char2float(cudaTextureObject_t uchar_tex, float *dst,
                                   int dst_width, int dst_height, int dst_pitch, int radius)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (y < dst_height && x < dst_width)
    {
        float val = tex2D<unsigned char>(uchar_tex, x + 0.5f, y + 0.5f);
        dst[y * dst_pitch + x] = val / 255.0f;
    }
}

// Kernel for mean filter (box filter) using shared memory optimization
__global__ void Guided_meanfilter(float *src, float *dst,
                                  int width, int height, int pitch, int radius)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (y >= height || x >= width)
        return;

    float sum = 0.0f;
    int count = 0;
    int window_size = 2 * radius + 1;
    int total_pixels = window_size * window_size;

    // Compute mean over the window
    for (int dy = -radius; dy <= radius; dy++)
    {
        int sy = border_interpolate(y + dy, height);
        for (int dx = -radius; dx <= radius; dx++)
        {
            int sx = border_interpolate(x + dx, width);
            sum += src[sy * pitch + sx];
            count++;
        }
    }

    dst[y * pitch + x] = sum / (float)total_pixels;
}

// Kernel to square float values
__global__ void Guided_floatsquare(float *data, int width, int height, int pitch)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (y < height && x < width)
    {
        float val = data[y * pitch + x];
        data[y * pitch + x] = val * val;
    }
}

// Kernel to compute a and b coefficients from meanIP and corrIP
__global__ void Guided_computeab(float *meanIP, float *corrIP, float *a, float *b,
                                 int width, int height, int pitch, float delta)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (y < height && x < width)
    {
        int idx = y * pitch + x;
        float mean_val = meanIP[idx];
        float corr_val = corrIP[idx];
        
        // varIP = corrIP - meanIP * meanIP
        float varIP = corr_val - mean_val * mean_val;
        
        // a = varIP / (varIP + delta)
        float a_val = varIP / (varIP + delta);
        a[idx] = a_val;
        
        // b = meanIP - a * meanIP
        b[idx] = mean_val - a_val * mean_val;
    }
}

// Kernel to convert float back to unsigned char using meanIP and corrIP
__global__ void Guided_float2char(cudaTextureObject_t src_tex, float *meanIP, float *corrIP,
                                   unsigned char *dst, int width, int height, int src_pitch, int dst_pitch)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (y < height && x < width)
    {
        int idx = y * width + x;  // meanIP and corrIP use width as pitch (they're allocated with width pitch)
        float src_val = tex2D<unsigned char>(src_tex, x + 0.5f, y + 0.5f) / 255.0f;
        float mean_val = meanIP[idx];
        float corr_val = corrIP[idx];
        
        // dst = (meanIP * src + corrIP) * 255
        float result = (mean_val * src_val + corr_val) * 255.0f;
        result = clamp_float(result, 0.0f, 255.0f);
        dst[y * dst_pitch + x] = (unsigned char)result;
    }
}

// Simple copy kernel
__global__ void Guided_copy(cudaTextureObject_t src_tex, unsigned char *dst,
                            int dst_width, int dst_height, int dst_pitch)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (y < dst_height && x < dst_width)
    {
        unsigned char val = tex2D<unsigned char>(src_tex, x + 0.5f, y + 0.5f);
        dst[y * dst_pitch + x] = val;
    }
}

}
