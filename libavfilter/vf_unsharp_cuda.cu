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

extern "C" {

// int model3x3[3][3] = {
//     {1, 2, 1},
//     {2, 4, 2},
//     {1, 2, 1}
// };
// int model5x5[5][5] = {
//     {1, 4, 6, 4,1},
//     {4,16,24,16,4},
//     {6,24,36,24,6},
//     {4,16,24,16,4},
//     {1, 4, 6, 4,1}
// };
// int model7x7[7][7] = {
//     {1,  6,   15,  20,  15,  6,   1 },
//     {6,  36,  90,  120, 90,  36,  6 },
//     {15, 90,  225, 300, 225, 90,  15},
//     {20, 120, 300, 400, 300, 120, 20},
//     {15, 90,  225, 300, 225, 90,  15},
//     {6,  36,  90,  120, 90,  36,  6 },
//     {1,  6,   15,  20,  15,  6,   1 }
   // };
__device__ int clamp(int val, int min, int max)
{
    if (val < min)
        return min;
    else if (val > max)
        return max;
    else
        return val;
}
__global__ void Unsharp_3x3_uchar_tex(cudaTextureObject_t uchar_tex,unsigned char *dst,
                                    int dst_width, int dst_height, int dst_pitch,int amount)
{
    int model3x3[3][3] = {
        {1, 2, 1},
        {2, 4, 2},
        {1, 2, 1}
    };
    int pix3x3[3][3] = {
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0}
    };
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    int x1= x - 1; //X o x
    if (x1 < 0) x1 = 0;
    int x2= x + 1; //x o X
    if (x2 >=dst_width ) x2 = dst_width-1;

    int y1= y - 1;
    if (y1 < 0) y1 = 0;
    int y2= y + 1;
    if (y2 >= dst_height) y2 = dst_height-1;

    if (y < dst_height && x < dst_width)
    {   
        float xo= x + 0.5f;
        float yo= y + 0.5f;
        float xo1= x1 + 0.5f;
        float xo2= x2 + 0.5f;

        float yo1= y1 + 0.5f;
        float yo2= y2 + 0.5f;

        //上一行
        pix3x3[0][0] = tex2D<unsigned char>(uchar_tex, xo1, yo1);
        pix3x3[0][1] = tex2D<unsigned char>(uchar_tex, xo, yo1);
        pix3x3[0][2] = tex2D<unsigned char>(uchar_tex, xo2, yo1);
        //当前行
        pix3x3[1][0] = tex2D<unsigned char>(uchar_tex, xo1, yo);
        pix3x3[1][1] = tex2D<unsigned char>(uchar_tex, xo, yo);
        pix3x3[1][2] = tex2D<unsigned char>(uchar_tex, xo2, yo);
        //下一行
        pix3x3[2][0] = tex2D<unsigned char>(uchar_tex, xo1, yo2);
        pix3x3[2][1] = tex2D<unsigned char>(uchar_tex, xo, yo2);
        pix3x3[2][2] = tex2D<unsigned char>(uchar_tex, xo2, yo2);

        int sum = 0;
        int scalebits = 4;//(radius + radius)*2
        int halfscale = 1 << (scalebits-1);
        for(int i = 0;i < 3;i++)
        {
            sum += pix3x3[i][0] * model3x3[i][0];
            sum += pix3x3[i][1] * model3x3[i][1];
            sum += pix3x3[i][2] * model3x3[i][2];
        }
        int res = (int32_t)pix3x3[1][1] + ((((int32_t) pix3x3[1][1] - (int32_t)((sum + halfscale) >> scalebits)) * amount) >> 16);
        dst[y*dst_pitch+x] = (unsigned char)(clamp(res,0,255));
    }
}

__global__ void Unsharp_5x5_uchar_tex(cudaTextureObject_t uchar_tex,unsigned char *dst,
                                    int dst_width, int dst_height, int dst_pitch,int amount)
{
    int model5x5[5][5] = {
        {1, 4, 6, 4,1},
        {4,16,24,16,4},
        {6,24,36,24,6},
        {4,16,24,16,4},
        {1, 4, 6, 4,1}
    };

    int pix5x5[5][5] = {
        {0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0}
    };
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x1= x - 2; //Xx o xx
    if (x1 < 0) x1 = 0;
    int x2= x - 1; //xX o xx
    if (x2 < 0) x2 = 0;
    int x3= x + 1; //xx o Xx
    if (x3 >=dst_width ) x3 = dst_width-1;
    int x4= x + 2; //xx o xX
    if (x4 >=dst_width ) x4 = dst_width-1;

    int y1= y - 2;
    if (y1 < 0) y1 = 0;
    int y2= y - 1;
    if (y2 < 0) y2 = 0;
    int y3= y + 1;
    if (y3 >= dst_height) y3 = dst_height-1;
    int y4= y + 2;
    if (y4 >= dst_height) y4 = dst_height-1;

    if (y < dst_height && x < dst_width)
    {   
        float xo= x + 0.5f;
        float yo= y + 0.5f;
        
        float xo1= x1 + 0.5f;
        float xo2= x2 + 0.5f;
        float xo3= x3 + 0.5f; 
        float xo4= x4 + 0.5f; 

        float yo1= y1 + 0.5f;
        float yo2= y2 + 0.5f;
        float yo3= y3 + 0.5f;
        float yo4= y4 + 0.5f;
        //上一行
        pix5x5[0][0] = tex2D<unsigned char>(uchar_tex, xo1, yo1);
        pix5x5[0][1] = tex2D<unsigned char>(uchar_tex, xo2, yo1);
        pix5x5[0][2] = tex2D<unsigned char>(uchar_tex, xo, yo1);
        pix5x5[0][3] = tex2D<unsigned char>(uchar_tex, xo3, yo1);
        pix5x5[0][4] = tex2D<unsigned char>(uchar_tex, xo4, yo1);

        pix5x5[1][0] = tex2D<unsigned char>(uchar_tex, xo1, yo2);
        pix5x5[1][1] = tex2D<unsigned char>(uchar_tex, xo2, yo2);
        pix5x5[1][2] = tex2D<unsigned char>(uchar_tex, xo, yo2);
        pix5x5[1][3] = tex2D<unsigned char>(uchar_tex, xo3, yo2);
        pix5x5[1][4] = tex2D<unsigned char>(uchar_tex, xo4, yo2);
       
        //当前行
        pix5x5[2][0] = tex2D<unsigned char>(uchar_tex, xo1, yo);
        pix5x5[2][1] = tex2D<unsigned char>(uchar_tex, xo2, yo);
        pix5x5[2][2] = tex2D<unsigned char>(uchar_tex, xo, yo);
        pix5x5[2][3] = tex2D<unsigned char>(uchar_tex, xo3, yo);
        pix5x5[2][4] = tex2D<unsigned char>(uchar_tex, xo4, yo);
        
        //下方行
        pix5x5[3][0] = tex2D<unsigned char>(uchar_tex, xo1, yo3);
        pix5x5[3][1] = tex2D<unsigned char>(uchar_tex, xo2, yo3);
        pix5x5[3][2] = tex2D<unsigned char>(uchar_tex, xo, yo3);
        pix5x5[3][3] = tex2D<unsigned char>(uchar_tex, xo3, yo3);
        pix5x5[3][4] = tex2D<unsigned char>(uchar_tex, xo4, yo3);
        
        pix5x5[4][0] = tex2D<unsigned char>(uchar_tex, xo1, yo4);
        pix5x5[4][1] = tex2D<unsigned char>(uchar_tex, xo2, yo4);
        pix5x5[4][2] = tex2D<unsigned char>(uchar_tex, xo, yo4);
        pix5x5[4][3] = tex2D<unsigned char>(uchar_tex, xo3, yo4);
        pix5x5[4][4] = tex2D<unsigned char>(uchar_tex, xo4, yo4);

        int sum = 0;//
        int scalebits = 8;//(radius + radius)*2
        int halfscale = 1 << (scalebits-1);
        for(int i = 0;i < 5;i++)
        {
            sum += pix5x5[i][0] * model5x5[i][0];
            sum += pix5x5[i][1] * model5x5[i][1];
            sum += pix5x5[i][2] * model5x5[i][2];
            sum += pix5x5[i][3] * model5x5[i][3];
            sum += pix5x5[i][4] * model5x5[i][4];
        }
        int res = (int32_t)pix5x5[2][2] + ((((int32_t) pix5x5[2][2] - (int32_t)((sum + halfscale) >> scalebits)) * amount) >> 16);
        dst[y*dst_pitch+x] = (unsigned char)(clamp(res,0,255));
    }
}

__global__ void Unsharp_7x7_uchar_tex(cudaTextureObject_t uchar_tex,unsigned char *dst,
                                    int dst_width, int dst_height, int dst_pitch,int amount)
{
    int model7x7[7][7] = {
        {1,  6,   15,  20,  15,  6,   1 },
        {6,  36,  90,  120, 90,  36,  6 },
        {15, 90,  225, 300, 225, 90,  15},
        {20, 120, 300, 400, 300, 120, 20},
        {15, 90,  225, 300, 225, 90,  15},
        {6,  36,  90,  120, 90,  36,  6 },
        {1,  6,   15,  20,  15,  6,   1 }
    };

    int pix7x7[7][7] = {
        {0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0}
    };
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x1= x - 3; //Xxx o xxx
    if (x1 < 0) x1 = 0;
    int x2= x - 2; //xXx o xxx
    if (x2 < 0) x2 = 0;
    int x3= x - 1; //xxX o xxx
    if (x3 < 0) x3 = 0;
    int x4= x + 1; //xxx o Xxx
    if (x4 >=dst_width ) x4 = dst_width-1;
    int x5= x + 2; //xxx o xXx
    if (x5 >=dst_width ) x5 = dst_width-1;
    int x6= x + 3; //xxx o xxX
    if (x6 >=dst_width ) x6 = dst_width-1;

    int y1= y - 3;
    if (y1 < 0) y1 = 0;
    int y2= y - 2;
    if (y2 < 0) y2 = 0;
    int y3= y - 1;
    if (y3 < 0) y3 = 0;
    int y4= y + 1;
    if (y4 >= dst_height) y4 = dst_height-1;
    int y5= y + 2;
    if (y5 >= dst_height) y5 = dst_height-1;
    int y6= y + 3;
    if (y6 >= dst_height) y6 = dst_height-1;

    if (y < dst_height && x < dst_width)
    {   
        float xo= x + 0.5f;
        float yo= y + 0.5f;
        
        float xo1= x1 + 0.5f;
        float xo2= x2 + 0.5f;
        float xo3= x3 + 0.5f; 
        float xo4= x4 + 0.5f; 
        float xo5= x5 + 0.5f; 
        float xo6= x6 + 0.5f; 
        
        float yo1= y1 + 0.5f;
        float yo2= y2 + 0.5f;
        float yo3= y3 + 0.5f;
        float yo4= y4 + 0.5f;
        float yo5= y5 + 0.5f;
        float yo6= y6 + 0.5f;

        //上一行
        pix7x7[0][0] = tex2D<unsigned char>(uchar_tex, xo1, yo1);
        pix7x7[0][1] = tex2D<unsigned char>(uchar_tex, xo2, yo1);
        pix7x7[0][2] = tex2D<unsigned char>(uchar_tex, xo3, yo1);
        pix7x7[0][3] = tex2D<unsigned char>(uchar_tex, xo, yo1);
        pix7x7[0][4] = tex2D<unsigned char>(uchar_tex, xo4, yo1);
        pix7x7[0][5] = tex2D<unsigned char>(uchar_tex, xo4, yo1);
        pix7x7[0][6] = tex2D<unsigned char>(uchar_tex, xo6, yo1);

        pix7x7[1][0] = tex2D<unsigned char>(uchar_tex, xo1, yo2);
        pix7x7[1][1] = tex2D<unsigned char>(uchar_tex, xo2, yo2);
        pix7x7[1][2] = tex2D<unsigned char>(uchar_tex, xo3, yo2);
        pix7x7[1][3] = tex2D<unsigned char>(uchar_tex, xo, yo2);
        pix7x7[1][4] = tex2D<unsigned char>(uchar_tex, xo4, yo2);
        pix7x7[1][5] = tex2D<unsigned char>(uchar_tex, xo5, yo2);
        pix7x7[1][6] = tex2D<unsigned char>(uchar_tex, xo6, yo2);
       
        pix7x7[2][0] = tex2D<unsigned char>(uchar_tex, xo1, yo3);
        pix7x7[2][1] = tex2D<unsigned char>(uchar_tex, xo2, yo3);
        pix7x7[2][2] = tex2D<unsigned char>(uchar_tex, xo3, yo3);
        pix7x7[2][3] = tex2D<unsigned char>(uchar_tex, xo, yo3);
        pix7x7[2][4] = tex2D<unsigned char>(uchar_tex, xo4, yo3);
        pix7x7[2][5] = tex2D<unsigned char>(uchar_tex, xo5, yo3);
        pix7x7[2][6] = tex2D<unsigned char>(uchar_tex, xo6, yo3);
       //当前行
        pix7x7[3][0] = tex2D<unsigned char>(uchar_tex, xo1, yo);
        pix7x7[3][1] = tex2D<unsigned char>(uchar_tex, xo2, yo);
        pix7x7[3][2] = tex2D<unsigned char>(uchar_tex, xo3, yo);
        pix7x7[3][3] = tex2D<unsigned char>(uchar_tex, xo, yo);
        pix7x7[3][4] = tex2D<unsigned char>(uchar_tex, xo4, yo);
        pix7x7[3][5] = tex2D<unsigned char>(uchar_tex, xo5, yo);
        pix7x7[3][6] = tex2D<unsigned char>(uchar_tex, xo6, yo);
        //下方行
        pix7x7[4][0] = tex2D<unsigned char>(uchar_tex, xo1, yo4);
        pix7x7[4][1] = tex2D<unsigned char>(uchar_tex, xo2, yo4);
        pix7x7[4][2] = tex2D<unsigned char>(uchar_tex, xo3, yo4);
        pix7x7[4][3] = tex2D<unsigned char>(uchar_tex, xo, yo4);
        pix7x7[4][4] = tex2D<unsigned char>(uchar_tex, xo4, yo4);
        pix7x7[4][5] = tex2D<unsigned char>(uchar_tex, xo5, yo4);
        pix7x7[4][6] = tex2D<unsigned char>(uchar_tex, xo6, yo4);

        pix7x7[5][0] = tex2D<unsigned char>(uchar_tex, xo1, yo5);
        pix7x7[5][1] = tex2D<unsigned char>(uchar_tex, xo2, yo5);
        pix7x7[5][2] = tex2D<unsigned char>(uchar_tex, xo3, yo5);
        pix7x7[5][3] = tex2D<unsigned char>(uchar_tex, xo, yo5);
        pix7x7[5][4] = tex2D<unsigned char>(uchar_tex, xo4, yo5);
        pix7x7[5][5] = tex2D<unsigned char>(uchar_tex, xo5, yo5);
        pix7x7[5][6] = tex2D<unsigned char>(uchar_tex, xo6, yo5);

        pix7x7[6][0] = tex2D<unsigned char>(uchar_tex, xo1, yo6);
        pix7x7[6][1] = tex2D<unsigned char>(uchar_tex, xo2, yo6);
        pix7x7[6][2] = tex2D<unsigned char>(uchar_tex, xo3, yo6);
        pix7x7[6][3] = tex2D<unsigned char>(uchar_tex, xo, yo6);
        pix7x7[6][4] = tex2D<unsigned char>(uchar_tex, xo4, yo6);
        pix7x7[6][5] = tex2D<unsigned char>(uchar_tex, xo5, yo6);
        pix7x7[6][6] = tex2D<unsigned char>(uchar_tex, xo6, yo6);

        int sum = 0;//
        int scalebits = 12;//(radius + radius)*2
        int halfscale = 1 << (scalebits-1);
        for(int i = 0;i < 7;i++)
        {
            sum += pix7x7[i][0] * model7x7[i][0];
            sum += pix7x7[i][1] * model7x7[i][1];
            sum += pix7x7[i][2] * model7x7[i][2];
            sum += pix7x7[i][3] * model7x7[i][3];
            sum += pix7x7[i][4] * model7x7[i][4];
            sum += pix7x7[i][5] * model7x7[i][5];
            sum += pix7x7[i][6] * model7x7[i][6];
        }
        int res = (int32_t)pix7x7[3][3] + ((((int32_t) pix7x7[3][3] - (int32_t)((sum + halfscale) >> scalebits)) * amount) >> 16);
        dst[y*dst_pitch+x] = (unsigned char)(clamp(res,0,255));
    }
}

__global__ void Unsharp_copy_tex(cudaTextureObject_t uchar_tex,unsigned char *dst,
    int dst_width, int dst_height, int dst_pitch,int amount)
{
    int xo = blockIdx.x * blockDim.x + threadIdx.x;
    int yo = blockIdx.y * blockDim.y + threadIdx.y;

    if (yo < dst_height && xo < dst_width)
    {
        int y0 = tex2D<unsigned char>(uchar_tex, xo+0.5, yo+0.5);
        dst[yo*dst_pitch+xo] = (unsigned char)(clamp(y0,0,255));
    }
}

}
