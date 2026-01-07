/*
 * Image enhancement filter
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

#include "libavutil/common.h"
#include "libavutil/imgutils.h"
#include "libavutil/mem.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"
#include "libavutil/x86/cpu.h"
#include "libavutil/avstring.h"

#include "avfilter.h"
#include "formats.h"
#include "internal.h"
#include "video.h"
#include "imgenhance.h"

#include <pthread.h>

#define ENHANCE_VF_CPU_FLG_AVX2 (HAVE_AVX2)
#define ENHANCE_VF_THREAD_FLG (HAVE_THREADS)

#if HAVE_AVX2_EXTERNAL
extern void ff_MeanFilter_Piex2Float_16_avx2(uint16_t* srcData,float* dstData,int width,int height,int src_stride,float *maxData);
extern void ff_MeanFilter_Float2Piex_16_avx2(uint16_t* srcData,uint16_t* dstData,float* pMeanIP,float *pCorrIP,int width,int height,int src_stride,int dst_stride,float *maxData);
#else
void ff_MeanFilter_Piex2Float_16_avx2(uint16_t* srcData,float* dstData,int width,int height,int src_stride,float *maxData){};
void ff_MeanFilter_Float2Piex_16_avx2(uint16_t* srcData,uint16_t* dstData,float* pMeanIP,float *pCorrIP,int width,int height,int src_stride,int dst_stride,float *maxData){};
#endif
static float __attribute__((aligned(32))) max_pixel_10bit[2][8] = {
    {1.0/1023.0, 1.0/1023.0, 1.0/1023.0, 1.0/1023.0,
     1.0/1023.0, 1.0/1023.0, 1.0/1023.0, 1.0/1023.0},
    {1023.0, 1023.0, 1023.0, 1023.0,
     1023.0, 1023.0, 1023.0, 1023.0}
};

static float __attribute__((aligned(32))) max_pixel_12bit[2][8] = {
    {1.0/4095.0, 1.0/4095.0, 1.0/4095.0, 1.0/4095.0,
     1.0/4095.0, 1.0/4095.0, 1.0/4095.0, 1.0/4095.0},
    {4095.0, 4095.0, 4095.0, 4095.0,
     4095.0, 4095.0, 4095.0, 4095.0}
};
/* Guided filter synchronization structure */
typedef struct EnhanceGuidedFilterSync {
    int init_flg;
    int result_cnt;
    pthread_mutex_t mutex;
    pthread_cond_t  cond;
} EnhanceGuidedFilterSync;

static EnhanceGuidedFilterSync g_guided_filter_sync = {0};

typedef struct GuidedThreadData {
    AVFrame* in;
    AVFrame* out;
} GuidedThreadData;

/* Filter parameter structure for guided filter */
typedef struct FilterParam {
    unsigned char *src_data;
    unsigned char *dst_data;
    int dst_linesize;
    int src_linesize;
    int src_stride;
    int dst_stride;
    int width;
    int height;
    int radius;
    float delta;
    int mask; /* Use low 2 bits: bit0=1 means first slice, bit1=1 means last slice */
    int thread_num;
} FilterParam;

static inline int borderInterpolate( int p, int len )
{
    if( (unsigned)p < (unsigned)len )
        ;
    
    else
    {
        int delta = 1;
        if( len == 1 )
            return 0;
        do
        {
            if( p < 0 )
                p = -p - 1 + delta;
            else
                p = len - 1 - (p - len) - delta;
        }
        while( (unsigned)p >= (unsigned)len );
    }
    return p;
}

/* Function pointer types for AVX2 optimized functions */
typedef void (*MeanFilterChar2FloatFunc)(unsigned char *srcData, float *dstData,
                                         int width, int height, int src_stride);
typedef void (*MeanFilterFloat2CharFunc)(unsigned char *srcCharData,
                                         unsigned char *dstData,
                                         float *pMeanIP, float *pCorrIP,
                                         int width, int height,
                                         int src_stride, int dst_stride);
typedef void (*MeanFilterChar2Float16Func)(uint16_t *srcData, float *dstData,
                                           int width, int height,
                                           int src_stride, float maxData);
typedef void (*MeanFilterFloat2Char16Func)(uint16_t *srcCharData,
                                           uint16_t *dstData,
                                           float *pMeanIP, float *pCorrIP,
                                           int width, int height,
                                           int src_stride, int dst_stride,
                                           float maxData);
typedef void (*MeanFilterFloatSquareFunc)(float *pData, int width, int height);
typedef void (*MeanFilterMeanIPCorrIPABFunc)(float *pMeanIP, float *pCorrIP,
                                             float *pA, float *pB,
                                             int width, int height, float delta);
typedef void (*MeanFilterSize3x3fFunc)(float *srcData, float *dstData,
                                       int width, int height, int src_stride);
typedef void (*MeanFilterFunc)(float *srcData, float *dstData,
                               int width, int height, int radius, int mask);

/* AVX2 optimized function pointers */
typedef struct EnhanceDSPContext {
    MeanFilterChar2FloatFunc     char2float;
    MeanFilterFloat2CharFunc     float2char;
    MeanFilterChar2Float16Func   char2float16;
    MeanFilterFloat2Char16Func   float2char16;
    MeanFilterFloatSquareFunc    float_square;
    MeanFilterMeanIPCorrIPABFunc meanip_corrip_ab;
    MeanFilterSize3x3fFunc       size3x3f;
    MeanFilterFunc               mean_filter;
} EnhanceDSPContext;

static EnhanceDSPContext dsp;

/* ============================================================================
 * AVX2 optimized functions (inline assembly)
 * ============================================================================ */
#if ENHANCE_VF_CPU_FLG_AVX2
static void MeanFilter_Char2Float_AVX2(unsigned char* srcData,float* dstData,int width,int height,int src_stride)
{
    unsigned char *pSrc;
    float *pDst;
    int loopCnt ;
    int freeColu = 0;   //剩余列数
    int offset = width / 8 * 8;   //偏移量
    const float val255f = 1.0 / 255.0;
    freeColu = width - offset;
    for(int i = 0;i < height;++i)
    {
        pSrc = srcData+(i * src_stride);
        pDst = dstData+(i * width);
        loopCnt = offset;
        __asm__ volatile(
            "vbroadcastss  %3,   %%ymm3              \n"      //用于除255.0转为乘上1.0/255.0
            "2:                                      \n"
            "vpmovzxbd (%0),                  %%ymm1 \n"      //将xmm0低8个char数据转为8个int
            "lea      0x8(%0),                %0     \n"      //dst偏移8
            "vcvtdq2ps %%ymm1,                %%ymm1 \n"      //将8个int转为8个float
            "vmulps   %%ymm1 ,%%ymm3,         %%ymm1 \n"
            "vmovups  %%ymm1,                 (%1)   \n" //将数据存放在dst
            "lea      0x20(%1),               %1     \n" //dst偏移32

            "sub     $0x8,%2                        \n"
            "jg      2b                             \n"
            "vzeroupper                             \n"
            :"+r"(pSrc),            //0
             "+r"(pDst),            //1
             "+r"(loopCnt)          //2
#if defined(__x86_64__)
            :"x"(val255f)           //3
#else
            :"m"(val255f)           //3
#endif
            : "memory","cc","xmm0","xmm1","xmm2","xmm3"
        );

        if(freeColu > 0)
        {
           for ( int j= offset; j < width; j++)
                dstData[i * width + j] = (float)srcData[i * src_stride + j]/255.0;
        }
    }
}

static void MeanFilter_Float2Char_AVX2(unsigned char* srcCharData,unsigned char* dstData,float* pMeanIP,float *pCorrIP,int width,int height,int src_stride,int dst_stride)
{
    int *pTmpInit = av_malloc(width * sizeof(int));
    unsigned char *pDst;
    unsigned char *pSrcChar;
    float *pMean;
    float *pCorr;
    int loopCnt ;
    int freeColu = 0;   //剩余列数
    int offset = width / 8 * 8;   //偏移量
    const float val1_255f = 1.0 / 255.0;
    const float val255f = 255.0;
    int *pTmp;
    freeColu = width - offset;
    for(int i = 0;i < height;++i)
    {
        pSrcChar = srcCharData +(i * src_stride);
        pMean = pMeanIP+(i * width);
        pCorr = pCorrIP+(i * width);
        loopCnt = offset;
        pTmp = pTmpInit;
        __asm__ volatile(
            "vbroadcastss  %5,        %%ymm3              \n"      //
            "vbroadcastss  %6,        %%ymm4              \n"      //
            "1:                                           \n"
            "vmovups      (%2),       %%ymm0              \n"  //加载8个pMean数据到ymm0 指令说明：由于width会出现不是32的倍数的情况所以会导致pMean不是32对齐不能使用vmovaps指令。pCorr同
            "lea      0x20(%2),       %2                  \n"  //地址偏移

            "vpmovzxbd (%0),           %%ymm1             \n"  //加载8个char数据转为8个int
            "lea       0x8(%0),        %0                 \n"  //dst偏移8
            "vcvtdq2ps %%ymm1,         %%ymm1             \n"  //将8个int转为8个float

            "vmulps    %%ymm0,         %%ymm1,     %%ymm1 \n"  //pMean * (float)srcChar

            "vmovups (%3),             %%ymm0             \n"  //加载8个pCorr数据到ymm0
            "lea      0x20(%3),        %3                 \n"  //地址偏移
            
            "vfmadd132ps %%ymm3,       %%ymm0,     %%ymm1 \n"  // (pMean * (float)srcChar) / 255.0+pCorr
            
            "vmulps   %%ymm1 ,         %%ymm4,     %%ymm1 \n"  //((pMean * (float)srcChar) / 255.0+pCorr) * 255.0
            "vcvtps2dq %%ymm1,         %%ymm1             \n"  //float 转 init

            "vmovups  %%ymm1,          (%1)               \n"  //存储到内存
            "lea      0x20(%1),        %1                 \n"  //地址偏移

            "sub     $0x8,%4                  \n"
            "jg      1b                       \n"
            "vzeroupper                       \n"
            :"+r"(pSrcChar),         //0
             "+r"(pTmp),             //1
             "+r"(pMean),            //2
             "+r"(pCorr),            //3
             "+r"(loopCnt)           //4
#if defined(__x86_64__)
            :"x"(val1_255f),         //5
             "x"(val255f)            //6
#else
            :"m"(val1_255f),         //5
             "m"(val255f)            //6
#endif
            : "memory","cc","xmm0","xmm1","xmm2","xmm3","xmm4"
        );

        for ( int j=0; j < offset; j+=8)
        {
            dstData[i * dst_stride + j] = (unsigned char)(av_clip(pTmpInit[j], 0, 255));
            dstData[i * dst_stride + j+1] = (unsigned char)(av_clip(pTmpInit[j+1], 0, 255));
            dstData[i * dst_stride + j+2] = (unsigned char)(av_clip(pTmpInit[j+2], 0, 255));
            dstData[i * dst_stride + j+3] = (unsigned char)(av_clip(pTmpInit[j+3], 0, 255));
            dstData[i * dst_stride + j+4] = (unsigned char)(av_clip(pTmpInit[j+4], 0, 255));
            dstData[i * dst_stride + j+5] = (unsigned char)(av_clip(pTmpInit[j+5], 0, 255));
            dstData[i * dst_stride + j+6] = (unsigned char)(av_clip(pTmpInit[j+6], 0, 255));
            dstData[i * dst_stride + j+7] = (unsigned char)(av_clip(pTmpInit[j+7], 0, 255));
        }

        if(freeColu > 0)
        {
           for ( int j=offset; j < width; j++)
            {
                dstData[i * dst_stride + j] = (unsigned char)(av_clip((pMeanIP[i * width + j] * (float)srcCharData[i * src_stride + j]*val1_255f + pCorrIP[i * width + j])*255.0f, 0, 255));
            }
        }
    }

    av_free(pTmpInit);
}

static void MeanFilter_Char2Float_AVX2_16(uint16_t* srcData,float* dstData,int width,int height,int src_stride,float maxData)
{
    uint16_t *pSrc;
    float *pDst;
    int loopCnt ;
    int freeColu = 0;   //剩余列数
    int offset = width / 8 * 8;   //偏移量
    const float val255f = 1.0 / 1023.0;
    freeColu = width - offset;
    for(int i = 0;i < height;++i)
    {
        pSrc = srcData+(i * src_stride);
        pDst = dstData+(i * width);
        loopCnt = offset;
        __asm__ volatile(
            "vbroadcastss  %3,   %%ymm3              \n"      //用于除255.0转为乘上1.0/255.0
            "2:                                      \n"
            "vpmovzxwd (%0),                  %%ymm1 \n"      //将xmm0低8个short数据转为8个int
            "lea      0x10(%0),                %0     \n"      //dst偏移16
            "vcvtdq2ps %%ymm1,                %%ymm1 \n"      //将8个int转为8个float
            "vmulps   %%ymm1 ,%%ymm3,         %%ymm1 \n"
            "vmovups  %%ymm1,                 (%1)   \n" //将数据存放在dst
            "lea      0x20(%1),               %1     \n" //dst偏移32

            "sub     $0x8,%2                        \n"
            "jg      2b                             \n"
            "vzeroupper                             \n"
            :"+r"(pSrc),            //0
             "+r"(pDst),            //1
             "+r"(loopCnt)          //2
#if defined(__x86_64__)
            :"x"(val255f)           //3
#else
            :"m"(val255f)           //3
#endif
            : "memory","cc","xmm0","xmm1","xmm2","xmm3"
        );

        if(freeColu > 0)
        {
           for ( int j= offset; j < width; j++)
                dstData[i * width + j] = (float)srcData[i * src_stride + j]/maxData;
        }
    }
}

static void MeanFilter_Float2Char_AVX2_16(uint16_t* srcCharData,uint16_t* dstData,float* pMeanIP,float *pCorrIP,int width,int height,int src_stride,int dst_stride,float maxData)
{
    uint16_t *pDst;
    uint16_t *pSrcChar;
    float *pMean;
    float *pCorr;
    int loopCnt ;
    int freeColu = 0;   //剩余列数
    int offset = width / 8 * 8;   //偏移量
    const float val1_255f = 1.0 / maxData;
    const float val255f = maxData;
    uint16_t *pTmp;
    freeColu = width - offset;
    for(int i = 0;i < height;++i)
    {
        pSrcChar = srcCharData +(i * src_stride);
        pMean = pMeanIP+(i * width);
        pCorr = pCorrIP+(i * width);
        loopCnt = offset;
        pTmp = dstData + (i * dst_stride);
        __asm__ volatile(
            "vbroadcastss  %5,        %%ymm3              \n"      //
            "vbroadcastss  %6,        %%ymm4              \n"      //
            "vcvtps2dq     %%ymm4,    %%ymm5         \n"  //float 转 init
            "1:                                           \n"
            "vmovups      (%2),       %%ymm0              \n"  //加载8个pMean数据到ymm0 指令说明：由于width会出现不是32的倍数的情况所以会导致pMean不是32对齐不能使用vmovaps指令。pCorr同
            "lea      0x20(%2),       %2                  \n"  //地址偏移

            "vpmovzxwd (%0),           %%ymm1             \n"  //加载8个short数据转为8个int
            "lea       0x10(%0),        %0                 \n"  //dst偏移8
            "vcvtdq2ps %%ymm1,         %%ymm1             \n"  //将8个int转为8个float

            "vmulps    %%ymm0,         %%ymm1,     %%ymm1 \n"  //pMean * (float)srcChar

            "vmovups (%3),             %%ymm0             \n"  //加载8个pCorr数据到ymm0
            "lea      0x20(%3),        %3                 \n"  //地址偏移
            
            "vfmadd132ps %%ymm3,       %%ymm0,     %%ymm1 \n"  // (pMean * (float)srcChar) / 255.0+pCorr
            
            "vmulps   %%ymm1 ,         %%ymm4,     %%ymm1 \n"  //((pMean * (float)srcChar) / 255.0+pCorr) * 255.0
            "vcvtps2dq %%ymm1,         %%ymm1             \n"  //float 转 init

            "vpxor                   %%ymm0,%%ymm0,%%ymm0 \n" //ymm0 清0
            "vpmaxsd                 %%ymm1,%%ymm0,%%ymm1 \n" //和0做比较 比0小的，都为0
            "vpminsd                 %%ymm1,%%ymm5,%%ymm1 \n" //和maxData比较比maxData大的都设置为maxData
            "vextractf128         $0x1,%%ymm1,     %%xmm0  \n"//将ymm1的128-255拷贝到xmm0的0-127
            "packssdw                %%xmm0,       %%xmm1  \n" //将xmm0的4个32bit和xmm1的4个32bit转8个16bit到xmm132->16
            "movdqu   %%xmm1,          (%1)               \n"  //存储到内存
            "lea      0x10(%1),        %1                 \n"  //地址偏移

            "sub     $0x8,%4                  \n"
            "jg      1b                       \n"
            "vzeroupper                       \n"
            :"+r"(pSrcChar),         //0
             "+r"(pTmp),          //1
             "+r"(pMean),            //2
             "+r"(pCorr),            //3
             "+r"(loopCnt)           //4
#if defined(__x86_64__)
            :"x"(val1_255f),         //5
             "x"(val255f)            //6
#else
            :"m"(val1_255f),         //5
             "m"(val255f)            //6
#endif
            : "memory","cc","xmm0","xmm1","xmm2","xmm3","xmm4","xmm5"
        );

        if(freeColu > 0)
        {
           for ( int j=offset; j < width; j++)
            {
                dstData[i * dst_stride + j] = (uint16_t)(av_clipf((pMeanIP[i * width + j] * (float)srcCharData[i * src_stride + j]* val1_255f + pCorrIP[i * width + j])*maxData, 0, maxData));
            }
        }
    }

}
#endif


static void MeanFilter_FloatSquare_AVX2(float* pData,int width,int height)
{
    unsigned int i;
    int cnt = width * height / 8 * 8;
    float *pSrc = pData;
    i = cnt;
    __asm__ volatile(
         "1:                               \n"
         "vmovups (%0),%%ymm0              \n"  //加载8个数据到ymm0
         "vmulps  %%ymm0, %%ymm0, %%ymm1   \n"  //ymm1 = ymm0*ymm0
         "vmovups %%ymm1,(%0)              \n"  //存储到内存

         "lea     0x20(%0),%0              \n"  //地址偏移

         "sub     $0x8,%1                  \n"
         "jg      1b                       \n"
         "vzeroupper                       \n"
         :"+r"(pSrc),   //0
          "+rm"(cnt)    //1
         :
         :"memory","cc","xmm0","xmm1","xmm2","xmm3"
    );

    for(; i < width * height; i++) {
        pData[i] = pData[i] * pData[i];
    }
}

static void MeanFilter_meanIPcorrIP_AB_AVX2(float* pMeanIP,float *pCorrIP,float * pA,float * pB,int width,int height, float delta)
{
    unsigned int i;
    int cnt = width * height / 8 * 8;
    float * pTempA = pA;
    float * pTempB = pB;
    float * pTempMeanIp = pMeanIP;
    float * pTempCorrIp = pCorrIP;
    i = cnt;
    __asm__ volatile(
         "vbroadcastss  %5,   %%ymm4         \n"   //加载delta到ymm5
         "3:                                 \n"
         "vmovups (%0),%%ymm0                \n"  //加载8个meanIp数据到ymm0
         "lea     0x20(%0),%0                \n"  //地址偏移32
         "vmovups (%1),%%ymm1                \n"  //加载8个corrIp数据到ymm1
         "lea     0x20(%1),%1                \n"  //地址偏移32
         "vmovaps %%ymm0,%%ymm2              \n"  //复制
         "vfnmadd132ps %%ymm0,%%ymm1,%%ymm2  \n"  // ymm2 = ymm1 - ymm2 * ymm0 ==》corrIP - meanIP * meanIP
         "vaddps   %%ymm2, %%ymm4,  %%ymm3   \n"  //
         "vdivps   %%ymm3, %%ymm2,  %%ymm2   \n"
         "vmovups  %%ymm2,           (%2)    \n" //将数据存放在a
         "lea      0x20(%2),          %2     \n" //地址偏移加32
         "vfnmadd132ps %%ymm0,%%ymm0,%%ymm2  \n"  // ymm2 = ymm0 - ymm2 * ymm0

         "vmovups  %%ymm2,           (%3)    \n" //将数据存放在b
         "lea      0x20(%3),          %3     \n" //地址偏移加32
         "sub     $0x8,%4                    \n"
         "jg      3b                         \n"
         "vzeroupper                         \n"
         :"+r"(pTempMeanIp),   //0
          "+r"(pTempCorrIp),   //1
          "+r"(pTempA),        //2
          "+r"(pTempB),        //3
          "+rm"(cnt)           //4
#if defined(__x86_64__)
         :"x"(delta)           //5
#else
         :"m"(delta)     //5
#endif
         :"memory","cc","xmm0","xmm1","xmm2","xmm3"
    );

     for(; i < width * height; i++)
     {
            pA[i] = (pCorrIP[i] - pMeanIP[i] * pMeanIP[i]) / (pCorrIP[i] - pMeanIP[i] * pMeanIP[i]  + delta);
            pB[i] = pMeanIP[i] - pA[i] * pMeanIP[i];
     }
}
/* 3x3 mean filter for float data */
static void MeanFilter_Size3x3f_AVX2(float* srcData,float* dstData,int width,int height,int src_stride)
{
    float *pSrc;
    float *pDst;
    int loopCnt ;
    const float win_size = 1.0 / 9.0;
    int32_t __attribute__((aligned(32))) indxTbl[8] = {1,2,3,4,5,6,7,0};
    int freeColu = 0;   //剩余列数
    int offset = (width -6) / 6 * 6;   //偏移量
    float sum = 0.0f;
    float number = 3.0f * 3.0f;
    freeColu = width + 1 - offset;
    for(int i = 1;i < height + 1;++i)
    {
        pSrc = srcData+((i-1) * src_stride);
        pDst = dstData+((i-1) * width);
        loopCnt = offset;
        __asm__ volatile(
            "vmovdqa      (%4),  %%ymm4         \n"   //加载索引表
            "vbroadcastss  %5,   %%ymm5         \n"   //用于除9.0转为乘上1.0/9.0
            "1:                               \n"
            "vmovups  (%0),            %%ymm0 \n" //加载第i-1行8个float数据
            "vmovups  0x00(%0,%3,4),          %%ymm1 \n" //加载第i行8个float数据
            "vaddps   %%ymm0, %%ymm1,         %%ymm1 \n" //计算第i-1和i行数据和，放在ymm1
            "vmovups  0x00(%0,%3,8),          %%ymm0 \n" //加载第i+1行8个float数据
            "vaddps   %%ymm0, %%ymm1,         %%ymm1 \n" //计算ymm1和第i+1的和（即第i-1、i、i+1的和）放在ymm1
            "vmulps   %%ymm1 ,%%ymm5,         %%ymm1 \n" //i-1、i、i+1的和 除上9
            "lea      0x18(%0),                %0    \n" //src 地址偏移加6*4
      
            "vpermps  %%ymm1, %%ymm4,        %%ymm2 \n"  //取出ymm1的第1，2，3，4，5，6，7的的元素数据放在ymm2
            "vpermps  %%ymm2, %%ymm4,        %%ymm3 \n"  //取出ymm1的第2，3，4，5，6，7 的的元素数据放在ymm3
            "vaddps   %%ymm1, %%ymm2,        %%ymm2 \n"  //求和
            "vaddps   %%ymm3, %%ymm2,        %%ymm3 \n"  //求和

            "vmovups  %%ymm3,                (%1)   \n" //将数据存放在dst
            "lea      0x18(%1),              %1     \n" //dst偏移

            "sub     $0x6,%2                        \n"
            "jg      1b                             \n"
            "vzeroupper                             \n"
            :"+r"(pSrc),            //0
             "+r"(pDst),            //1
             "+r"(loopCnt)          //2
            :"r"((long)src_stride), //3
             "r"(indxTbl),          //4
#if defined(__x86_64__)
             "x"(win_size)  // 5
#else
             "m"(win_size)  // 5
#endif
            : "memory","cc","xmm0","xmm1","xmm2","xmm3","xmm4","xmm5"
        );
        
        if(freeColu > 2)
        {
            for (int j = offset + 1; j < width + 1;++j)
            {
                for (int r = i - 1; r <= i + 1; ++r){
                    for (int c = j - 1; c <= j + 1;++c){
                        sum = srcData[ r * src_stride + c ] + sum;
                    }
                }
                dstData[(i-1) * width +(j-1)] = sum / number;
                sum = 0;
            }
        }
    }
}

static void MeanFilter_AVX2(float* srcData, float* dstData, int width, int height, int radius, int mask){

    int hh = radius;
    int hw = radius;
    int i, j;
  
    /* Border padding for mean filter calculation */
    float* dst = (float*)av_malloc(sizeof(float) * (width + radius * 2) * (height + radius * 2));
    float* dstTemp = dst;
    float* srcTemp = srcData; /* Expand border before mean filter */
    int srcstep = width;
    int dststep = width + 2 * radius;
    int left,right, top, bottom;
    left = right = top = bottom = radius;

    int* tab = (int*)malloc(sizeof(int) * radius * 2);
    for( i = 0; i < left; i++ )
    {
        j = borderInterpolate(i - left, width);
        tab[i] = j ;
    }

    for( i = 0; i < right; i++ )
    {
        j = borderInterpolate(width + i, width);
        tab[i+left] = j;
    }

    float* dstInner = dst + dststep * top + left;

    for( i = 0; i < height; i++, dstInner += dststep, srcTemp += srcstep )
    {
        if( dstInner != srcData )
            memcpy(dstInner, srcTemp, width * sizeof(float));
        for( j = 0; j < left; j++ )
            dstInner[j - left] = srcTemp[tab[j]];
        for( j = 0; j < right; j++ )
            dstInner[j + width] = srcTemp[tab[j + left]];
    }

    if ((mask & 0x01) == 0x01) //第一个slice
    {
        dstTemp = dst + dststep * top;
        for( i = 0; i < top; i++ ) //上边界
        {
            j = borderInterpolate(i - top, height);
            memcpy(dstTemp + (i - top) * dststep, dstTemp + j * dststep, dststep * sizeof(float));
        }
    }
    else
    {
        dstTemp = dst + dststep * top +left;
        srcTemp = srcData;
        for( i = 0; i < top; i++ ) //上边界
        {
            memcpy(dstTemp + (i - top) * dststep, srcTemp + (i - top) * srcstep, srcstep * sizeof(float));
            dstInner = dstTemp + (i - top) * dststep;
            for( j = 0; j < left; j++ )
                dstInner[j - left] = srcTemp[(i - top) * srcstep + tab[j]];
            for( j = 0; j < right; j++ )
                dstInner[j + width] = srcTemp[(i - top) * srcstep + tab[j + left]];
        }
    }
    
    if((mask & 0x2) == 0x2) //最后一个slice
    {
        dstTemp = dst + dststep * top;
        for( i = 0; i < bottom; i++ )//下边界
        {
            j = borderInterpolate(i + height, height);
            memcpy(dstTemp + (i + height) * dststep, dstTemp + j * dststep, dststep * sizeof(float));
        }
    }
    else
    {
        dstTemp = dst + dststep * top + left;
        srcTemp = srcData;
        for( i = 0; i < bottom; i++ )//下边界
        {
            memcpy(dstTemp + (i + height) * dststep, srcTemp + (i + height) * srcstep, srcstep * sizeof(float));
            dstInner = dstTemp + (i + height) * dststep;
            for( j = 0; j < left; j++ )
                dstInner[j - left] = srcTemp[(i + height) * srcstep+ tab[j]];
            for( j = 0; j < right; j++ )
                dstInner[j + width] = srcTemp[(i + height) * srcstep + tab[j + left]];
        }
    }

    dsp.size3x3f(dst, dstData, width, height, dststep);

    free(tab);
    av_free(dst);
}

static void guided_filter_avx2_slice(AVFilterContext *ctx, FilterParam *pThreadData)
{
    /* EnhanceContext *s = ctx->priv; */
    unsigned char* srcData;
    unsigned char* dstData;
    int srcStride;
    int dstStride;
    int width;
    int height;
    int radius;
    float delta;
    int add_h;
    float *data;
    float *meanIP;
    float *corrIP;
    float *varIP;
    float *a;
    float *b;
    float *pTmp;
    int offset;
    
    if (pThreadData->src_data == NULL || pThreadData->dst_data == NULL)
        return;

    radius = pThreadData->radius;
    if (radius == 0) {
        av_image_copy_plane(pThreadData->dst_data, pThreadData->dst_linesize,
                            pThreadData->src_data, pThreadData->src_linesize,
                            pThreadData->width, pThreadData->height);
        return;
    }

    srcData   = pThreadData->src_data;
    srcStride = pThreadData->src_linesize;
    dstStride = pThreadData->dst_linesize;
    width     = pThreadData->width;
    height    = pThreadData->height;
    delta     = pThreadData->delta;
    dstData   = pThreadData->dst_data;

    /* For middle and last slice, offset to read radius*2 more lines from previous slice */
    if ((pThreadData->mask & 0x01) != 0x01)
        srcData -= pThreadData->src_linesize * radius * 2;

    if (pThreadData->mask == 0x00) {
        /* Middle slice: read 2*radius more lines from adjacent slices */
        add_h = radius * 4;
    } else if (pThreadData->mask == 0x01 || pThreadData->mask == 0x02) {
        /* First or last slice: read radius*2 more lines */
        add_h = radius * 2;
    } else {
        add_h = 0;
    }
    
    data   = av_malloc(sizeof(float) * width * (height + add_h));
    meanIP = av_malloc(sizeof(float) * width * (height + add_h));
    corrIP = av_malloc(sizeof(float) * width * (height + add_h));
    varIP  = av_malloc(sizeof(float) * width * (height + add_h));
    a      = av_malloc(sizeof(float) * width * (height + add_h));
    b      = av_malloc(sizeof(float) * width * (height + add_h));

    if (!data || !meanIP || !corrIP || !varIP || !a || !b) {
        av_free(data);
        av_free(meanIP);
        av_free(corrIP);
        av_free(varIP);
        av_free(a);
        av_free(b);
        return;
    }

    pTmp = NULL;
    offset = 0;

    /* Offset result for middle and last slice */
    if ((pThreadData->mask & 0x01) != 0x01)
        offset = width * radius;
    
    /* Convert to float and extend by 2*radius */
    dsp.char2float(srcData, data, width, height + add_h, srcStride);

    pTmp = data + offset;
    dsp.mean_filter(pTmp, meanIP, width, height + add_h / 2, radius, pThreadData->mask);
    
    /* Square and extend by 2*radius */
    dsp.float_square(data, width, height + add_h);
    pTmp = data + offset;
    dsp.mean_filter(pTmp, corrIP, width, height + add_h / 2, radius, pThreadData->mask);
    
    dsp.meanip_corrip_ab(meanIP, corrIP, a, b, width, height + add_h / 2, delta);
    
    pTmp = a + offset;
    dsp.mean_filter(pTmp, meanIP, width, height, radius, pThreadData->mask);
    pTmp = b + offset;
    dsp.mean_filter(pTmp, corrIP, width, height, radius, pThreadData->mask);
    dsp.float2char(pThreadData->src_data, dstData, meanIP, corrIP,
                   width, height, srcStride, dstStride);
    
    av_free(data);
    av_free(meanIP);
    av_free(corrIP);
    av_free(varIP);
    av_free(a);
    av_free(b);
    
    if (g_guided_filter_sync.init_flg) {
        pthread_mutex_lock(&g_guided_filter_sync.mutex);
        g_guided_filter_sync.result_cnt += 1;
        if (g_guided_filter_sync.result_cnt == pThreadData->thread_num)
            pthread_cond_signal(&g_guided_filter_sync.cond);
        pthread_mutex_unlock(&g_guided_filter_sync.mutex);
    }
}

static void guided_filter_avx2_slice16(AVFilterContext *ctx, FilterParam *pThreadData)
{
    uint16_t *srcData;
    uint16_t *dstData;
    EnhanceContext *s = ctx->priv; 
    float maxData =  (float)((1 << s->bitdepth) - 1);

    int srcStride;
    int dstStride;
    int width;
    int height;
    int radius;
    float delta;
    int add_h;
    float *data;
    float *meanIP;
    float *corrIP;
    float *varIP;
    float *a;
    float *b;
    float *pTmp;
    int offset;
    
    if (pThreadData->src_data == NULL || pThreadData->dst_data == NULL)
        return;

    radius = pThreadData->radius;
    if (radius == 0) {
        av_image_copy_plane(pThreadData->dst_data, pThreadData->dst_linesize,
                            pThreadData->src_data, pThreadData->src_linesize,
                            pThreadData->width * s->bps, pThreadData->height);
        return;
    }

    srcData   = (uint16_t *)pThreadData->src_data;
    srcStride = pThreadData->src_linesize / s->bps;
    dstStride = pThreadData->dst_linesize / s->bps;
    width     = pThreadData->width;
    height    = pThreadData->height;
    delta     = pThreadData->delta;
    dstData   = (uint16_t *)pThreadData->dst_data;

    /* For middle and last slice, offset to read radius*2 more lines from previous slice */
    if ((pThreadData->mask & 0x01) != 0x01)
        srcData -= srcStride * radius * 2;

    if (pThreadData->mask == 0x00) {
        /* Middle slice: read 2*radius more lines from adjacent slices */
        add_h = radius * 4;
    } else if (pThreadData->mask == 0x01 || pThreadData->mask == 0x02) {
        /* First or last slice: read radius*2 more lines */
        add_h = radius * 2;
    } else {
        add_h = 0;
    }
    
    data   = av_malloc(sizeof(float) * width * (height + add_h));
    meanIP = av_malloc(sizeof(float) * width * (height + add_h));
    corrIP = av_malloc(sizeof(float) * width * (height + add_h));
    varIP  = av_malloc(sizeof(float) * width * (height + add_h));
    a      = av_malloc(sizeof(float) * width * (height + add_h));
    b      = av_malloc(sizeof(float) * width * (height + add_h));

    if (!data || !meanIP || !corrIP || !varIP || !a || !b) {
        av_free(data);
        av_free(meanIP);
        av_free(corrIP);
        av_free(varIP);
        av_free(a);
        av_free(b);
        return;
    }

    pTmp = NULL;
    offset = 0;

    /* Offset result for middle and last slice */
    if ((pThreadData->mask & 0x01) != 0x01)
        offset = width * radius;

    /* Convert to float and extend by 2*radius */
    dsp.char2float16(srcData, data, width, height + add_h, srcStride, maxData);

    pTmp = data + offset;
    dsp.mean_filter(pTmp, meanIP, width, height + add_h / 2, radius, pThreadData->mask);
    
    /* Square and extend by 2*radius */
    dsp.float_square(data, width, height + add_h);
    pTmp = data + offset;
    dsp.mean_filter(pTmp, corrIP, width, height + add_h / 2, radius, pThreadData->mask);
    
    dsp.meanip_corrip_ab(meanIP, corrIP, a, b, width, height + add_h / 2, delta);
    
    pTmp = a + offset;
    dsp.mean_filter(pTmp, meanIP, width, height, radius, pThreadData->mask);
    pTmp = b + offset;
    dsp.mean_filter(pTmp, corrIP, width, height, radius, pThreadData->mask);
    srcData = (uint16_t *)pThreadData->src_data;

    dsp.float2char16(srcData, dstData, meanIP, corrIP,
                     width, height, srcStride, dstStride, maxData);
    
    av_free(data);
    av_free(meanIP);
    av_free(corrIP);
    av_free(varIP);
    av_free(a);
    av_free(b);
    
    if (g_guided_filter_sync.init_flg) {
        pthread_mutex_lock(&g_guided_filter_sync.mutex);
        g_guided_filter_sync.result_cnt += 1;
        if (g_guided_filter_sync.result_cnt == pThreadData->thread_num)
            pthread_cond_signal(&g_guided_filter_sync.cond);
        pthread_mutex_unlock(&g_guided_filter_sync.mutex);
    }
}

/* ============================================================================
 * C versions (fallback)
 * ============================================================================ */
static void MeanFilter(float* srcData, float* dstData, int width, int height, int radius,int mask);
static void MeanFilter_Char2Float_C(unsigned char *srcData, float *dstData,
                                    int width, int height, int src_stride)
{
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++)
            dstData[i * width + j] = (float)srcData[i * src_stride + j] / 255.0;
    }
}

static void MeanFilter_Float2Char_C(unsigned char *srcCharData,
                                     unsigned char *dstData,
                                     float *pMeanIP, float *pCorrIP,
                                     int width, int height,
                                     int src_stride, int dst_stride)
{
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int index = i * width + j;
            float val = pMeanIP[index] * (float)srcCharData[i * src_stride + j] / 255.0f + pCorrIP[index];
            dstData[i * dst_stride + j] = (unsigned char)av_clipf(val * 255.0f, 0.0, 255.0);
        }
    }
}

static void MeanFilter_Char2Float_16_C(uint16_t *srcData, float *dstData,
                                        int width, int height,
                                        int src_stride, float maxData)
{
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++)
            dstData[i * width + j] = (float)srcData[i * src_stride + j] / maxData;
    }
}

static void MeanFilter_Float2Char_16_C(uint16_t *srcCharData,
                                       uint16_t *dstData,
                                       float *pMeanIP, float *pCorrIP,
                                       int width, int height,
                                       int src_stride, int dst_stride,
                                       float maxData)
{
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int index = i * width + j;
            float val = pMeanIP[index] * (float)srcCharData[i * src_stride + j] / maxData + pCorrIP[index];
            dstData[i * dst_stride + j] = (uint16_t)av_clipf(val * maxData, 0.0, maxData);
        }
    }
}

static void MeanFilter_FloatSquare_C(float *pData, int width, int height)
{
    for (int i = 0; i < width * height; i++)
        pData[i] = pData[i] * pData[i];
}

static void MeanFilter_meanIPcorrIP_AB_C(float *pMeanIP, float *pCorrIP,
                                         float *pA, float *pB,
                                         int width, int height, float delta)
{
    for (int i = 0; i < width * height; i++) {
        pA[i] = (pCorrIP[i] - pMeanIP[i] * pMeanIP[i]) / (pCorrIP[i] - pMeanIP[i] * pMeanIP[i] + delta);
        pB[i] = pMeanIP[i] - pA[i] * pMeanIP[i];
    }
}

static void MeanFilter_Size3x3f_C(float *srcData, float *dstData,
                                   int width, int height, int src_stride)
{
    float sum;
    float number = 3.0f * 3.0f;
    for (int i = 1; i < height + 1; i++) {
        for (int j = 1; j < width + 1; j++) {
            sum = 0.0f;
            for (int r = i - 1; r <= i + 1; r++) {
                for (int c = j - 1; c <= j + 1; c++) {
                    sum += srcData[r * src_stride + c];
                }
            }
            dstData[(i - 1) * width + (j - 1)] = sum / number;
        }
    }
}

/* ============================================================================
 * Wrapper functions for external assembly functions
 * ============================================================================ */
#if HAVE_AVX2_EXTERNAL
static void MeanFilter_Char2Float_16_Asm_Wrapper(uint16_t *srcData, float *dstData,
                                                  int width, int height,
                                                  int src_stride, float maxData)
{
    float *max_data_ptr = (maxData == 1023.0f) ? max_pixel_10bit[0] : 
                          (maxData == 4095.0f) ? max_pixel_12bit[0] : NULL;
    if (max_data_ptr)
        ff_MeanFilter_Piex2Float_16_avx2(srcData, dstData, width, height, src_stride, max_data_ptr);
    else
        MeanFilter_Char2Float_AVX2_16(srcData, dstData, width, height, src_stride, maxData);
}

static void MeanFilter_Float2Char_16_Asm_Wrapper(uint16_t *srcCharData,
                                                  uint16_t *dstData,
                                                  float *pMeanIP, float *pCorrIP,
                                                  int width, int height,
                                                  int src_stride, int dst_stride,
                                                  float maxData)
{
    float *max_data_ptr = (maxData == 1023.0f) ? max_pixel_10bit[1] : 
                          (maxData == 4095.0f) ? max_pixel_12bit[1] : NULL;
    if (max_data_ptr)
        ff_MeanFilter_Float2Piex_16_avx2(srcCharData, dstData, pMeanIP, pCorrIP,
                                         width, height, src_stride, dst_stride, max_data_ptr);
    else
        MeanFilter_Float2Char_AVX2_16(srcCharData, dstData, pMeanIP, pCorrIP,
                                      width, height, src_stride, dst_stride, maxData);
}
#endif

/* Initialize DSP function pointers based on CPU capabilities */
static av_cold void enhance_dsp_init(void)
{
    int cpu_flags = av_get_cpu_flags();
    
    /* Default to C versions */
    dsp.char2float      = MeanFilter_Char2Float_C;
    dsp.float2char      = MeanFilter_Float2Char_C;
    dsp.char2float16    = MeanFilter_Char2Float_16_C;
    dsp.float2char16    = MeanFilter_Float2Char_16_C;
    dsp.float_square    = MeanFilter_FloatSquare_C;
    dsp.meanip_corrip_ab = MeanFilter_meanIPcorrIP_AB_C;
    dsp.size3x3f        = MeanFilter_Size3x3f_C;
    dsp.mean_filter     = MeanFilter;
    
#if ENHANCE_VF_CPU_FLG_AVX2
    if (EXTERNAL_AVX2(cpu_flags)) {
        /* Use inline assembly versions */
        dsp.char2float      = MeanFilter_Char2Float_AVX2;
        dsp.float2char      = MeanFilter_Float2Char_AVX2;
        dsp.float_square    = MeanFilter_FloatSquare_AVX2;
        dsp.meanip_corrip_ab = MeanFilter_meanIPcorrIP_AB_AVX2;
        dsp.size3x3f        = MeanFilter_Size3x3f_AVX2;
        dsp.mean_filter     = MeanFilter_AVX2;
        av_log(NULL, AV_LOG_INFO,"Using AVX2 for image enhancement\n");
        
#if HAVE_AVX2_EXTERNAL
        /* Use external assembly for 16-bit if available, otherwise use inline assembly */
        if (HAVE_X86ASM) {
            dsp.char2float16 = MeanFilter_Char2Float_16_Asm_Wrapper;
            dsp.float2char16 = MeanFilter_Float2Char_16_Asm_Wrapper;
            av_log(NULL, AV_LOG_INFO,"Using external assembly for 16-bit image enhancement\n");
        } else {
            dsp.char2float16 = MeanFilter_Char2Float_AVX2_16;
            dsp.float2char16 = MeanFilter_Float2Char_AVX2_16;
            av_log(NULL, AV_LOG_INFO,"Using inline assembly for 16-bit image enhancement\n");
        }
#else
        dsp.char2float16 = MeanFilter_Char2Float_AVX2_16;
        dsp.float2char16 = MeanFilter_Float2Char_AVX2_16;
        av_log(NULL, AV_LOG_INFO,"Using inline assembly for 16-bit image enhancement\n");
#endif
    } else {
        av_log(NULL, AV_LOG_INFO,"Using C for image enhancement\n");
    }
#endif
}

static void MeanFilter(float* srcData, float* dstData, int width, int height, int radius,int mask){

    int hh = radius;
    int hw = radius;
    int i, j;
    
    /* Border padding for mean filter calculation */
    float* dst = (float*)av_malloc(sizeof(float) * (width + radius * 2) * (height + radius * 2));
    float* dstTemp = dst;
    float* srcTemp = srcData; /* Expand border before mean filter */
    int srcstep = width;
    int dststep = width + 2 * radius;
    int left,right, top, bottom;
    left = right = top = bottom = radius;

    int* tab = (int*)malloc(sizeof(int) * radius * 2);
    for( i = 0; i < left; i++ )
    {
        j = borderInterpolate(i - left, width);
        tab[i] = j ;
    }

    for( i = 0; i < right; i++ )
    {
        j = borderInterpolate(width + i, width);
        tab[i+left] = j;
    }

    float* dstInner = dst + dststep * top + left;

    for( i = 0; i < height; i++, dstInner += dststep, srcTemp += srcstep )
    {
        if( dstInner != srcData )
            memcpy(dstInner, srcTemp, width * sizeof(float));
        for( j = 0; j < left; j++ )
            dstInner[j - left] = srcTemp[tab[j]];
        for( j = 0; j < right; j++ )
            dstInner[j + width] = srcTemp[tab[j + left]];
    }

    if ((mask & 0x01) == 0x01) //第一块
    {
        dstTemp = dst + dststep * top;
        for( i = 0; i < top; i++ ) //上边界
        {
            j = borderInterpolate(i - top, height);
            memcpy(dstTemp + (i - top) * dststep, dstTemp + j * dststep, dststep * sizeof(float));
        }
    }
    else
    {
        dstTemp = dst + dststep * top +left;
        srcTemp = srcData;
        for( i = 0; i < top; i++ ) //上边界
        {
            memcpy(dstTemp + (i - top) * dststep, srcTemp + (i - top) * srcstep, srcstep * sizeof(float));
            dstInner = dstTemp + (i - top)* dststep;
            for( j = 0; j < left; j++ )
                dstInner[j - left] = srcTemp[(i - top)* srcstep + tab[j]];
            for( j = 0; j < right; j++ )
                dstInner[j + width] = srcTemp[(i - top)* srcstep + tab[j + left]];
        }
    }
    
    if((mask & 0x2) == 0x2) //最后一块
    {
        dstTemp = dst + dststep * top;
        for( i = 0; i < bottom; i++ )//下边界
        {
            j = borderInterpolate(i + height, height);
            memcpy(dstTemp + (i + height) * dststep, dstTemp + j * dststep, dststep * sizeof(float));
        }
    }
    else
    {
        dstTemp = dst + dststep * top + left;
        srcTemp = srcData;
        for( i = 0; i < bottom; i++ )//下边界
        {
            memcpy(dstTemp + (i + height) * dststep, srcTemp + (i + height) * srcstep, srcstep * sizeof(float));
            dstInner = dstTemp + (i + height)* dststep;
            for( j = 0; j < left; j++ )
                dstInner[j - left] = srcTemp[(i + height)* srcstep + tab[j]];
            for( j = 0; j < right; j++ )
                dstInner[j + width] = srcTemp[(i + height)* srcstep + tab[j + left]];
        }
    }
    /* Mean filter */
    {
        float sum = 0;
        float mean = 0;
        float number = (2 * radius + 1) * (2 * radius + 1);
        int i, j, r, c;
        for (i = hh; i < height + hh; ++i){
            for (j = hw; j < width + hw;++j){

                for (r = i - hh; r <= i + hh; ++r){
                    for (c = j - hw; c <= j + hw;++c){
                        sum = dst[ r * dststep + c ] + sum;
                    }
                }
                mean = sum / number;
                dstData[(i-hh) * width +(j-hw)] = mean;
                sum = 0;
                mean = 0;
            }
        }
    }
    free(tab);
    av_free(dst);
}

/*
 * Process y/uv planes separately in vertical slices
 */
static void guided_filter_slice(AVFilterContext *ctx, FilterParam *pThreadData)
{
    unsigned char *srcData;
    unsigned char *dstData;
    int srcStride;
    int dstStride;
    int width;
    int height;
    int radius;
    float delta;
    int add_h;
    float *data;
    float *meanIP;
    float *corrIP;
    float *varIP;
    float *a;
    float *b;
    float *pTmp;
    int offset;
    
    if (pThreadData->src_data == NULL || pThreadData->dst_data == NULL)
        return;

    radius = pThreadData->radius;
    if (radius == 0) {
        av_image_copy_plane(pThreadData->dst_data, pThreadData->dst_linesize,
                            pThreadData->src_data, pThreadData->src_linesize,
                            pThreadData->width, pThreadData->height);
        return;
    }

    srcData   = pThreadData->src_data;
    srcStride = pThreadData->src_linesize;
    dstStride = pThreadData->dst_linesize;
    width     = pThreadData->width;
    height    = pThreadData->height;
    delta     = pThreadData->delta;
    dstData   = pThreadData->dst_data;

    /* For middle and last slice, offset to read radius*2 more lines from previous slice */
    if ((pThreadData->mask & 0x01) != 0x01)
        srcData -= pThreadData->src_linesize * radius * 2;

    if (pThreadData->mask == 0x00) {
        /* Middle slice: read 2*radius more lines from adjacent slices */
        add_h = radius * 4;
    } else if (pThreadData->mask == 0x01 || pThreadData->mask == 0x02) {
        /* First or last slice: read radius*2 more lines */
        add_h = radius * 2;
    } else {
        add_h = 0;
    }
    
    data   = av_malloc(sizeof(float) * width * (height + add_h));
    meanIP = av_malloc(sizeof(float) * width * (height + add_h));
    corrIP = av_malloc(sizeof(float) * width * (height + add_h));
    varIP  = av_malloc(sizeof(float) * width * (height + add_h));
    a      = av_malloc(sizeof(float) * width * (height + add_h));
    b      = av_malloc(sizeof(float) * width * (height + add_h));

    if (!data || !meanIP || !corrIP || !varIP || !a || !b) {
        av_free(data);
        av_free(meanIP);
        av_free(corrIP);
        av_free(varIP);
        av_free(a);
        av_free(b);
        return;
    }

    pTmp = NULL;
    offset = 0;
    
    /* Offset result for middle and last slice */
    if ((pThreadData->mask & 0x01) != 0x01)
        offset = width * radius;

    /* Convert to float and extend by 2*radius */
    dsp.char2float(srcData, data, width, height + add_h, srcStride);
    
    pTmp = data + offset;
    dsp.mean_filter(pTmp, meanIP, width, height + add_h / 2, radius, pThreadData->mask);
    
    /* Square and extend by 2*radius */
    dsp.float_square(data, width, height + add_h);
    
    pTmp = data + offset;
    dsp.mean_filter(pTmp, corrIP, width, height + add_h / 2, radius, pThreadData->mask);
    
    dsp.meanip_corrip_ab(meanIP, corrIP, a, b, width, height + add_h / 2, delta);
    
    pTmp = a + offset;
    dsp.mean_filter(pTmp, meanIP, width, height, radius, pThreadData->mask);
    pTmp = b + offset;
    dsp.mean_filter(pTmp, corrIP, width, height, radius, pThreadData->mask);

    /* Convert float to unsigned char */
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int index = i * width + j;
            dstData[i * dstStride + j] = (unsigned char)av_clipf(
                (meanIP[index] * (float)pThreadData->src_data[i * srcStride + j] / 255.0f +
                 corrIP[index]) * 255.0f, 0.0, 255.0);
        }
    }
    
    av_free(data);
    av_free(meanIP);
    av_free(corrIP);
    av_free(varIP);
    av_free(a);
    av_free(b);
    
    if (g_guided_filter_sync.init_flg) {
        pthread_mutex_lock(&g_guided_filter_sync.mutex);
        g_guided_filter_sync.result_cnt += 1;
        if (g_guided_filter_sync.result_cnt == pThreadData->thread_num)
            pthread_cond_signal(&g_guided_filter_sync.cond);
        pthread_mutex_unlock(&g_guided_filter_sync.mutex);
    }
}

static void guided_filter_slice16(AVFilterContext *ctx, FilterParam *pThreadData)
{
    uint16_t *srcData;
    uint16_t *dstData;
    EnhanceContext *s = ctx->priv;
    float maxData;
    int srcStride;
    int dstStride;
    int width;
    int height;
    int radius;
    float delta;
    int add_h;
    float *data;
    float *meanIP;
    float *corrIP;
    float *varIP;
    float *a;
    float *b;
    float *pTmp;
    int offset;
    
    if (pThreadData->src_data == NULL || pThreadData->dst_data == NULL)
        return;

    radius = pThreadData->radius;
    if (radius == 0) {
        av_image_copy_plane(pThreadData->dst_data, pThreadData->dst_linesize,
                            pThreadData->src_data, pThreadData->src_linesize,
                            pThreadData->width * s->bps, pThreadData->height);
        return;
    }

    maxData = (float)((1 << s->bitdepth) - 1);
    srcData   = (uint16_t *)pThreadData->src_data;
    srcStride = pThreadData->src_linesize / s->bps;
    dstStride = pThreadData->dst_linesize / s->bps;
    width     = pThreadData->width;
    height    = pThreadData->height;
    delta     = pThreadData->delta;
    dstData   = (uint16_t *)pThreadData->dst_data;

    /* For middle and last slice, offset to read radius*2 more lines from previous slice */
    if ((pThreadData->mask & 0x01) != 0x01)
        srcData -= srcStride * radius * 2;

    if (pThreadData->mask == 0x00) {
        /* Middle slice: read 2*radius more lines from adjacent slices */
        add_h = radius * 4;
    } else if (pThreadData->mask == 0x01 || pThreadData->mask == 0x02) {
        /* First or last slice: read radius*2 more lines */
        add_h = radius * 2;
    } else {
        add_h = 0;
    }
    
    data   = av_malloc(sizeof(float) * width * (height + add_h));
    meanIP = av_malloc(sizeof(float) * width * (height + add_h));
    corrIP = av_malloc(sizeof(float) * width * (height + add_h));
    varIP  = av_malloc(sizeof(float) * width * (height + add_h));
    a      = av_malloc(sizeof(float) * width * (height + add_h));
    b      = av_malloc(sizeof(float) * width * (height + add_h));

    if (!data || !meanIP || !corrIP || !varIP || !a || !b) {
        av_free(data);
        av_free(meanIP);
        av_free(corrIP);
        av_free(varIP);
        av_free(a);
        av_free(b);
        return;
    }

    pTmp = NULL;
    offset = 0;
    
    /* Offset result for middle and last slice */
    if ((pThreadData->mask & 0x01) != 0x01)
        offset = width * radius;

    /* Convert to float and extend by 2*radius */
    dsp.char2float16(srcData, data, width, height + add_h, srcStride, maxData);
    
    pTmp = data + offset;
    dsp.mean_filter(pTmp, meanIP, width, height + add_h / 2, radius, pThreadData->mask);
    
    /* Square and extend by 2*radius */
    dsp.float_square(data, width, height + add_h);
    
    pTmp = data + offset;
    dsp.mean_filter(pTmp, corrIP, width, height + add_h / 2, radius, pThreadData->mask);
    
    dsp.meanip_corrip_ab(meanIP, corrIP, a, b, width, height + add_h / 2, delta);
    
    pTmp = a + offset;
    dsp.mean_filter(pTmp, meanIP, width, height, radius, pThreadData->mask);
    pTmp = b + offset;
    dsp.mean_filter(pTmp, corrIP, width, height, radius, pThreadData->mask);

    /* Convert float to uint16_t */
    srcData = (uint16_t *)pThreadData->src_data;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int index = i * width + j;
            dstData[i * dstStride + j] = (uint16_t)av_clipf(
                (meanIP[index] * (float)srcData[i * srcStride + j] / maxData +
                 corrIP[index]) * maxData, 0.0, maxData);
        }
    }
    
    av_free(data);
    av_free(meanIP);
    av_free(corrIP);
    av_free(varIP);
    av_free(a);
    av_free(b);
    
    if (g_guided_filter_sync.init_flg) {
        pthread_mutex_lock(&g_guided_filter_sync.mutex);
        g_guided_filter_sync.result_cnt += 1;
        if (g_guided_filter_sync.result_cnt == pThreadData->thread_num)
            pthread_cond_signal(&g_guided_filter_sync.cond);
        pthread_mutex_unlock(&g_guided_filter_sync.mutex);
    }
}

static int guided_filter_slices(AVFilterContext *ctx, void *arg, int jobnr, int nb_jobs)
{
    GuidedThreadData *pData = (GuidedThreadData *)arg;
    AVFrame *in = pData->in;
    AVFrame *out = pData->out;
    EnhanceContext *s = ctx->priv;
    int cw = AV_CEIL_RSHIFT(in->width, s->hsub), ch = AV_CEIL_RSHIFT(in->height, s->vsub);
    int w[3] = { in->width, cw, cw};
    int h[3] = { in->height, ch, ch};

    int mask = 0x0;
    if(nb_jobs == 1)
    {
        mask = 0x3;//单线程只有一块
    }
    else
    {
        if(jobnr == 0) //第一块
        {
            mask = 0x1;
        }
        else if(jobnr == (nb_jobs - 1)) //最后一块
        {
            mask = 0x2;
        }
        else
        {
            mask = 0x0;
        }
    }
    for (int plane = 0; plane < 3 ; plane++)
    {
        FilterParam pInParam = {0};
        int offset  = h[plane] / nb_jobs;
        int freeRow = h[plane] % nb_jobs;
        pInParam.src_data      = in->data[plane] + jobnr * offset * in->linesize[plane];
        pInParam.src_linesize  = pInParam.src_stride = in->linesize[plane];
        pInParam.dst_linesize  = pInParam.dst_stride = out->linesize[plane];
        pInParam.dst_data      = out->data[plane] + jobnr * offset * out->linesize[plane];
        pInParam.width         = w[plane];
        pInParam.height        = (jobnr == nb_jobs - 1) ? h[plane] / nb_jobs + freeRow : h[plane] / nb_jobs;
        pInParam.delta         = s->eps;
        pInParam.radius        = (plane == 0) ? 1 : 0;
        pInParam.mask          = mask;
        pInParam.thread_num    = nb_jobs;
#if ENHANCE_VF_CPU_FLG_AVX2
        int cpu_flags = av_get_cpu_flags();
        if (EXTERNAL_AVX2(cpu_flags)) {
            if (s->bitdepth > 8)
                guided_filter_avx2_slice16(ctx, &pInParam);
            else
                guided_filter_avx2_slice(ctx, &pInParam);
        } else {
            if (s->bitdepth > 8)
                guided_filter_slice16(ctx, &pInParam);
            else
                guided_filter_slice(ctx, &pInParam);
        }
#else
        if (s->bitdepth > 8)
            guided_filter_slice16(ctx, &pInParam);
        else
            guided_filter_slice(ctx, &pInParam);
#endif
    }
    return 0;
}

static int guided_filter_thread(AVFilterContext *ctx, AVFrame *in, AVFrame *out)
{
    GuidedThreadData threadData = {0};
    EnhanceContext *s = ctx->priv;
    
    g_guided_filter_sync.result_cnt = 0;
    threadData.in = in;
    threadData.out = out;
    ctx->internal->execute(ctx, guided_filter_slices, &threadData, NULL, s->nb_threads);
    
    if (g_guided_filter_sync.init_flg) {
        pthread_mutex_lock(&g_guided_filter_sync.mutex);
        if (g_guided_filter_sync.result_cnt != s->nb_threads)
            pthread_cond_wait(&g_guided_filter_sync.cond, &g_guided_filter_sync.mutex);
        pthread_mutex_unlock(&g_guided_filter_sync.mutex);
    }
    
    return 0;
}

typedef struct ThreadData {
    EnhanceFilterParam *fp;
    uint8_t *dst;
    const uint8_t *src;
    int dst_stride;
    int src_stride;
    int width;
    int height;
} ThreadData;

static int unsharp_slice_8bit(AVFilterContext *ctx, void *arg, int jobnr, int nb_jobs)
{
    ThreadData *td = arg;
    EnhanceFilterParam *fp = td->fp;
    uint32_t **sc = fp->sc;
    uint32_t *sr = fp->sr;
    const uint8_t *src2 = NULL;  //silence a warning
    const int amount = fp->amount;
    const int steps_x = fp->steps_x;
    const int steps_y = fp->steps_y;
    const int scalebits = fp->scalebits;
    const int32_t halfscale = fp->halfscale;

    uint8_t *dst = td->dst;
    const uint8_t *src = td->src;
    const int dst_stride = td->dst_stride;
    const int src_stride = td->src_stride;
    const int width = td->width;
    const int height = td->height;
    const int sc_offset = jobnr * 2 * steps_y;
    const int sr_offset = jobnr * (MAX_MATRIX_SIZE - 1);
    const int slice_start = (height * jobnr) / nb_jobs;
    const int slice_end = (height * (jobnr+1)) / nb_jobs;

    int32_t res;
    int x, y, z;
    uint32_t tmp1, tmp2;
    if (!amount) {
        av_image_copy_plane(dst + slice_start * dst_stride, dst_stride,
                            src + slice_start * src_stride, src_stride,
                            width, slice_end - slice_start);
        return 0;
    }

    for (y = 0; y < 2 * steps_y; y++)
        memset(sc[sc_offset + y], 0, sizeof(sc[y][0]) * (width + 2 * steps_x));

    // if this is not the first tile, we start from (slice_start - steps_y),
    // so we can get smooth result at slice boundary
    if (slice_start > steps_y) {
        src += (slice_start - steps_y) * src_stride;
        dst += (slice_start - steps_y) * dst_stride;
    }

    for (y = -steps_y + slice_start; y < steps_y + slice_end; y++) {
        if (y < height)
            src2 = src;

        memset(sr + sr_offset, 0, sizeof(sr[0]) * (2 * steps_x - 1));
        for (x = -steps_x; x < width + steps_x; x++) {
            tmp1 = x <= 0 ? src2[0] : x >= width ? src2[width-1] : src2[x];
            for (z = 0; z < steps_x * 2; z += 2) {
                tmp2 = sr[sr_offset + z + 0] + tmp1;sr[sr_offset + z + 0] = tmp1;
                tmp1 = sr[sr_offset + z + 1] + tmp2;sr[sr_offset + z + 1] = tmp2;
            }
            for (z = 0; z < steps_y * 2; z += 2) {
                tmp2 = sc[sc_offset + z + 0][x + steps_x] + tmp1;sc[sc_offset + z + 0][x + steps_x] = tmp1;
                tmp1 = sc[sc_offset + z + 1][x + steps_x] + tmp2;sc[sc_offset + z + 1][x + steps_x] = tmp2;
            }
            if (x >= steps_x && y >= (steps_y + slice_start)) {
                const uint8_t *srx = src - steps_y * src_stride + x - steps_x;
                uint8_t *dsx       = dst - steps_y * dst_stride + x - steps_x;

                res = (int32_t)*srx + ((((int32_t) * srx - (int32_t)((tmp1 + halfscale) >> scalebits)) * amount) >> 16);
                *dsx = av_clip_uint8(res);
            }
        }
        if (y >= 0) {
            dst += dst_stride;
            src += src_stride;
        }
    }
    return 0;
}

static int unsharp_slice_16bit(AVFilterContext *ctx, void *arg, int jobnr, int nb_jobs)
{
    ThreadData *td = arg;
    EnhanceFilterParam *fp = td->fp;
    EnhanceContext *s = ctx->priv;                                                                    
    uint32_t **sc = fp->sc;                                                                           
    uint32_t *sr = fp->sr;                                                                            
    const uint16_t *src2 = NULL;                                                              
    const int amount = fp->amount;                                                                    
    const int steps_x = fp->steps_x;                                                                  
    const int steps_y = fp->steps_y;                                                                  
    const int scalebits = fp->scalebits;                                                              
    const int32_t halfscale = fp->halfscale;                                                          
                                                                                                    
    uint16_t *dst = (uint16_t*)td->dst;                                                 
    const uint16_t *src = (const uint16_t *)td->src;                                    
    int dst_stride = td->dst_stride;                                                                  
    int src_stride = td->src_stride;                                                                  
    const int width = td->width;                                                                     
    const int height = td->height;                                                                    
    const int sc_offset = jobnr * 2 * steps_y;                                                        
    const int sr_offset = jobnr * (MAX_MATRIX_SIZE - 1);                                              
    const int slice_start = (height * jobnr) / nb_jobs;                                               
    const int slice_end = (height * (jobnr+1)) / nb_jobs;                                            
                                                                                                      
    int32_t res;                                                                                      
    int x, y, z;                                                                                      
    uint32_t tmp1, tmp2;                                                                              
                                                                                                      
    if (!amount) {                                                                                    
        av_image_copy_plane(td->dst + slice_start * dst_stride, dst_stride,                           
                            td->src + slice_start * src_stride, src_stride,                           
                            width * s->bps, slice_end - slice_start);                                 
        return 0;                                                                                     
    }                                                                                                 
                                                                                                      
    for (y = 0; y < 2 * steps_y; y++)                                                                 
        memset(sc[sc_offset + y], 0, sizeof(sc[y][0]) * (width + 2 * steps_x));                       
                                                                                                      
    dst_stride = dst_stride / s->bps;                                                                 
    src_stride = src_stride / s->bps;                                                                 
    /* if this is not the first tile, we start from (slice_start - steps_y) */                        
    /* so we can get smooth result at slice boundary */                                               
    if (slice_start > steps_y) {                                                                      
        src += (slice_start - steps_y) * src_stride;                                                  
        dst += (slice_start - steps_y) * dst_stride;                                                  
    }                                                                                                 
                                                                                                      
    for (y = -steps_y + slice_start; y < steps_y + slice_end; y++) {                                  
        if (y < height)                                                                               
            src2 = src;                                                                               
                                                                                                      
        memset(sr + sr_offset, 0, sizeof(sr[0]) * (2 * steps_x - 1));                                 
        for (x = -steps_x; x < width + steps_x; x++) {                                                
            tmp1 = x <= 0 ? src2[0] : x >= width ? src2[width-1] : src2[x];                           
            for (z = 0; z < steps_x * 2; z += 2) {                                                    
                tmp2 = sr[sr_offset + z + 0] + tmp1; sr[sr_offset + z + 0] = tmp1;                    
                tmp1 = sr[sr_offset + z + 1] + tmp2; sr[sr_offset + z + 1] = tmp2;                    
            }                                                                                         
            for (z = 0; z < steps_y * 2; z += 2) {                                                    
                tmp2 = sc[sc_offset + z + 0][x + steps_x] + tmp1;                                     
                sc[sc_offset + z + 0][x + steps_x] = tmp1;                                            
                tmp1 = sc[sc_offset + z + 1][x + steps_x] + tmp2;                                     
                sc[sc_offset + z + 1][x + steps_x] = tmp2;                                            
            }                                                                                         
            if (x >= steps_x && y >= (steps_y + slice_start)) {                                       
                const uint16_t *srx = src - steps_y * src_stride + x - steps_x;                
                uint16_t *dsx       = dst - steps_y * dst_stride + x - steps_x;                
                                                                                                      
                res = (int32_t)*srx + ((((int32_t) * srx -                                            
                      (int32_t)((tmp1 + halfscale) >> scalebits)) * amount) >> (8+16));            
                *dsx = av_clip_uint16(res);                                                      
            }                                                                                         
        }                                                                                             
        if (y >= 0) {                                                                                 
            dst += dst_stride;                                                                        
            src += src_stride;                                                                        
        }                                                                                             
    }                                                                                                 
    return 0;                                                                                         
}

static int apply_enhance_c(AVFilterContext *ctx, AVFrame *in, AVFrame *out)
{
    AVFilterLink *inlink = ctx->inputs[0];
    EnhanceContext *s = ctx->priv;
    int i, plane_w[3], plane_h[3];
    EnhanceFilterParam *fp[3];
    ThreadData td;
    AVFrame *guided = NULL;
    AVFrame *src_frame = NULL;

    /* Do guided filter before enhance */
    if (s->is_guided) {
        guided = av_frame_alloc();
        if (guided) {
            guided->format = inlink->format;
            guided->width  = in->width;
            guided->height = in->height;
            if (!av_frame_get_buffer(guided, 32))
                guided_filter_thread(ctx, in, guided);
            src_frame = guided;
        }
    }
    if (!src_frame)
        src_frame = in;
    
    plane_w[0] = inlink->w;
    plane_w[1] = plane_w[2] = AV_CEIL_RSHIFT(inlink->w, s->hsub);
    plane_h[0] = inlink->h;
    plane_h[1] = plane_h[2] = AV_CEIL_RSHIFT(inlink->h, s->vsub);
    fp[0] = &s->luma;
    fp[1] = fp[2] = &s->chroma;
    
    for (i = 0; i < 3; i++) {
        td.fp = fp[i];
        td.dst = out->data[i];
        td.src = src_frame->data[i];
        td.width = plane_w[i];
        td.height = plane_h[i];
        td.dst_stride = out->linesize[i];
        td.src_stride = src_frame->linesize[i];
        ctx->internal->execute(ctx, s->unsharp_slice, &td, NULL,
                               FFMIN(plane_h[i], s->nb_threads));
    }
    
    if (guided)
        av_frame_free(&guided);
    
    return 0;
}

static void set_filter_param(EnhanceFilterParam *fp, int msize_x, int msize_y, float amount)
{
    fp->msize_x = msize_x;
    fp->msize_y = msize_y;
    fp->amount = amount * 65536.0;

    fp->steps_x = msize_x / 2;
    fp->steps_y = msize_y / 2;
    fp->scalebits = (fp->steps_x + fp->steps_y) * 2;
    fp->halfscale = 1 << (fp->scalebits - 1);
}

static av_cold int init(AVFilterContext *ctx)
{
    EnhanceContext *s = ctx->priv;

    set_filter_param(&s->luma,   s->lmsize_x, s->lmsize_y, s->lamount);
    set_filter_param(&s->chroma, s->cmsize_x, s->cmsize_y, s->camount);

    if (s->luma.scalebits >= 26 || s->chroma.scalebits >= 26) {
        av_log(ctx, AV_LOG_ERROR, "luma or chroma matrix size too big\n");
        return AVERROR(EINVAL);
    }
    s->apply_enhance = apply_enhance_c;

    /* Initialize DSP function pointers */
    enhance_dsp_init();

    /* Use FFmpeg internal synchronization, no need for custom sync */
    g_guided_filter_sync.init_flg = 0;
    return 0;
}

static int query_formats(AVFilterContext *ctx)
{
    static const enum AVPixelFormat pix_fmts[] = {
    AV_PIX_FMT_YUV420P,  AV_PIX_FMT_YUV422P,  AV_PIX_FMT_YUV444P,  AV_PIX_FMT_YUV410P,
    AV_PIX_FMT_YUV411P,  AV_PIX_FMT_YUV440P,  AV_PIX_FMT_YUVJ420P, AV_PIX_FMT_YUVJ422P,
    AV_PIX_FMT_YUV420P9, AV_PIX_FMT_YUV422P9, AV_PIX_FMT_YUV444P9,
    AV_PIX_FMT_YUV420P10, AV_PIX_FMT_YUV422P10, AV_PIX_FMT_YUV444P10, AV_PIX_FMT_YUV440P10,
    AV_PIX_FMT_YUV420P12, AV_PIX_FMT_YUV422P12, AV_PIX_FMT_YUV444P12, AV_PIX_FMT_YUV440P12,
    AV_PIX_FMT_YUV420P16, AV_PIX_FMT_YUV422P16, AV_PIX_FMT_YUV444P16,
    AV_PIX_FMT_YUVJ444P, AV_PIX_FMT_YUVJ440P, AV_PIX_FMT_NONE
    };

    AVFilterFormats *fmts_list = ff_make_format_list(pix_fmts);
    if (!fmts_list)
        return AVERROR(ENOMEM);
    return ff_set_common_formats(ctx, fmts_list);
}

static int init_filter_param(AVFilterContext *ctx, EnhanceFilterParam *fp, const char *effect_type, int width)
{
    int z;
    EnhanceContext *s = ctx->priv;
    const char *effect = fp->amount == 0 ? "none" : fp->amount < 0 ? "blur" : "sharpen";

    if  (!(fp->msize_x & fp->msize_y & 1)) {
        av_log(ctx, AV_LOG_ERROR,
               "Invalid even size for %s matrix size %dx%d\n",
               effect_type, fp->msize_x, fp->msize_y);
        return AVERROR(EINVAL);
    }

    av_log(ctx, AV_LOG_VERBOSE, "effect:%s type:%s msize_x:%d msize_y:%d amount:%0.2f\n",
           effect, effect_type, fp->msize_x, fp->msize_y, fp->amount / 65535.0);

    fp->sr = av_malloc_array((MAX_MATRIX_SIZE - 1) * s->nb_threads, sizeof(uint32_t));
    fp->sc = av_mallocz_array(2 * fp->steps_y * s->nb_threads, sizeof(uint32_t *));
    if (!fp->sr || !fp->sc)
        return AVERROR(ENOMEM);

    for (z = 0; z < 2 * fp->steps_y * s->nb_threads; z++)
        if (!(fp->sc[z] = av_malloc_array(width + 2 * fp->steps_x,
                                          sizeof(*(fp->sc[z])))))
            return AVERROR(ENOMEM);

    return 0;
}

static int config_input(AVFilterLink *inlink)
{
    EnhanceContext *s = inlink->dst->priv;
    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(inlink->format);
    int ret;
    
    s->hsub = desc->log2_chroma_w;
    s->vsub = desc->log2_chroma_h;
    s->bitdepth = desc->comp[0].depth;
    s->bps = s->bitdepth > 8 ? 2 : 1;
    s->unsharp_slice = s->bitdepth > 8 ? unsharp_slice_16bit : unsharp_slice_8bit;
    
    /* Set max_pixel pointer for 16-bit formats */
    if (s->bitdepth == 10)
        s->max_pixel = max_pixel_10bit;
    else if (s->bitdepth == 12)
        s->max_pixel = max_pixel_12bit;
    else
        s->max_pixel = NULL;
    // ensure (height / nb_threads) > 4 * steps_y,
    // so that we don't have too much overlap between two threads
    s->nb_threads = FFMIN(ff_filter_get_nb_threads(inlink->dst),
                          inlink->h / (4 * s->luma.steps_y));

    ret = init_filter_param(inlink->dst, &s->luma,   "luma",   inlink->w);
    if (ret < 0)
        return ret;
    ret = init_filter_param(inlink->dst, &s->chroma, "chroma", AV_CEIL_RSHIFT(inlink->w, s->hsub));
    if (ret < 0)
        return ret;

    return 0;
}

static void free_filter_param(EnhanceFilterParam *fp, int nb_threads)
{
    int z;

    if (fp->sc) {
        for (z = 0; z < 2 * fp->steps_y * nb_threads; z++)
            av_freep(&fp->sc[z]);
        av_freep(&fp->sc);
    }
    av_freep(&fp->sr);
}

static av_cold void uninit(AVFilterContext *ctx)
{
    EnhanceContext *s = ctx->priv;

    free_filter_param(&s->luma, s->nb_threads);
    free_filter_param(&s->chroma, s->nb_threads);
    
    if (g_guided_filter_sync.init_flg) {
        pthread_mutex_destroy(&g_guided_filter_sync.mutex);
        pthread_cond_destroy(&g_guided_filter_sync.cond);
        g_guided_filter_sync.init_flg = 0;
    }
}

static int filter_frame(AVFilterLink *link, AVFrame *in)
{
    EnhanceContext *s = link->dst->priv;
    AVFilterLink *outlink   = link->dst->outputs[0];
    AVFrame *out;
    int ret = 0;

    out = ff_get_video_buffer(outlink, outlink->w, outlink->h);
    if (!out) {
        av_frame_free(&in);
        return AVERROR(ENOMEM);
    }
    av_frame_copy_props(out, in);

    ret = s->apply_enhance(link->dst, in, out);

    av_frame_free(&in);

    if (ret < 0) {
        av_frame_free(&out);
        return ret;
    }
    return ff_filter_frame(outlink, out);
}

#define OFFSET(x) offsetof(EnhanceContext, x)
#define FLAGS (AV_OPT_FLAG_FILTERING_PARAM | AV_OPT_FLAG_VIDEO_PARAM)
#define MIN_SIZE 3
#define MAX_SIZE 23

static const AVOption imgenhance_options[] = {
    { "is_guided",   "if do guided filter",   OFFSET(is_guided), AV_OPT_TYPE_INT,   { .i64 = 1 }, 0, 1, FLAGS },
    { "eps",    "Eps to guided filter",  OFFSET(eps), AV_OPT_TYPE_FLOAT, {.dbl = 0.0005}, 0, 1, .flags = FLAGS },
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
    { "opencl",         "ignored",                           OFFSET(opencl),   AV_OPT_TYPE_BOOL,  { .i64 = 0 },        0,        1, FLAGS },
    { NULL }
};

AVFILTER_DEFINE_CLASS(imgenhance);

static const AVFilterPad avfilter_vf_imgenhance_inputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .filter_frame = filter_frame,
        .config_props = config_input,
    },
    { NULL }
};

static const AVFilterPad avfilter_vf_imgenhance_outputs[] = {
    {
        .name = "default",
        .type = AVMEDIA_TYPE_VIDEO,
    },
    { NULL }
};

AVFilter ff_vf_imgenhance = {
    .name          = "imgenhance",
    .description   = NULL_IF_CONFIG_SMALL("Enhance the input video."),
    .priv_size     = sizeof(EnhanceContext),
    .priv_class    = &imgenhance_class,
    .init          = init,
    .uninit        = uninit,
    .query_formats = query_formats,
    .inputs        = avfilter_vf_imgenhance_inputs,
    .outputs       = avfilter_vf_imgenhance_outputs,
    .flags         = AVFILTER_FLAG_SUPPORT_TIMELINE_GENERIC | AVFILTER_FLAG_SLICE_THREADS,
};
