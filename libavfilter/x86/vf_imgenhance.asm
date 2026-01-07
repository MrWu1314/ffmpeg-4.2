;******************************************************************************
;* x86-optimized functions for gradfun filter
;*
;* This file is part of FFmpeg.
;*
;* FFmpeg is free software; you can redistribute it and/or
;* modify it under the terms of the GNU Lesser General Public
;* License as published by the Free Software Foundation; either
;* version 2.1 of the License, or (at your option) any later version.
;*
;* FFmpeg is distributed in the hope that it will be useful,
;* but WITHOUT ANY WARRANTY; without even the implied warranty of
;* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
;* Lesser General Public License for more details.
;*
;* You should have received a copy of the GNU Lesser General Public
;* License along with FFmpeg; if not, write to the Free Software
;* Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
;******************************************************************************

%include "libavutil/x86/x86util.asm"

SECTION_RODATA 32

const md_10, times 8 dd 0.00097752 ;1 / 1023
const md_12, times 8 dd 0.0002442  ;1 / 1023

SECTION .text
%if HAVE_AVX2_EXTERNAL
;MeanFilter_Char2Float_AVX2_16(uint16_t* srcData,float* dstData,int width,int height,int src_stride,float *maxData)
INIT_YMM avx2
cglobal MeanFilter_Piex2Float_16, 6,10,4
    vmovaps         m3, [r5]
.loopH:
    mov             r8, r0
    mov             r7, r1
    mov             r9d, r2d
.loopL:
    cmp             r9d, 8
    jle             .proc8 ;
    vpxor           m0, m0
    vpmovzxwd       m0, [r8]
    vcvtdq2ps       m0, m0
    vmulps          m0, m0, m3
    vmovups         [r7],m0    
      
    sub             r9d, 8
    add             r8, 16
    add             r7, 32
    jmp             .loopL
.proc8:

%rep 8
    movd          xm0, [r8] ;
    pmovzxwd      xm1, xm0    ;
    cvtdq2ps      xm2, xm1   ;
    mulps         xm2, xm3    ;
    movd          [r7],xm2  ;
    add           r8, 2
    add           r7, 4
    dec           r9d
    jz           .next
%endrep

.next:
    mov         r6d, r4d
    shl         r6d, 1
    add         r0, r6   
    mov         r6d, r2d          
    shl         r6d, 2
    add         r1, r6
    dec         r3d
    jnz        .loopH
    RET
;MeanFilter_Float2Char_AVX2_16(uint16_t* srcData,uint16_t* dstData,
;                             float* pMeanIP,float *pCorrIP,
;                             int width,int height,
;                             int src_stride,int dst_stride,
;                             float maxData)
INIT_YMM avx2
cglobal MeanFilter_Float2Piex_16, 9,14,6
    vmovaps         m3, [r8] ;
    vrcpps          m5, m3   ;
    vcvtps2dq       m4, m3  ;
.loopH:
    mov             r12, r0 ;
    mov             r11, r1 ;
    mov             r10, r2 ;
    mov             r9,  r3 ;
    mov             r8d, r4d;
    
.loopL:
    cmp             r8d, 8
    jle             .proc8 ;
    vmovups         m0, [r10];
    vpmovzxwd       m1, [r12];
    vcvtdq2ps       m1, m1   ;
    vmulps          m1, m0   ;
    vmulps          m1, m5   ;
    vmovups         m0, [r9] ;
    vaddps          m1, m0  
    vmulps          m1, m3   ;
    vcvtps2dq       m1, m1
    vpxor           m0, m0
    vpmaxsd         m1, m0
    vpminsd         m1, m4
vextractf128        xm0,m1,1
    packssdw        xm1,xm0
    movdqu          [r11], xm1
      
    sub             r8d, 8
    add             r12, 16
    add             r11, 16
    add             r10, 32
    add             r9,  32
    jmp             .loopL

.proc8:
%rep 8
    ;mov           r10, r2
    movd          xm0, [r10]
    movd          xm1, [r12] ;
    pmovzxwd      xm1, xm1   ;
    cvtdq2ps      xm1, xm1   ;
    mulps         xm1, xm0   ;
    mulps         xm1, xm5   ;
    movups        xm0, [r9] ;
    addps         xm1, xm0  
    mulps         xm1, xm3   ;
    cvtps2dq      xm1, xm1
    pxor          xm0, xm0
    pmaxsd        xm1, xm0
    pminsd        xm1, xm4
    packssdw      xm1, xm1
    movq          r13, xm1
    mov           [r11], r13w
   
    add           r12, 2
    add           r11, 2
    add           r10, 4
    add           r9,  4
    dec           r8d
    jz           .next
%endrep

.next:
    mov         r10d, r6d
    shl         r10d, 1
    add         r0, r10                
    mov         r10d, r7d
    shl         r10d, 1
    add         r1, r10

    mov         r10d, r4d
    shl         r10d, 2
    add         r2, r10               
    add         r3, r10

    dec         r5d
    jnz        .loopH
.endl:
    RET
%endif