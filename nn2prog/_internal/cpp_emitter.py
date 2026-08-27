"""Emit standalone C++ from a lowered NN2Prog program."""

from dataclasses import dataclass
import re

from .lowering import elements


@dataclass(frozen=True)
class CppContext:
    output_dir: object
    graph: dict
    tensors: dict
    program: object
    globals: tuple
    argmax_output: bool
    decision_tensor: object
    masked_simd_block: int = 32


def emit_cpp(context):
    out = context.output_dir
    graph = context.graph
    tensors = context.tensors
    ops = graph["operators"]
    program = context.program
    target = program.target
    globals_ = context.globals
    argmax_output = context.argmax_output
    decision_tensor = context.decision_tensor
    masked_simd_block = context.masked_simd_block
    input_tensor = graph["inputs"][0]
    output_tensor = graph["outputs"][0]
    input_elements = elements(tensors[input_tensor])
    output_elements = elements(tensors[output_tensor])
    class_name = "Model"
    return_type = ("std::uint8_t" if argmax_output or output_elements == 1
                   else f"std::array<std::int8_t,{output_elements}>")

    handles = {}
    for op in ops:
        if op["opcode"] == "VAR_HANDLE":
            handles[op["outputs"][0]] = op["options"]["shared_name"]
    state_sizes = {}
    state_zero_points = {}
    for op in ops:
        if op["opcode"] == "READ_VARIABLE":
            name = handles[op["inputs"][0]]
            state_sizes[name] = elements(tensors[op["outputs"][0]])
            state_zero_points[name] = tensors[op["outputs"][0]]["quantization"]["zero_point"][0]
    state_field = {name: "state_" + re.sub(r"[^a-zA-Z0-9_]", "_", name) for name in state_sizes}

    header = out / "model.h"
    with header.open("w") as h:
        h.write("""#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
namespace nn2prog::generated {
class CLASS_NAME {
 public:
  CLASS_NAME() { reset(); }
  void reset();
  RETURN_TYPE invoke(const std::array<std::int8_t,INPUT_ELEMENTS>& input);
""".replace("CLASS_NAME", class_name).replace("INPUT_ELEMENTS", str(input_elements)).replace("RETURN_TYPE", return_type))
        h.write("  static std::size_t scratch_bytes();\n")
        h.write(""" private:
""")
        for name, size in state_sizes.items():
            h.write(f"  std::array<std::int8_t,{size}> {state_field[name]}{{}};\n")
        h.write("};\n}\n")

    def ptr(index):
        if index == input_tensor: return "input"
        return f"t{index}"
    def raw(index): return f"sg0_tensor{index}_bytes"
    def nhwc(index):
        shape = tensors[index]["shape"]
        if len(shape) != 4 or shape[0] != 1:
            raise ValueError(f"tensor {index} must have a single-batch NHWC shape")
        return shape[1], shape[2], shape[3]
    def padding_before(input_size, output_size, filter_size, stride, dilation, padding):
        if padding == "VALID": return 0
        if padding != "SAME": raise ValueError(f"unsupported padding {padding}")
        effective = (filter_size - 1) * dilation + 1
        return max(0, (output_size - 1) * stride + effective - input_size) // 2
    source = out / "model.cpp"
    with source.open("w") as c:
        if target == "esp32":
            c.write("#define NN2PROG_GENERATED_ESP32 1\n")
        c.write("""#include "model.h"
#include "model.constants.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <new>
#include <type_traits>
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif
namespace nn2prog::generated {
namespace {
#if defined(ESP_PLATFORM) && defined(NN2PROG_GENERATED_ESP32)
#define NN2PROG_ESP32_IRAM __attribute__((section(".iram1")))
#else
#define NN2PROG_ESP32_IRAM
#endif
constexpr std::int8_t signed_byte(std::uint8_t x) { return x < 128 ? static_cast<std::int8_t>(x) : static_cast<std::int8_t>(static_cast<int>(x)-256); }
std::int32_t load_i32(const std::uint8_t* p) { return static_cast<std::int32_t>(static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1])<<8) | (static_cast<std::uint32_t>(p[2])<<16) | (static_cast<std::uint32_t>(p[3])<<24)); }
std::int32_t saturating_high_mul(std::int32_t a, std::int32_t b) {
  if (a == std::numeric_limits<std::int32_t>::min() && b == a) return std::numeric_limits<std::int32_t>::max();
  const std::int64_t product=static_cast<std::int64_t>(a)*b;
  const std::int64_t nudge=product>=0 ? (std::int64_t{1}<<30) : (1-(std::int64_t{1}<<30));
  return static_cast<std::int32_t>((product+nudge)/(std::int64_t{1}<<31));
}
std::int32_t rounding_divide_pot(std::int32_t x,int exponent) {
  if (!exponent) return x;
  const std::uint32_t mask=(std::uint32_t{1}<<exponent)-1;
  const std::uint32_t remainder=static_cast<std::uint32_t>(x)&mask;
  const std::uint32_t threshold=(mask>>1)+(x<0);
  return (x>>exponent)+(remainder>threshold);
}
NN2PROG_ESP32_IRAM std::int32_t requantize(std::int32_t x,std::int32_t multiplier,int shift) {
  const int left=shift>0?shift:0, right=shift>0?0:-shift;
  return rounding_divide_pot(saturating_high_mul(x*(1<<left),multiplier),right);
}
#if defined(__x86_64__) || defined(__i386__)
__attribute__((target("avx2"))) bool any_nonzero_16(const std::int8_t* input,std::int8_t zero){
  const __m128i values=_mm_loadu_si128(reinterpret_cast<const __m128i*>(input));
  const __m128i zeros=_mm_set1_epi8(zero);
  return static_cast<unsigned>(_mm_movemask_epi8(_mm_cmpeq_epi8(values,zeros)))!=0xffffu;
}
__attribute__((target("avx2"))) std::int32_t dot_16_zero_minus_128(const std::int8_t* input,const std::uint8_t* weights){
  const __m128i inputs=_mm_xor_si128(_mm_loadu_si128(reinterpret_cast<const __m128i*>(input)),_mm_set1_epi8(static_cast<char>(0x80)));
  const __m128i signed_weights=_mm_loadu_si128(reinterpret_cast<const __m128i*>(weights));
  const __m128i input_lo=_mm_cvtepu8_epi16(inputs),input_hi=_mm_cvtepu8_epi16(_mm_srli_si128(inputs,8));
  const __m128i weight_lo=_mm_cvtepi8_epi16(signed_weights),weight_hi=_mm_cvtepi8_epi16(_mm_srli_si128(signed_weights,8));
  const __m128i ones=_mm_set1_epi16(1);
  __m128i sums=_mm_add_epi32(_mm_madd_epi16(_mm_mullo_epi16(input_lo,weight_lo),ones),
                            _mm_madd_epi16(_mm_mullo_epi16(input_hi,weight_hi),ones));
  sums=_mm_hadd_epi32(sums,sums);sums=_mm_hadd_epi32(sums,sums);
  return _mm_cvtsi128_si32(sums);
}
#endif
template<std::size_t N,std::size_t W,std::size_t B,std::size_t O>
void dense(const std::array<std::int8_t,N>& in,const std::array<std::uint8_t,W>& weights,const std::array<std::uint8_t,B>& bias,
           std::array<std::int8_t,O>& out,int input_zero,int output_zero,const std::array<std::int32_t,O>& mult,const std::array<int,O>& shift,int amin,int amax) {
  static_assert(W==N*O && B==O*4);
  for(std::size_t oc=0;oc<O;++oc){ std::int32_t acc=load_i32(bias.data()+oc*4);
    for(std::size_t i=0;i<N;++i) acc += static_cast<std::int32_t>(signed_byte(weights[oc*N+i]))*(static_cast<int>(in[i])-input_zero);
    acc=requantize(acc,mult[oc],shift[oc])+output_zero; out[oc]=static_cast<std::int8_t>(std::clamp<std::int32_t>(acc,amin,amax)); }
}
template<std::size_t IH,std::size_t IW,std::size_t IC,std::size_t FH,std::size_t FW,std::size_t OC,
         std::size_t OH,std::size_t OW,typename Input,std::size_t W,std::size_t B>
void conv2d_nhwc(const Input& in,const std::array<std::uint8_t,W>& weights,const std::array<std::uint8_t,B>& bias,
           std::array<std::int8_t,OH*OW*OC>& out,int stride_h,int stride_w,int dilation_h,int dilation_w,
           int pad_h,int pad_w,int input_zero,int weight_zero,int output_zero,
           const std::array<std::int32_t,OC>& mult,const std::array<int,OC>& shift,int amin,int amax) {
  static_assert(W==OC*FH*FW*IC && B==OC*4);
  for(std::size_t oy=0;oy<OH;++oy)for(std::size_t ox=0;ox<OW;++ox)for(std::size_t oc=0;oc<OC;++oc){
    std::int32_t acc=load_i32(bias.data()+oc*4);
    for(std::size_t fy=0;fy<FH;++fy)for(std::size_t fx=0;fx<FW;++fx){
      const int iy=static_cast<int>(oy)*stride_h+static_cast<int>(fy)*dilation_h-pad_h;
      const int ix=static_cast<int>(ox)*stride_w+static_cast<int>(fx)*dilation_w-pad_w;
      if(iy<0||iy>=static_cast<int>(IH)||ix<0||ix>=static_cast<int>(IW))continue;
      for(std::size_t ic=0;ic<IC;++ic){
        const int a=static_cast<int>(in[(static_cast<std::size_t>(iy)*IW+static_cast<std::size_t>(ix))*IC+ic])-input_zero;
        const int w=static_cast<int>(signed_byte(weights[((oc*FH+fy)*FW+fx)*IC+ic]))-weight_zero;
        acc+=a*w;
      }
    }
    acc=requantize(acc,mult[oc],shift[oc])+output_zero;
    out[(oy*OW+ox)*OC+oc]=static_cast<std::int8_t>(std::clamp<std::int32_t>(acc,amin,amax));
  }
}
template<std::size_t IH,std::size_t IW,std::size_t IC,std::size_t OC,std::size_t OH,std::size_t OW,
         typename Input,std::size_t W>
NN2PROG_ESP32_IRAM void esp32_conv_1x1(const Input& in,const std::array<std::uint8_t,W>& weights,
           const std::array<std::int32_t,OC>& adjusted_bias,std::array<std::int8_t,OH*OW*OC>& out,
           int stride_h,int stride_w,int output_zero,const std::array<std::int32_t,OC>& mult,
           const std::array<int,OC>& shift,int amin,int amax) {
  static_assert(W==OC*IC);
  for(std::size_t oy=0;oy<OH;++oy)for(std::size_t ox=0;ox<OW;++ox){
    const auto* input=in.data()+(oy*stride_h*IW+ox*stride_w)*IC;
    const auto* filter=weights.data();
    auto* output=out.data()+(oy*OW+ox)*OC;
    for(std::size_t oc=0;oc<OC;++oc){
      std::int32_t acc=adjusted_bias[oc];std::size_t ic=0;
      for(;ic+8<=IC;ic+=8){
        acc+=static_cast<std::int32_t>(input[ic+0])*signed_byte(filter[ic+0]);
        acc+=static_cast<std::int32_t>(input[ic+1])*signed_byte(filter[ic+1]);
        acc+=static_cast<std::int32_t>(input[ic+2])*signed_byte(filter[ic+2]);
        acc+=static_cast<std::int32_t>(input[ic+3])*signed_byte(filter[ic+3]);
        acc+=static_cast<std::int32_t>(input[ic+4])*signed_byte(filter[ic+4]);
        acc+=static_cast<std::int32_t>(input[ic+5])*signed_byte(filter[ic+5]);
        acc+=static_cast<std::int32_t>(input[ic+6])*signed_byte(filter[ic+6]);
        acc+=static_cast<std::int32_t>(input[ic+7])*signed_byte(filter[ic+7]);
      }
      for(;ic<IC;++ic)acc+=static_cast<std::int32_t>(input[ic])*signed_byte(filter[ic]);
      filter+=IC;
      acc=requantize(acc,mult[oc],shift[oc])+output_zero;
      output[oc]=static_cast<std::int8_t>(std::clamp<std::int32_t>(acc,amin,amax));
    }
  }
}
template<std::size_t IH,std::size_t IW,std::size_t IC,std::size_t FH,std::size_t FW,std::size_t OC,
         std::size_t OH,std::size_t OW,typename Input,std::size_t W,std::size_t B>
NN2PROG_ESP32_IRAM void esp32_conv_nhwc(const Input& in,const std::array<std::uint8_t,W>& weights,
           const std::array<std::uint8_t,B>& bias,std::array<std::int8_t,OH*OW*OC>& out,
           int stride_h,int stride_w,int pad_h,int pad_w,int input_zero,int output_zero,
           const std::array<std::int32_t,OC>& mult,const std::array<int,OC>& shift,int amin,int amax) {
  static_assert(W==OC*FH*FW*IC && B==OC*4);
  for(std::size_t oy=0;oy<OH;++oy)for(std::size_t ox=0;ox<OW;++ox){
    const int base_y=static_cast<int>(oy)*stride_h-pad_h;
    const int base_x=static_cast<int>(ox)*stride_w-pad_w;
    const int fy_begin=std::max(0,-base_y),fy_end=std::min(static_cast<int>(FH),static_cast<int>(IH)-base_y);
    const int fx_begin=std::max(0,-base_x),fx_end=std::min(static_cast<int>(FW),static_cast<int>(IW)-base_x);
    auto* output=out.data()+(oy*OW+ox)*OC;
    for(std::size_t oc=0;oc<OC;++oc){
      std::int32_t acc=load_i32(bias.data()+oc*4);
      for(int fy=fy_begin;fy<fy_end;++fy)for(int fx=fx_begin;fx<fx_end;++fx){
        const auto* input=in.data()+((base_y+fy)*IW+(base_x+fx))*IC;
        const auto* filter=weights.data()+((oc*FH+fy)*FW+fx)*IC;
        std::size_t ic=0;
        for(;ic+8<=IC;ic+=8){
          acc+=(static_cast<int>(input[ic+0])-input_zero)*signed_byte(filter[ic+0]);
          acc+=(static_cast<int>(input[ic+1])-input_zero)*signed_byte(filter[ic+1]);
          acc+=(static_cast<int>(input[ic+2])-input_zero)*signed_byte(filter[ic+2]);
          acc+=(static_cast<int>(input[ic+3])-input_zero)*signed_byte(filter[ic+3]);
          acc+=(static_cast<int>(input[ic+4])-input_zero)*signed_byte(filter[ic+4]);
          acc+=(static_cast<int>(input[ic+5])-input_zero)*signed_byte(filter[ic+5]);
          acc+=(static_cast<int>(input[ic+6])-input_zero)*signed_byte(filter[ic+6]);
          acc+=(static_cast<int>(input[ic+7])-input_zero)*signed_byte(filter[ic+7]);
        }
        for(;ic<IC;++ic)acc+=(static_cast<int>(input[ic])-input_zero)*signed_byte(filter[ic]);
      }
      acc=requantize(acc,mult[oc],shift[oc])+output_zero;
      output[oc]=static_cast<std::int8_t>(std::clamp<std::int32_t>(acc,amin,amax));
    }
  }
}
template<std::size_t IH,std::size_t IW,std::size_t IC,std::size_t FH,std::size_t FW,std::size_t DM,
         std::size_t OH,std::size_t OW,typename Input,std::size_t W,std::size_t B>
void depthwise_nhwc(const Input& in,const std::array<std::uint8_t,W>& weights,const std::array<std::uint8_t,B>& bias,
           std::array<std::int8_t,OH*OW*IC*DM>& out,int stride_h,int stride_w,int dilation_h,int dilation_w,
           int pad_h,int pad_w,int input_zero,int weight_zero,int output_zero,
           const std::array<std::int32_t,IC*DM>& mult,const std::array<int,IC*DM>& shift,int amin,int amax) {
  constexpr std::size_t OC=IC*DM; static_assert(W==FH*FW*OC && B==OC*4);
  for(std::size_t oy=0;oy<OH;++oy)for(std::size_t ox=0;ox<OW;++ox)for(std::size_t ic=0;ic<IC;++ic)for(std::size_t m=0;m<DM;++m){
    const std::size_t oc=ic*DM+m; std::int32_t acc=load_i32(bias.data()+oc*4);
    for(std::size_t fy=0;fy<FH;++fy)for(std::size_t fx=0;fx<FW;++fx){
      const int iy=static_cast<int>(oy)*stride_h+static_cast<int>(fy)*dilation_h-pad_h;
      const int ix=static_cast<int>(ox)*stride_w+static_cast<int>(fx)*dilation_w-pad_w;
      if(iy<0||iy>=static_cast<int>(IH)||ix<0||ix>=static_cast<int>(IW))continue;
      const int a=static_cast<int>(in[(static_cast<std::size_t>(iy)*IW+static_cast<std::size_t>(ix))*IC+ic])-input_zero;
      const int w=static_cast<int>(signed_byte(weights[(fy*FW+fx)*OC+oc]))-weight_zero;
      acc+=a*w;
    }
    acc=requantize(acc,mult[oc],shift[oc])+output_zero;
    out[(oy*OW+ox)*OC+oc]=static_cast<std::int8_t>(std::clamp<std::int32_t>(acc,amin,amax));
  }
}
template<std::size_t IH,std::size_t IW,std::size_t C,std::size_t FH,std::size_t FW,
         std::size_t OH,std::size_t OW,typename Input,std::size_t W,std::size_t B>
NN2PROG_ESP32_IRAM void esp32_depthwise_channels4(const Input& in,const std::array<std::uint8_t,W>& weights,
           const std::array<std::uint8_t,B>& bias,std::array<std::int8_t,OH*OW*C>& out,
           int stride_h,int stride_w,int pad_h,int pad_w,int input_zero,int output_zero,
           const std::array<std::int32_t,C>& mult,const std::array<int,C>& shift,int amin,int amax) {
  static_assert(W==FH*FW*C && B==C*4 && C%4==0);
  for(std::size_t oy=0;oy<OH;++oy)for(std::size_t ox=0;ox<OW;++ox){
    const int base_y=static_cast<int>(oy)*stride_h-pad_h;
    const int base_x=static_cast<int>(ox)*stride_w-pad_w;
    const int fy_begin=std::max(0,-base_y),fy_end=std::min(static_cast<int>(FH),static_cast<int>(IH)-base_y);
    const int fx_begin=std::max(0,-base_x),fx_end=std::min(static_cast<int>(FW),static_cast<int>(IW)-base_x);
    auto* output=out.data()+(oy*OW+ox)*C;
    for(std::size_t c=0;c<C;c+=4){
      std::int32_t a0=load_i32(bias.data()+(c+0)*4),a1=load_i32(bias.data()+(c+1)*4);
      std::int32_t a2=load_i32(bias.data()+(c+2)*4),a3=load_i32(bias.data()+(c+3)*4);
      for(int fy=fy_begin;fy<fy_end;++fy)for(int fx=fx_begin;fx<fx_end;++fx){
        const auto* input=in.data()+((base_y+fy)*IW+(base_x+fx))*C+c;
        const auto* filter=weights.data()+(fy*FW+fx)*C+c;
        a0+=(static_cast<int>(input[0])-input_zero)*signed_byte(filter[0]);
        a1+=(static_cast<int>(input[1])-input_zero)*signed_byte(filter[1]);
        a2+=(static_cast<int>(input[2])-input_zero)*signed_byte(filter[2]);
        a3+=(static_cast<int>(input[3])-input_zero)*signed_byte(filter[3]);
      }
      a0=requantize(a0,mult[c+0],shift[c+0])+output_zero;
      a1=requantize(a1,mult[c+1],shift[c+1])+output_zero;
      a2=requantize(a2,mult[c+2],shift[c+2])+output_zero;
      a3=requantize(a3,mult[c+3],shift[c+3])+output_zero;
      output[c+0]=static_cast<std::int8_t>(std::clamp<std::int32_t>(a0,amin,amax));
      output[c+1]=static_cast<std::int8_t>(std::clamp<std::int32_t>(a1,amin,amax));
      output[c+2]=static_cast<std::int8_t>(std::clamp<std::int32_t>(a2,amin,amax));
      output[c+3]=static_cast<std::int8_t>(std::clamp<std::int32_t>(a3,amin,amax));
    }
  }
}
template<std::size_t IH,std::size_t IW,std::size_t C,std::size_t OH,std::size_t OW,typename Input>
void average_pool_nhwc(const Input& in,std::array<std::int8_t,OH*OW*C>& out,int filter_h,int filter_w,
           int stride_h,int stride_w,int pad_h,int pad_w,int amin,int amax) {
  for(std::size_t oy=0;oy<OH;++oy)for(std::size_t ox=0;ox<OW;++ox)for(std::size_t c=0;c<C;++c){
    std::int32_t sum=0;int count=0;
    for(int fy=0;fy<filter_h;++fy)for(int fx=0;fx<filter_w;++fx){
      const int iy=static_cast<int>(oy)*stride_h+fy-pad_h,ix=static_cast<int>(ox)*stride_w+fx-pad_w;
      if(iy<0||iy>=static_cast<int>(IH)||ix<0||ix>=static_cast<int>(IW))continue;
      sum+=in[(static_cast<std::size_t>(iy)*IW+static_cast<std::size_t>(ix))*C+c];++count;
    }
    const int rounded=sum>0?(sum+count/2)/count:(sum-count/2)/count;
    out[(oy*OW+ox)*C+c]=static_cast<std::int8_t>(std::clamp(rounded,amin,amax));
  }
}
template<typename Input,std::size_t W,std::size_t B,std::size_t O>
void depthwise_single_output(const Input& in,const std::array<std::uint8_t,W>& weights,
           const std::array<std::uint8_t,B>& bias,std::array<std::int8_t,O>& out,int input_zero,int output_zero,
           const std::array<std::int32_t,O>& mult,const std::array<int,O>& shift,int amin,int amax) {
  static_assert(B==O*4 && W%O==0); constexpr std::size_t Kernel=W/O;
  for(std::size_t channel=0;channel<O;++channel){std::int32_t acc=load_i32(bias.data()+channel*4);
    for(std::size_t kernel=0;kernel<Kernel;++kernel){const std::size_t i=kernel*O+channel;
      acc+=static_cast<std::int32_t>(signed_byte(weights[i]))*(static_cast<int>(in[i])-input_zero);}
    acc=requantize(acc,mult[channel],shift[channel])+output_zero;
    out[channel]=static_cast<std::int8_t>(std::clamp<std::int32_t>(acc,amin,amax));}
}
template<std::size_t Block,typename Input,std::size_t W,std::size_t B,std::size_t O>
void dense_masked_simd_blocks(const Input& in,const std::array<std::uint8_t,W>& weights,const std::array<std::uint8_t,B>& bias,
           std::array<std::int8_t,O>& out,int input_zero,int output_zero,const std::array<std::int32_t,O>& mult,
           const std::array<int,O>& shift,int amin,int amax) {
  constexpr std::size_t N=W/O, Blocks=(N+Block-1)/Block;static_assert(W==N*O && B==O*4);
  using BlockIndex=std::conditional_t<(Blocks<=256),std::uint8_t,std::uint16_t>;
  std::array<BlockIndex,Blocks> active{};std::size_t count=0;
  for(std::size_t block=0;block<Blocks;++block){const std::size_t begin=block*Block,end=std::min(begin+Block,N);bool nonzero=false;
    std::size_t i=begin;
#if defined(__x86_64__) || defined(__i386__)
    for(;i+16<=end;i+=16)nonzero|=any_nonzero_16(in.data()+i,static_cast<std::int8_t>(input_zero));
#endif
    for(;i<end;++i)nonzero|=static_cast<int>(in[i])!=input_zero;
    if(nonzero)active[count++]=static_cast<BlockIndex>(block);}
  for(std::size_t oc=0;oc<O;++oc){std::int32_t acc=load_i32(bias.data()+oc*4);
    for(std::size_t k=0;k<count;++k){const std::size_t begin=active[k]*Block,end=std::min(begin+Block,N);
      std::size_t i=begin;
#if defined(__x86_64__) || defined(__i386__)
      if(input_zero==-128)for(;i+16<=end;i+=16)acc+=dot_16_zero_minus_128(in.data()+i,weights.data()+oc*N+i);
#endif
      for(;i<end;++i)acc+=static_cast<std::int32_t>(signed_byte(weights[oc*N+i]))*(static_cast<int>(in[i])-input_zero);}
    acc=requantize(acc,mult[oc],shift[oc])+output_zero;out[oc]=static_cast<std::int8_t>(std::clamp<std::int32_t>(acc,amin,amax));}
}
template<typename T,std::size_t N> class ArrayView {
 public:
  explicit ArrayView(T* data):data_(data){}
  T* begin() const{return data_;} T* end() const{return data_+N;} T* data() const{return data_;}
  constexpr std::size_t size() const{return N;} T& operator[](std::size_t i) const{return data_[i];}
 private: T* data_;
};
""")
        if any(op["opcode"] == "PAD" for op in ops):
            c.write("""
template<std::size_t Rank,std::size_t N,std::size_t O,typename Input>
void pad_tensor(const Input& in,std::array<std::int8_t,O>& out,
                const std::array<std::size_t,Rank>& input_shape,
                const std::array<std::size_t,Rank>& output_shape,
                const std::array<std::size_t,Rank>& before,std::int8_t zero) {
  out.fill(zero);
  for(std::size_t source=0;source<N;++source){
    std::size_t remainder=source,destination=0,stride=1;
    for(std::size_t reverse=0;reverse<Rank;++reverse){
      const std::size_t dimension=Rank-1-reverse;
      const std::size_t coordinate=remainder%input_shape[dimension];
      remainder/=input_shape[dimension];
      destination+=(coordinate+before[dimension])*stride;
      stride*=output_shape[dimension];
    }
    out[destination]=in[source];
  }
}
""")
        if any(op["opcode"] == "MEAN" for op in ops):
            c.write("""
template<std::size_t Rank,std::size_t N,std::size_t O,typename Input>
void mean_tensor(const Input& in,std::array<std::int8_t,O>& out,
                 const std::array<std::size_t,Rank>& input_shape,
                 const std::array<bool,Rank>& reduced,std::size_t count,
                 int input_zero,int output_zero,std::int32_t multiplier,int shift) {
  std::array<std::int32_t,O> sums{};
  for(std::size_t source=0;source<N;++source){
    std::size_t remainder=source,destination=0,stride=1;
    for(std::size_t reverse=0;reverse<Rank;++reverse){
      const std::size_t dimension=Rank-1-reverse;
      const std::size_t coordinate=remainder%input_shape[dimension];
      remainder/=input_shape[dimension];
      if(!reduced[dimension]){destination+=coordinate*stride;stride*=input_shape[dimension];}
    }
    sums[destination]+=in[source];
  }
  for(std::size_t i=0;i<O;++i){
    const std::int32_t centered=sums[i]-input_zero*static_cast<std::int32_t>(count);
    const std::int32_t value=requantize(centered,multiplier,shift)+output_zero;
    out[i]=static_cast<std::int8_t>(std::clamp<std::int32_t>(value,-128,127));
  }
}
""")
        for line in globals_: c.write(line+"\n")
        c.write("}\n\n")
        c.write(f"void {class_name}::reset(){{\n")
        for name in state_sizes:
            c.write(f"  {state_field[name]}.fill(static_cast<std::int8_t>({state_zero_points[name]}));\n")
        c.write("}\n")
        c.write(f"\nstd::size_t {class_name}::scratch_bytes(){{return {program.scratch_bytes};}}\n")
        c.write(f"\nNN2PROG_ESP32_IRAM {return_type} {class_name}::invoke(const std::array<std::int8_t,{input_elements}>& input){{\n")
        c.write(f"  alignas(8) std::array<std::uint8_t,{program.scratch_bytes}> scratch_arena;\n")
        for op in ops:
            code=op["opcode"]; ins=op["inputs"]; outs=op["outputs"]
            if code in ("CALL_ONCE","VAR_HANDLE"): continue
            for index in outs:
                allocation = program.allocation(index)
                if allocation is None: continue
                typ = "std::uint8_t" if tensors[index]["type"] == "UINT8" else "std::int8_t"
                c.write(f"  auto& t{index}=*::new (static_cast<void*>(scratch_arena.data()+{allocation.offset})) std::array<{typ},{elements(tensors[index])}>;\n")
            c.write(f"  // op {op['index']}: {code}\n")
            kernel = program.kernel(op["index"])
            if code == "RESHAPE":
                c.write(f"  const auto& t{outs[0]}={ptr(ins[0])};\n")
            elif code == "READ_VARIABLE":
                field=state_field[handles[ins[0]]]
                c.write(f"  const auto& t{outs[0]}={field};\n")
            elif code == "STRIDED_SLICE":
                n=elements(tensors[outs[0]]); ni=elements(tensors[ins[0]])
                c.write(f"  ArrayView<const std::int8_t,{n}> t{outs[0]}({ptr(ins[0])}.data()+{ni-n});\n")
            elif code == "SPLIT_V":
                split = op["split_v"]
                offset = 0
                for output, size in zip(outs, split["sizes"]):
                    chunk = size * split["inner"]
                    c.write(f"  for(std::size_t outer=0;outer<{split['outer']};++outer) "
                            f"std::copy_n({ptr(ins[0])}.data()+outer*{split['axis_size'] * split['inner']}+{offset * split['inner']},"
                            f"{chunk},t{output}.data()+outer*{chunk});\n")
                    offset += size
            elif code == "PAD":
                inp=ins[0];o=outs[0];shape=tensors[inp]["shape"];oshape=tensors[o]["shape"]
                rank=len(shape);before=op["pad"]["before"];zero=tensors[inp]["quantization"]["zero_point"][0]
                c.write(f"  pad_tensor<{rank},{elements(tensors[inp])},{elements(tensors[o])}>({ptr(inp)},t{o},"
                        f"std::array<std::size_t,{rank}>{{{','.join(map(str,shape))}}},"
                        f"std::array<std::size_t,{rank}>{{{','.join(map(str,oshape))}}},"
                        f"std::array<std::size_t,{rank}>{{{','.join(map(str,before))}}},static_cast<std::int8_t>({zero}));\n")
            elif code == "MEAN":
                inp=ins[0];o=outs[0];shape=tensors[inp]["shape"];rank=len(shape);mean=op["mean"]
                reduced=["true" if index in mean["axes"] else "false" for index in range(rank)]
                iq=tensors[inp]["quantization"];oq=tensors[o]["quantization"]
                c.write(f"  mean_tensor<{rank},{elements(tensors[inp])},{elements(tensors[o])}>({ptr(inp)},t{o},"
                        f"std::array<std::size_t,{rank}>{{{','.join(map(str,shape))}}},"
                        f"std::array<bool,{rank}>{{{','.join(reduced)}}},{mean['count']},"
                        f"{iq['zero_point'][0]},{oq['zero_point'][0]},{mean['multiplier']},{mean['shift']});\n")
            elif code == "CONCATENATION":
                offset=0
                for x in ins:
                    n=elements(tensors[x]); c.write(f"  std::copy_n({ptr(x)}.begin(),{n},t{outs[0]}.begin()+{offset});\n"); offset+=n
            elif code == "ASSIGN_VARIABLE":
                field=state_field[handles[ins[0]]]
                c.write(f"  std::copy_n({ptr(ins[1])}.begin(),{ptr(ins[1])}.size(),{field}.begin());\n")
            elif code in ("CONV_2D","FULLY_CONNECTED"):
                inp,w,b=ins[:3]; o=outs[0]; iq=tensors[inp]["quantization"]; oq=tensors[o]["quantization"]
                lo=op["lower"]
                spatial_conv = code == "CONV_2D" and elements(tensors[w]) != elements(tensors[inp]) * tensors[w]["shape"][0]
                if spatial_conv:
                    ih,iw,ic=nhwc(inp);oh,ow,oc=nhwc(o);_oc,fh,fw,wic=tensors[w]["shape"]
                    if (_oc, wic) != (oc, ic): raise ValueError(f"CONV_2D shape mismatch at op {op['index']}")
                    opts=op["options"];sh=opts["stride_h"];sw=opts["stride_w"];dh=opts["dilation_h"];dw=opts["dilation_w"]
                    ph=padding_before(ih,oh,fh,sh,dh,opts["padding"]);pw=padding_before(iw,ow,fw,sw,dw,opts["padding"])
                    wz=tensors[w]["quantization"]["zero_point"][0]
                    if kernel == "esp32_conv_1x1_unrolled8_bias_fold":
                        c.write(f"  esp32_conv_1x1<{ih},{iw},{ic},{oc},{oh},{ow}>({ptr(inp)},{raw(w)},op{op['index']}_esp32_bias,t{o},{sh},{sw},{oq['zero_point'][0]},op{op['index']}_mult,op{op['index']}_shift,{lo['amin']},{lo['amax']});\n")
                    elif kernel == "esp32_conv_nhwc_unrolled8":
                        c.write(f"  esp32_conv_nhwc<{ih},{iw},{ic},{fh},{fw},{oc},{oh},{ow}>({ptr(inp)},{raw(w)},{raw(b)},t{o},{sh},{sw},{ph},{pw},{iq['zero_point'][0]},{oq['zero_point'][0]},op{op['index']}_mult,op{op['index']}_shift,{lo['amin']},{lo['amax']});\n")
                    else:
                        c.write(f"  conv2d_nhwc<{ih},{iw},{ic},{fh},{fw},{oc},{oh},{ow}>({ptr(inp)},{raw(w)},{raw(b)},t{o},{sh},{sw},{dh},{dw},{ph},{pw},{iq['zero_point'][0]},{wz},{oq['zero_point'][0]},op{op['index']}_mult,op{op['index']}_shift,{lo['amin']},{lo['amax']});\n")
                elif kernel == "x86_dense_masked_simd32":
                    c.write(f"  dense_masked_simd_blocks<{masked_simd_block}>({ptr(inp)},{raw(w)},{raw(b)},t{o},{iq['zero_point'][0]},{oq['zero_point'][0]},op{op['index']}_mult,op{op['index']}_shift,{lo['amin']},{lo['amax']});\n")
                else: c.write(f"  dense({ptr(inp)},{raw(w)},{raw(b)},t{o},{iq['zero_point'][0]},{oq['zero_point'][0]},op{op['index']}_mult,op{op['index']}_shift,{lo['amin']},{lo['amax']});\n")
            elif code == "DEPTHWISE_CONV_2D":
                inp,w,b=ins[:3];o=outs[0];iq=tensors[inp]["quantization"];oq=tensors[o]["quantization"];lo=op["lower"]
                if (op["options"].get("padding") == "VALID" and op["options"].get("stride_w") == 1
                      and op["options"].get("stride_h") == 1 and op["options"].get("dilation_w") == 1
                      and op["options"].get("dilation_h") == 1 and op["options"].get("depth_multiplier") == 1
                      and elements(tensors[inp]) == elements(tensors[w])
                      and elements(tensors[o]) == tensors[o]["shape"][-1]):
                    c.write(f"  depthwise_single_output({ptr(inp)},{raw(w)},{raw(b)},t{o},{iq['zero_point'][0]},{oq['zero_point'][0]},op{op['index']}_mult,op{op['index']}_shift,{lo['amin']},{lo['amax']});\n")
                else:
                    ih,iw,ic=nhwc(inp);oh,ow,oc=nhwc(o);one,fh,fw,woc=tensors[w]["shape"]
                    dm=op["options"]["depth_multiplier"]
                    if one != 1 or woc != oc or oc != ic*dm: raise ValueError(f"DEPTHWISE_CONV_2D shape mismatch at op {op['index']}")
                    opts=op["options"];sh=opts["stride_h"];sw=opts["stride_w"];dh=opts["dilation_h"];dw=opts["dilation_w"]
                    ph=padding_before(ih,oh,fh,sh,dh,opts["padding"]);pw=padding_before(iw,ow,fw,sw,dw,opts["padding"])
                    wz=tensors[w]["quantization"]["zero_point"][0]
                    if kernel == "esp32_depthwise_channels4":
                        c.write(f"  esp32_depthwise_channels4<{ih},{iw},{ic},{fh},{fw},{oh},{ow}>({ptr(inp)},{raw(w)},{raw(b)},t{o},{sh},{sw},{ph},{pw},{iq['zero_point'][0]},{oq['zero_point'][0]},op{op['index']}_mult,op{op['index']}_shift,{lo['amin']},{lo['amax']});\n")
                    else:
                        c.write(f"  depthwise_nhwc<{ih},{iw},{ic},{fh},{fw},{dm},{oh},{ow}>({ptr(inp)},{raw(w)},{raw(b)},t{o},{sh},{sw},{dh},{dw},{ph},{pw},{iq['zero_point'][0]},{wz},{oq['zero_point'][0]},op{op['index']}_mult,op{op['index']}_shift,{lo['amin']},{lo['amax']});\n")
            elif code == "AVERAGE_POOL_2D":
                inp=ins[0];o=outs[0];ih,iw,channels=nhwc(inp);oh,ow,ochannels=nhwc(o)
                if channels != ochannels: raise ValueError(f"AVERAGE_POOL_2D channel mismatch at op {op['index']}")
                opts=op["options"];fh=opts["filter_h"];fw=opts["filter_w"];sh=opts["stride_h"];sw=opts["stride_w"]
                ph=padding_before(ih,oh,fh,sh,1,opts["padding"]);pw=padding_before(iw,ow,fw,sw,1,opts["padding"])
                lo=op["lower"]
                c.write(f"  average_pool_nhwc<{ih},{iw},{channels},{oh},{ow}>({ptr(inp)},t{o},{fh},{fw},{sh},{sw},{ph},{pw},{lo['amin']},{lo['amax']});\n")
            elif code == "MUL":
                a,b=ins;o=outs[0];aq,bq,oq=(tensors[x]["quantization"] for x in (a,b,o));lo=op["lower"]; bn=elements(tensors[b])
                c.write(f"  for(std::size_t i=0;i<t{o}.size();++i){{int av=static_cast<int>({ptr(a)}[i])-({aq['zero_point'][0]});int bv=static_cast<int>(signed_byte({raw(b)}[i%{bn}]))-({bq['zero_point'][0]});int v=requantize(av*bv,{lo['m']},{lo['s']})+({oq['zero_point'][0]});t{o}[i]=static_cast<std::int8_t>(std::clamp(v,{lo['amin']},{lo['amax']}));}}\n")
            elif code == "ADD":
                a,b=ins;o=outs[0];aq,bq,oq=(tensors[x]["quantization"] for x in (a,b,o));lo=op["lower"];bn=elements(tensors[b])
                bvalue=(f"signed_byte({raw(b)}[i%{bn}])" if tensors[b]["buffer_bytes"]
                        else f"{ptr(b)}[i%{bn}]")
                c.write(f"  for(std::size_t i=0;i<t{o}.size();++i){{int av=(static_cast<int>({ptr(a)}[i])-({aq['zero_point'][0]}))*(1<<{lo['left']});int bv=(static_cast<int>({bvalue})-({bq['zero_point'][0]}))*(1<<{lo['left']});int x=requantize(av,{lo['m1']},{lo['s1']})+requantize(bv,{lo['m2']},{lo['s2']});int v=requantize(x,{lo['mo']},{lo['so']})+({oq['zero_point'][0]});t{o}[i]=static_cast<std::int8_t>(std::clamp(v,{lo['amin']},{lo['amax']}));}}\n")
            elif code == "LOGISTIC":
                c.write(f"  t{outs[0]}[0]=logistic_lut[static_cast<std::uint8_t>(static_cast<int>({ptr(ins[0])}[0])+128)];\n")
            elif code == "QUANTIZE":
                c.write(f"  t{outs[0]}[0]=static_cast<std::uint8_t>(static_cast<int>({ptr(ins[0])}[0])+128);\n")
            elif code == "SOFTMAX" and argmax_output:
                c.write("  // Class decision is argmax(logits); softmax is monotone and unnecessary for top-1.\n")
            else: raise ValueError(f"unsupported runtime op {code}")
        if argmax_output:
            count=elements(tensors[decision_tensor])
            c.write(f"  std::uint8_t decision=0;for(std::size_t i=1;i<{count};++i)if(t{decision_tensor}[i]>t{decision_tensor}[decision])decision=static_cast<std::uint8_t>(i);\n")
            c.write("  return decision;\n}\n}\n")
        elif output_elements == 1:
            c.write(f"  return t{output_tensor}[0];\n}}\n}}\n")
        else:
            c.write(f"  std::array<std::int8_t,{output_elements}> result{{}};"
                    f"std::copy_n({ptr(output_tensor)}.begin(),{output_elements},result.begin());return result;\n}}\n}}\n")
    print(f"generated {header} and {source}")
