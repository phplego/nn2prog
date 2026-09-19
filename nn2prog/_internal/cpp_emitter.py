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
        h.write("  static std::size_t working_memory_bytes();\n")
        h.write(""" private:
""")
        for name, size in state_sizes.items():
            h.write(f"  std::array<std::int8_t,{size}> {state_field[name]}{{}};\n")
        h.write("};\n}\n")

    def local(index): return f"tensor_{index}"
    def ptr(index):
        if index == input_tensor: return "input"
        slot = program.storage_slot(index)
        if slot is not None:
            storage = program.storage_slots[slot]
            count = elements(tensors[index])
            if storage.elements == count:
                return f"buffer_{slot}"
            typ = "std::uint8_t" if tensors[index]["type"] == "UINT8" else "std::int8_t"
            return f"ArrayView<const {typ},{count}>(buffer_{slot}.data())"
        return local(index)
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
        if target in ("esp32", "esp32s3"):
            c.write("#define NN2PROG_GENERATED_ESP32 1\n")
        if target == "esp32s3":
            c.write('#if defined(ESP_PLATFORM)\n#include "sdkconfig.h"\n#endif\n')
            c.write('#include <utility>\n')
        c.write("""#include "model.h"
#include "model.constants.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
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
""")
        if target == "esp32s3":
            c.write("  return static_cast<std::int32_t>((product+(std::int64_t{1}<<30))>>31);\n")
        else:
            c.write("""  const std::int64_t nudge=product>=0 ? (std::int64_t{1}<<30) : (1-(std::int64_t{1}<<30));
  return static_cast<std::int32_t>((product+nudge)/(std::int64_t{1}<<31));
""")
        c.write("""}
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
template<std::size_t W,std::size_t B,std::size_t O,typename Input,typename Output>
void dense(const Input& in,const std::array<std::uint8_t,W>& weights,const std::array<std::uint8_t,B>& bias,
           Output& out,int input_zero,int output_zero,const std::array<std::int32_t,O>& mult,const std::array<int,O>& shift,int amin,int amax) {
  static_assert(W%O==0 && B==O*4);constexpr std::size_t N=W/O;
  for(std::size_t oc=0;oc<O;++oc){ std::int32_t acc=load_i32(bias.data()+oc*4);
    for(std::size_t i=0;i<N;++i) acc += static_cast<std::int32_t>(signed_byte(weights[oc*N+i]))*(static_cast<int>(in[i])-input_zero);
    acc=requantize(acc,mult[oc],shift[oc])+output_zero; out[oc]=static_cast<std::int8_t>(std::clamp<std::int32_t>(acc,amin,amax)); }
}
""")
        if target == "esp32s3":
            c.write("""// Both pointers must be 16-byte aligned; N is padded with zero weights.
template<std::size_t N>
inline std::int32_t s3_dot16(const std::int8_t* input,const std::int8_t* weights) {
  static_assert(N>0 && N%16==0);
#if defined(__XTENSA__) && defined(CONFIG_IDF_TARGET_ESP32S3)
  std::int32_t result;
  if constexpr(N<=64) {
    __asm__ volatile(
      "ee.zero.accx\\n"
      ".rept %[count]\\n"
      "ee.vld.128.ip q0, %[x], 16\\n"
      "ee.vld.128.ip q1, %[w], 16\\n"
      "ee.vmulas.s8.accx q0, q1\\n"
      ".endr\\n"
      "nop\\n"
      "nop\\n"
      "rur.accx_0 %[r]\\n"
      : [x] "+&r"(input), [w] "+&r"(weights), [r] "=r"(result)
      : [count] "i"(N/16) : "memory");
  } else {
    unsigned blocks=N/16;
    __asm__ volatile(
      "ee.zero.accx\\n"
      "1: ee.vld.128.ip q0, %[x], 16\\n"
      "ee.vld.128.ip q1, %[w], 16\\n"
      "ee.vmulas.s8.accx q0, q1\\n"
      "addi %[n], %[n], -1\\n"
      "bnez %[n], 1b\\n"
      "nop\\n"
      "nop\\n"
      "rur.accx_0 %[r]\\n"
      : [x] "+&r"(input), [w] "+&r"(weights), [n] "+&r"(blocks), [r] "=r"(result)
      : : "memory");
  }
  return result;
#else
  std::int32_t result=0;
  for(std::size_t i=0;i<N;++i)result+=static_cast<std::int32_t>(input[i])*weights[i];
  return result;
#endif
}
// S8 QACC stores sixteen signed 20-bit lane sums in two 160-bit halves.
// The compiler proves that no lane overflows before selecting this kernel.
template<std::size_t C>
inline void s3_depthwise_sums(const std::int8_t* input,const std::int8_t* weights,std::uint8_t* raw) {
#if defined(__XTENSA__) && defined(CONFIG_IDF_TARGET_ESP32S3)
  unsigned taps=9;
  __asm__ volatile(
    "ee.zero.qacc\\n"
    "1: ee.vld.128.xp q0, %[x], %[stride]\\n"
    "ee.vld.128.xp q1, %[w], %[stride]\\n"
    "ee.vmulas.s8.qacc q0, q1\\n"
    "addi %[n], %[n], -1\\n"
    "bnez %[n], 1b\\n"
    "ee.st.qacc_l.l.128.ip %[s], 16\\n"
    "ee.st.qacc_l.h.32.ip %[s], 16\\n"
    "ee.st.qacc_h.l.128.ip %[s], 16\\n"
    "ee.st.qacc_h.h.32.ip %[s], 0\\n"
    : [x] "+&r"(input), [w] "+&r"(weights), [n] "+&r"(taps), [s] "+&r"(raw)
    : [stride] "r"(C) : "memory");
#else
  for(unsigned c=0;c<16;c+=2){
    std::int32_t a=0,b=0;
    for(unsigned tap=0;tap<9;++tap){a+=int(input[tap*C+c])*weights[tap*C+c];b+=int(input[tap*C+c+1])*weights[tap*C+c+1];}
    const auto u=static_cast<std::uint32_t>(a)&0xfffffu,v=static_cast<std::uint32_t>(b)&0xfffffu;
    auto* p=raw+(c/8)*32+((c%8)/2)*5;
    p[0]=u;p[1]=u>>8;p[2]=(u>>16)|(v<<4);p[3]=v>>4;p[4]=v>>12;
  }
#endif
}
inline std::uint32_t s3_load_word(const std::uint8_t* p) {
#if defined(__XTENSA__) && defined(CONFIG_IDF_TARGET_ESP32S3)
  // QACC storage and every requested word offset are four-byte aligned.
  std::uint32_t value;
  __asm__("l32i %0, %1, 0" : "=r"(value) : "r"(p) : "memory");
  return value;
#else
  return static_cast<std::uint32_t>(load_i32(p));
#endif
}
template<unsigned Lane>
inline std::int32_t s3_lane20_fixed(const std::uint8_t* raw) {
  static_assert(Lane<16);
  constexpr unsigned bit=(Lane%8)*20, word=bit/32, shift=bit%32;
  const auto* p=raw+(Lane/8)*32+word*4;
  const auto lo=s3_load_word(p);
  auto u=lo>>shift;
  if constexpr(shift>12)u|=s3_load_word(p+4)<<(32-shift);
  return static_cast<std::int32_t>(u<<12)>>12;
}
template<std::size_t C>
constexpr bool s3_right_shift_only(const std::array<std::int32_t,C>& mult,const std::array<int,C>& shift) {
  for(std::size_t c=0;c<C;++c)if(mult[c]<0 || shift[c]>=0 || shift[c]<-31)return false;
  return true;
}
template<bool RightShiftOnly>
inline std::int32_t s3_output_scale(std::int32_t x,std::int32_t mult,int shift) {
  if constexpr(!RightShiftOnly)return requantize(x,mult,shift);
  else {
    // Nonnegative multiplier excludes INT32_MIN * INT32_MIN; no left shift.
    const auto high=static_cast<std::int32_t>((std::int64_t(x)*mult+(std::int64_t{1}<<30))>>31);
    const unsigned right=-shift;
    const std::uint32_t mask=(std::uint32_t{1}<<right)-1;
    return (high>>right)+((static_cast<std::uint32_t>(high)&mask)>((mask>>1)+(high<0)));
  }
}
template<bool RightShiftOnly=false,std::size_t... Lane>
NN2PROG_ESP32_IRAM inline void s3_output16(const std::uint8_t* raw,const std::int32_t* bias,
    const std::int32_t* mult,const int* shift,std::int8_t* out,int oz,int amin,int amax,
    std::index_sequence<Lane...>) {
  ((out[Lane]=static_cast<std::int8_t>(std::clamp<std::int32_t>(
      s3_output_scale<RightShiftOnly>(bias[Lane]+s3_lane20_fixed<Lane>(raw),mult[Lane],shift[Lane])+oz,amin,amax))),...);
}
// Weights: [input channel][16 output lanes]; the compiler bounds every prefix sum.
template<std::size_t N>
inline void s3_pointwise_sums(const std::int8_t* input,const std::int8_t* weights,std::uint8_t* raw) {
  static_assert(N>0 && N%16==0);
#if defined(__XTENSA__) && defined(CONFIG_IDF_TARGET_ESP32S3)
  unsigned blocks=N/16;
  __asm__ volatile(
    "ee.zero.qacc\\n"
    "1: ee.vld.128.ip q1, %[x], 16\\n"
    "ee.vld.128.ip q0, %[w], 16\\n"
    "ee.vsmulas.s8.qacc.ld.incp q0, %[w], q0, q1, 0\\n"
    "ee.vsmulas.s8.qacc.ld.incp q0, %[w], q0, q1, 1\\n"
    "ee.vsmulas.s8.qacc.ld.incp q0, %[w], q0, q1, 2\\n"
    "ee.vsmulas.s8.qacc.ld.incp q0, %[w], q0, q1, 3\\n"
    "ee.vsmulas.s8.qacc.ld.incp q0, %[w], q0, q1, 4\\n"
    "ee.vsmulas.s8.qacc.ld.incp q0, %[w], q0, q1, 5\\n"
    "ee.vsmulas.s8.qacc.ld.incp q0, %[w], q0, q1, 6\\n"
    "ee.vsmulas.s8.qacc.ld.incp q0, %[w], q0, q1, 7\\n"
    "ee.vsmulas.s8.qacc.ld.incp q0, %[w], q0, q1, 8\\n"
    "ee.vsmulas.s8.qacc.ld.incp q0, %[w], q0, q1, 9\\n"
    "ee.vsmulas.s8.qacc.ld.incp q0, %[w], q0, q1, 10\\n"
    "ee.vsmulas.s8.qacc.ld.incp q0, %[w], q0, q1, 11\\n"
    "ee.vsmulas.s8.qacc.ld.incp q0, %[w], q0, q1, 12\\n"
    "ee.vsmulas.s8.qacc.ld.incp q0, %[w], q0, q1, 13\\n"
    "ee.vsmulas.s8.qacc.ld.incp q0, %[w], q0, q1, 14\\n"
    "ee.vsmulas.s8.qacc q0, q1, 15\\n"
    "addi %[n], %[n], -1\\n"
    "bnez %[n], 1b\\n"
    "ee.st.qacc_l.l.128.ip %[s], 16\\n"
    "ee.st.qacc_l.h.32.ip %[s], 16\\n"
    "ee.st.qacc_h.l.128.ip %[s], 16\\n"
    "ee.st.qacc_h.h.32.ip %[s], 0\\n"
    : [x] "+&r"(input), [w] "+&r"(weights), [n] "+&r"(blocks), [s] "+&r"(raw)
    : : "memory");
#else
  for(unsigned c=0;c<16;c+=2){
    std::int32_t a=0,b=0;
    for(std::size_t i=0;i<N;++i){a+=int(input[i])*weights[i*16+c];b+=int(input[i])*weights[i*16+c+1];}
    const auto u=static_cast<std::uint32_t>(a)&0xfffffu,v=static_cast<std::uint32_t>(b)&0xfffffu;
    auto* p=raw+(c/8)*32+((c%8)/2)*5;
    p[0]=u;p[1]=u>>8;p[2]=(u>>16)|(v<<4);p[3]=v>>4;p[4]=v>>12;
  }
#endif
}
template<std::size_t IH,std::size_t IW,std::size_t IC,std::size_t OC,std::size_t OH,std::size_t OW,bool RightShiftOnly=false,
         typename Input,std::size_t W,typename Output>
NN2PROG_ESP32_IRAM void esp32s3_conv_1x1_qacc(const Input& in,const std::array<std::int8_t,W>& weights,
           const std::array<std::int32_t,OC>& bias,Output& out,
           int sh,int sw,int oz,const std::array<std::int32_t,OC>& mult,
           const std::array<int,OC>& shift,int amin,int amax) {
  constexpr std::size_t Padded=(IC+15)/16*16;
  static_assert(OC%16==0 && W==OC*Padded);
  alignas(16) std::array<std::int8_t,Padded> input{};
  alignas(16) std::array<std::uint8_t,64> raw;
  for(std::size_t oy=0;oy<OH;++oy)for(std::size_t ox=0;ox<OW;++ox){
    std::copy_n(in.data()+(oy*sh*IW+ox*sw)*IC,IC,input.data());
    for(std::size_t oc=0;oc<OC;oc+=16){
      s3_pointwise_sums<Padded>(input.data(),weights.data()+oc*Padded,raw.data());
      s3_output16<RightShiftOnly>(raw.data(),bias.data()+oc,mult.data()+oc,shift.data()+oc,
          out.data()+(oy*OW+ox)*OC+oc,oz,amin,amax,std::make_index_sequence<16>{});
    }
  }
}
template<std::size_t IH,std::size_t IW,std::size_t C,std::size_t OH,std::size_t OW,bool RightShiftOnly=false,
         typename Input,typename Output>
NN2PROG_ESP32_IRAM void esp32s3_depthwise_3x3(const Input& in,const std::array<std::int8_t,9*C>& weights,
           const std::array<std::int32_t,C>& bias,Output& out,int sh,int sw,int ph,int pw,int iz,int oz,
           const std::array<std::int32_t,C>& mult,const std::array<int,C>& shift,int amin,int amax) {
  static_assert(C%16==0);
  alignas(16) std::array<std::int8_t,9*C> window;
  alignas(16) std::array<std::uint8_t,64> raw;
  for(std::size_t oy=0;oy<OH;++oy)for(std::size_t ox=0;ox<OW;++ox){
    for(int fy=0;fy<3;++fy)for(int fx=0;fx<3;++fx){
      const int iy=static_cast<int>(oy)*sh+fy-ph,ix=static_cast<int>(ox)*sw+fx-pw;
      auto* pixel=window.data()+(fy*3+fx)*C;
      if(iy>=0 && ix>=0 && iy<static_cast<int>(IH) && ix<static_cast<int>(IW))
        std::copy_n(in.data()+(iy*IW+ix)*C,C,pixel);
      else std::fill_n(pixel,C,static_cast<std::int8_t>(iz));
    }
    for(std::size_t c=0;c<C;c+=16){
      s3_depthwise_sums<C>(window.data()+c,weights.data()+c,raw.data());
      s3_output16<RightShiftOnly>(raw.data(),bias.data()+c,mult.data()+c,shift.data()+c,
          out.data()+(oy*OW+ox)*C+c,oz,amin,amax,std::make_index_sequence<16>{});
    }
  }
}
template<std::size_t IH,std::size_t IW,std::size_t IC,std::size_t FH,std::size_t FW,std::size_t OC,
         std::size_t OH,std::size_t OW,bool RightShiftOnly=false,typename Input,std::size_t W,typename Output>
NN2PROG_ESP32_IRAM void esp32s3_conv_im2col(const Input& in,const std::array<std::int8_t,W>& weights,
           const std::array<std::int32_t,OC>& adjusted_bias,Output& out,
           int stride_h,int stride_w,int pad_h,int pad_w,int input_zero,int output_zero,
           const std::array<std::int32_t,OC>& mult,const std::array<int,OC>& shift,int amin,int amax) {
  constexpr std::size_t Padded=(FH*FW*IC+15)/16*16;
  static_assert(W==OC*Padded);
  alignas(16) std::array<std::int8_t,Padded> window{};
  for(std::size_t oy=0;oy<OH;++oy)for(std::size_t ox=0;ox<OW;++ox){
    const int y=static_cast<int>(oy)*stride_h-pad_h,x=static_cast<int>(ox)*stride_w-pad_w;
    if(y>=0 && x>=0 && y+static_cast<int>(FH)<=static_cast<int>(IH) && x+static_cast<int>(FW)<=static_cast<int>(IW)){
      for(std::size_t fy=0;fy<FH;++fy)
        std::copy_n(in.data()+((y+fy)*IW+x)*IC,FW*IC,window.data()+fy*FW*IC);
    }else{
      for(std::size_t fy=0;fy<FH;++fy)for(std::size_t fx=0;fx<FW;++fx){
        const int iy=y+static_cast<int>(fy),ix=x+static_cast<int>(fx);
        auto* pixel=window.data()+(fy*FW+fx)*IC;
        if(iy>=0 && ix>=0 && iy<static_cast<int>(IH) && ix<static_cast<int>(IW))
          std::copy_n(in.data()+(iy*IW+ix)*IC,IC,pixel);
        else std::fill_n(pixel,IC,static_cast<std::int8_t>(input_zero));
      }
    }
    for(std::size_t oc=0;oc<OC;++oc){
      std::int32_t acc=adjusted_bias[oc]+s3_dot16<Padded>(window.data(),weights.data()+oc*Padded);
      acc=s3_output_scale<RightShiftOnly>(acc,mult[oc],shift[oc])+output_zero;
      out[(oy*OW+ox)*OC+oc]=static_cast<std::int8_t>(std::clamp<std::int32_t>(acc,amin,amax));
    }
  }
}
template<std::size_t IH,std::size_t IW,std::size_t IC,std::size_t OC,std::size_t OH,std::size_t OW,bool RightShiftOnly=false,
         typename Input,std::size_t W,typename Output>
NN2PROG_ESP32_IRAM void esp32s3_conv_1x1(const Input& in,const std::array<std::int8_t,W>& weights,
           const std::array<std::int32_t,OC>& adjusted_bias,Output& out,
           int stride_h,int stride_w,int output_zero,const std::array<std::int32_t,OC>& mult,
           const std::array<int,OC>& shift,int amin,int amax) {
  constexpr std::size_t Padded=(IC+15)/16*16;
  static_assert(W==OC*Padded);
  alignas(16) std::array<std::int8_t,Padded> aligned_input{};
  for(std::size_t oy=0;oy<OH;++oy)for(std::size_t ox=0;ox<OW;++ox){
    std::copy_n(in.data()+(oy*stride_h*IW+ox*stride_w)*IC,IC,aligned_input.data());
    for(std::size_t oc=0;oc<OC;++oc){
      std::int32_t acc=adjusted_bias[oc]+s3_dot16<Padded>(aligned_input.data(),weights.data()+oc*Padded);
      acc=s3_output_scale<RightShiftOnly>(acc,mult[oc],shift[oc])+output_zero;
      out[(oy*OW+ox)*OC+oc]=static_cast<std::int8_t>(std::clamp<std::int32_t>(acc,amin,amax));
    }
  }
}
""")
        c.write("""template<std::size_t IH,std::size_t IW,std::size_t IC,std::size_t FH,std::size_t FW,std::size_t OC,
         std::size_t OH,std::size_t OW,typename Input,std::size_t W,std::size_t B,typename Output>
void conv2d_nhwc(const Input& in,const std::array<std::uint8_t,W>& weights,const std::array<std::uint8_t,B>& bias,
           Output& out,int stride_h,int stride_w,int dilation_h,int dilation_w,
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
         typename Input,std::size_t W,typename Output>
NN2PROG_ESP32_IRAM void esp32_conv_1x1(const Input& in,const std::array<std::uint8_t,W>& weights,
           const std::array<std::int32_t,OC>& adjusted_bias,Output& out,
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
         std::size_t OH,std::size_t OW,typename Input,std::size_t W,std::size_t B,typename Output>
NN2PROG_ESP32_IRAM void esp32_conv_nhwc(const Input& in,const std::array<std::uint8_t,W>& weights,
           const std::array<std::uint8_t,B>& bias,Output& out,
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
         std::size_t OH,std::size_t OW,typename Input,std::size_t W,std::size_t B,typename Output>
void depthwise_nhwc(const Input& in,const std::array<std::uint8_t,W>& weights,const std::array<std::uint8_t,B>& bias,
           Output& out,int stride_h,int stride_w,int dilation_h,int dilation_w,
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
         std::size_t OH,std::size_t OW,typename Input,std::size_t W,std::size_t B,typename Output>
NN2PROG_ESP32_IRAM void esp32_depthwise_channels4(const Input& in,const std::array<std::uint8_t,W>& weights,
           const std::array<std::uint8_t,B>& bias,Output& out,
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
template<std::size_t IH,std::size_t IW,std::size_t C,std::size_t OH,std::size_t OW,typename Input,typename Output>
void average_pool_nhwc(const Input& in,Output& out,int filter_h,int filter_w,
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
template<typename Input,std::size_t W,std::size_t B,std::size_t O,typename Output>
void depthwise_single_output(const Input& in,const std::array<std::uint8_t,W>& weights,
           const std::array<std::uint8_t,B>& bias,Output& out,int input_zero,int output_zero,
           const std::array<std::int32_t,O>& mult,const std::array<int,O>& shift,int amin,int amax) {
  static_assert(B==O*4 && W%O==0); constexpr std::size_t Kernel=W/O;
  for(std::size_t channel=0;channel<O;++channel){std::int32_t acc=load_i32(bias.data()+channel*4);
    for(std::size_t kernel=0;kernel<Kernel;++kernel){const std::size_t i=kernel*O+channel;
      acc+=static_cast<std::int32_t>(signed_byte(weights[i]))*(static_cast<int>(in[i])-input_zero);}
    acc=requantize(acc,mult[channel],shift[channel])+output_zero;
    out[channel]=static_cast<std::int8_t>(std::clamp<std::int32_t>(acc,amin,amax));}
}
template<std::size_t Block,typename Input,std::size_t W,std::size_t B,std::size_t O,typename Output>
void dense_masked_simd_blocks(const Input& in,const std::array<std::uint8_t,W>& weights,const std::array<std::uint8_t,B>& bias,
           Output& out,int input_zero,int output_zero,const std::array<std::int32_t,O>& mult,
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
  void fill(std::remove_const_t<T> value) const{std::fill(begin(),end(),value);}
 private: T* data_;
};
""")
        if any(op["opcode"] == "PAD" for op in ops):
            c.write("""
template<std::size_t Rank,std::size_t N,std::size_t O,typename Input,typename Output>
void pad_tensor(const Input& in,Output& out,
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
template<std::size_t Rank,std::size_t N,std::size_t O,typename Input,typename Output>
void mean_tensor(const Input& in,Output& out,
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
        c.write(f"\nstd::size_t {class_name}::working_memory_bytes(){{return {program.working_memory_bytes};}}\n")
        c.write(f"\nNN2PROG_ESP32_IRAM {return_type} {class_name}::invoke(const std::array<std::int8_t,{input_elements}>& input){{\n")
        for slot in program.storage_slots:
            typ = "std::uint8_t" if slot.type == "UINT8" else "std::int8_t"
            c.write(f"  std::array<{typ},{slot.elements}> buffer_{slot.index};\n")
        for op in ops:
            code=op["opcode"]; ins=op["inputs"]; outs=op["outputs"]
            if code in ("CALL_ONCE","VAR_HANDLE"): continue
            stored_outputs = [index for index in outs if program.storage_slot(index) is not None]
            if stored_outputs:
                c.write("  {\n")
                for index in stored_outputs:
                    typ = "std::uint8_t" if tensors[index]["type"] == "UINT8" else "std::int8_t"
                    allocation = program.storage_allocation(index)
                    slot = program.storage_slots[allocation.slot]
                    if allocation.temporary:
                        c.write(f"    std::array<{typ},{elements(tensors[index])}> {local(index)};\n")
                    elif slot.elements == elements(tensors[index]):
                        c.write(f"    auto& {local(index)}=buffer_{allocation.slot};\n")
                    else:
                        c.write(f"    ArrayView<{typ},{elements(tensors[index])}> {local(index)}(buffer_{allocation.slot}.data());\n")
            indent = "    " if stored_outputs else "  "
            c.write(f"{indent}// op {op['index']}: {code}\n")
            kernel = program.kernel(op["index"])
            if code == "RESHAPE":
                c.write(f"{indent}const auto& {local(outs[0])}={ptr(ins[0])};\n")
            elif code == "READ_VARIABLE":
                field=state_field[handles[ins[0]]]
                c.write(f"{indent}const auto& {local(outs[0])}={field};\n")
            elif code == "STRIDED_SLICE":
                n=elements(tensors[outs[0]]); ni=elements(tensors[ins[0]])
                c.write(f"{indent}ArrayView<const std::int8_t,{n}> {local(outs[0])}({ptr(ins[0])}.data()+{ni-n});\n")
            elif code == "SPLIT_V":
                split = op["split_v"]
                offset = 0
                for output, size in zip(outs, split["sizes"]):
                    chunk = size * split["inner"]
                    c.write(f"{indent}for(std::size_t outer=0;outer<{split['outer']};++outer) "
                            f"std::copy_n({ptr(ins[0])}.data()+outer*{split['axis_size'] * split['inner']}+{offset * split['inner']},"
                            f"{chunk},{local(output)}.data()+outer*{chunk});\n")
                    offset += size
            elif code == "PAD":
                inp=ins[0];o=outs[0];shape=tensors[inp]["shape"];oshape=tensors[o]["shape"]
                rank=len(shape);before=op["pad"]["before"];zero=tensors[inp]["quantization"]["zero_point"][0]
                c.write(f"{indent}pad_tensor<{rank},{elements(tensors[inp])},{elements(tensors[o])}>({ptr(inp)},{local(o)},"
                        f"std::array<std::size_t,{rank}>{{{','.join(map(str,shape))}}},"
                        f"std::array<std::size_t,{rank}>{{{','.join(map(str,oshape))}}},"
                        f"std::array<std::size_t,{rank}>{{{','.join(map(str,before))}}},static_cast<std::int8_t>({zero}));\n")
            elif code == "MEAN":
                inp=ins[0];o=outs[0];shape=tensors[inp]["shape"];rank=len(shape);mean=op["mean"]
                reduced=["true" if index in mean["axes"] else "false" for index in range(rank)]
                iq=tensors[inp]["quantization"];oq=tensors[o]["quantization"]
                c.write(f"{indent}mean_tensor<{rank},{elements(tensors[inp])},{elements(tensors[o])}>({ptr(inp)},{local(o)},"
                        f"std::array<std::size_t,{rank}>{{{','.join(map(str,shape))}}},"
                        f"std::array<bool,{rank}>{{{','.join(reduced)}}},{mean['count']},"
                        f"{iq['zero_point'][0]},{oq['zero_point'][0]},{mean['multiplier']},{mean['shift']});\n")
            elif code == "CONCATENATION":
                offset=0
                for x in ins:
                    n=elements(tensors[x]); c.write(f"{indent}std::copy_n({ptr(x)}.begin(),{n},{local(outs[0])}.begin()+{offset});\n"); offset+=n
            elif code == "ASSIGN_VARIABLE":
                field=state_field[handles[ins[0]]]
                c.write(f"{indent}std::copy_n({ptr(ins[1])}.begin(),{ptr(ins[1])}.size(),{field}.begin());\n")
            elif code in ("CONV_2D","FULLY_CONNECTED"):
                inp,w,b=ins[:3]; o=outs[0]; iq=tensors[inp]["quantization"]; oq=tensors[o]["quantization"]
                lo=op["lower"]
                spatial_conv = code == "CONV_2D" and elements(tensors[w]) != elements(tensors[inp]) * tensors[w]["shape"][0]
                if spatial_conv or kernel in ("esp32s3_conv_1x1_dot16", "esp32s3_conv_1x1_qacc16", "esp32s3_conv_im2col_dot16"):
                    ih,iw,ic=nhwc(inp);oh,ow,oc=nhwc(o);_oc,fh,fw,wic=tensors[w]["shape"]
                    if (_oc, wic) != (oc, ic): raise ValueError(f"CONV_2D shape mismatch at op {op['index']}")
                    opts=op["options"];sh=opts["stride_h"];sw=opts["stride_w"];dh=opts["dilation_h"];dw=opts["dilation_w"]
                    ph=padding_before(ih,oh,fh,sh,dh,opts["padding"]);pw=padding_before(iw,ow,fw,sw,dw,opts["padding"])
                    wz=tensors[w]["quantization"]["zero_point"][0]
                    if kernel == "esp32s3_conv_im2col_dot16":
                        fast = f"s3_right_shift_only(op{op['index']}_mult,op{op['index']}_shift)"
                        c.write(f"{indent}esp32s3_conv_im2col<{ih},{iw},{ic},{fh},{fw},{oc},{oh},{ow},{fast}>({ptr(inp)},op{op['index']}_packed,op{op['index']}_esp32_bias,{local(o)},{sh},{sw},{ph},{pw},{iq['zero_point'][0]},{oq['zero_point'][0]},op{op['index']}_mult,op{op['index']}_shift,{lo['amin']},{lo['amax']});\n")
                    elif kernel in ("esp32s3_conv_1x1_dot16", "esp32s3_conv_1x1_qacc16"):
                        fast = f"s3_right_shift_only(op{op['index']}_mult,op{op['index']}_shift)"
                        fn = "esp32s3_conv_1x1_qacc" if kernel.endswith("qacc16") else "esp32s3_conv_1x1"
                        c.write(f"{indent}{fn}<{ih},{iw},{ic},{oc},{oh},{ow},{fast}>({ptr(inp)},op{op['index']}_packed,op{op['index']}_esp32_bias,{local(o)},{sh},{sw},{oq['zero_point'][0]},op{op['index']}_mult,op{op['index']}_shift,{lo['amin']},{lo['amax']});\n")
                    elif kernel == "esp32_conv_1x1_unrolled8_bias_fold":
                        c.write(f"{indent}esp32_conv_1x1<{ih},{iw},{ic},{oc},{oh},{ow}>({ptr(inp)},{raw(w)},op{op['index']}_esp32_bias,{local(o)},{sh},{sw},{oq['zero_point'][0]},op{op['index']}_mult,op{op['index']}_shift,{lo['amin']},{lo['amax']});\n")
                    elif kernel == "esp32_conv_nhwc_unrolled8":
                        c.write(f"{indent}esp32_conv_nhwc<{ih},{iw},{ic},{fh},{fw},{oc},{oh},{ow}>({ptr(inp)},{raw(w)},{raw(b)},{local(o)},{sh},{sw},{ph},{pw},{iq['zero_point'][0]},{oq['zero_point'][0]},op{op['index']}_mult,op{op['index']}_shift,{lo['amin']},{lo['amax']});\n")
                    else:
                        c.write(f"{indent}conv2d_nhwc<{ih},{iw},{ic},{fh},{fw},{oc},{oh},{ow}>({ptr(inp)},{raw(w)},{raw(b)},{local(o)},{sh},{sw},{dh},{dw},{ph},{pw},{iq['zero_point'][0]},{wz},{oq['zero_point'][0]},op{op['index']}_mult,op{op['index']}_shift,{lo['amin']},{lo['amax']});\n")
                elif kernel == "x86_dense_masked_simd32":
                    c.write(f"{indent}dense_masked_simd_blocks<{masked_simd_block}>({ptr(inp)},{raw(w)},{raw(b)},{local(o)},{iq['zero_point'][0]},{oq['zero_point'][0]},op{op['index']}_mult,op{op['index']}_shift,{lo['amin']},{lo['amax']});\n")
                else: c.write(f"{indent}dense({ptr(inp)},{raw(w)},{raw(b)},{local(o)},{iq['zero_point'][0]},{oq['zero_point'][0]},op{op['index']}_mult,op{op['index']}_shift,{lo['amin']},{lo['amax']});\n")
            elif code == "DEPTHWISE_CONV_2D":
                inp,w,b=ins[:3];o=outs[0];iq=tensors[inp]["quantization"];oq=tensors[o]["quantization"];lo=op["lower"]
                if (op["options"].get("padding") == "VALID" and op["options"].get("stride_w") == 1
                      and op["options"].get("stride_h") == 1 and op["options"].get("dilation_w") == 1
                      and op["options"].get("dilation_h") == 1 and op["options"].get("depth_multiplier") == 1
                      and elements(tensors[inp]) == elements(tensors[w])
                      and elements(tensors[o]) == tensors[o]["shape"][-1]):
                    c.write(f"{indent}depthwise_single_output({ptr(inp)},{raw(w)},{raw(b)},{local(o)},{iq['zero_point'][0]},{oq['zero_point'][0]},op{op['index']}_mult,op{op['index']}_shift,{lo['amin']},{lo['amax']});\n")
                else:
                    ih,iw,ic=nhwc(inp);oh,ow,oc=nhwc(o);one,fh,fw,woc=tensors[w]["shape"]
                    dm=op["options"]["depth_multiplier"]
                    if one != 1 or woc != oc or oc != ic*dm: raise ValueError(f"DEPTHWISE_CONV_2D shape mismatch at op {op['index']}")
                    opts=op["options"];sh=opts["stride_h"];sw=opts["stride_w"];dh=opts["dilation_h"];dw=opts["dilation_w"]
                    ph=padding_before(ih,oh,fh,sh,dh,opts["padding"]);pw=padding_before(iw,ow,fw,sw,dw,opts["padding"])
                    wz=tensors[w]["quantization"]["zero_point"][0]
                    if kernel == "esp32s3_depthwise_3x3_qacc16":
                        fast = f"s3_right_shift_only(op{op['index']}_mult,op{op['index']}_shift)"
                        c.write(f"{indent}esp32s3_depthwise_3x3<{ih},{iw},{ic},{oh},{ow},{fast}>({ptr(inp)},op{op['index']}_packed,op{op['index']}_esp32_bias,{local(o)},{sh},{sw},{ph},{pw},{iq['zero_point'][0]},{oq['zero_point'][0]},op{op['index']}_mult,op{op['index']}_shift,{lo['amin']},{lo['amax']});\n")
                    elif kernel == "esp32_depthwise_channels4":
                        c.write(f"{indent}esp32_depthwise_channels4<{ih},{iw},{ic},{fh},{fw},{oh},{ow}>({ptr(inp)},{raw(w)},{raw(b)},{local(o)},{sh},{sw},{ph},{pw},{iq['zero_point'][0]},{oq['zero_point'][0]},op{op['index']}_mult,op{op['index']}_shift,{lo['amin']},{lo['amax']});\n")
                    else:
                        c.write(f"{indent}depthwise_nhwc<{ih},{iw},{ic},{fh},{fw},{dm},{oh},{ow}>({ptr(inp)},{raw(w)},{raw(b)},{local(o)},{sh},{sw},{dh},{dw},{ph},{pw},{iq['zero_point'][0]},{wz},{oq['zero_point'][0]},op{op['index']}_mult,op{op['index']}_shift,{lo['amin']},{lo['amax']});\n")
            elif code == "AVERAGE_POOL_2D":
                inp=ins[0];o=outs[0];ih,iw,channels=nhwc(inp);oh,ow,ochannels=nhwc(o)
                if channels != ochannels: raise ValueError(f"AVERAGE_POOL_2D channel mismatch at op {op['index']}")
                opts=op["options"];fh=opts["filter_h"];fw=opts["filter_w"];sh=opts["stride_h"];sw=opts["stride_w"]
                ph=padding_before(ih,oh,fh,sh,1,opts["padding"]);pw=padding_before(iw,ow,fw,sw,1,opts["padding"])
                lo=op["lower"]
                c.write(f"{indent}average_pool_nhwc<{ih},{iw},{channels},{oh},{ow}>({ptr(inp)},{local(o)},{fh},{fw},{sh},{sw},{ph},{pw},{lo['amin']},{lo['amax']});\n")
            elif code == "MUL":
                a,b=ins;o=outs[0];aq,bq,oq=(tensors[x]["quantization"] for x in (a,b,o));lo=op["lower"]; bn=elements(tensors[b])
                c.write(f"{indent}for(std::size_t i=0;i<{local(o)}.size();++i){{int av=static_cast<int>({ptr(a)}[i])-({aq['zero_point'][0]});int bv=static_cast<int>(signed_byte({raw(b)}[i%{bn}]))-({bq['zero_point'][0]});int v=requantize(av*bv,{lo['m']},{lo['s']})+({oq['zero_point'][0]});{local(o)}[i]=static_cast<std::int8_t>(std::clamp(v,{lo['amin']},{lo['amax']}));}}\n")
            elif code == "ADD":
                a,b=ins;o=outs[0];aq,bq,oq=(tensors[x]["quantization"] for x in (a,b,o));lo=op["lower"];bn=elements(tensors[b])
                bvalue=(f"signed_byte({raw(b)}[i%{bn}])" if tensors[b]["buffer_bytes"]
                        else f"{ptr(b)}[i%{bn}]")
                c.write(f"{indent}for(std::size_t i=0;i<{local(o)}.size();++i){{int av=(static_cast<int>({ptr(a)}[i])-({aq['zero_point'][0]}))*(1<<{lo['left']});int bv=(static_cast<int>({bvalue})-({bq['zero_point'][0]}))*(1<<{lo['left']});int x=requantize(av,{lo['m1']},{lo['s1']})+requantize(bv,{lo['m2']},{lo['s2']});int v=requantize(x,{lo['mo']},{lo['so']})+({oq['zero_point'][0]});{local(o)}[i]=static_cast<std::int8_t>(std::clamp(v,{lo['amin']},{lo['amax']}));}}\n")
            elif code == "LOGISTIC":
                c.write(f"{indent}{local(outs[0])}[0]=logistic_lut[static_cast<std::uint8_t>(static_cast<int>({ptr(ins[0])}[0])+128)];\n")
            elif code == "QUANTIZE":
                c.write(f"{indent}{local(outs[0])}[0]=static_cast<std::uint8_t>(static_cast<int>({ptr(ins[0])}[0])+128);\n")
            elif code == "SOFTMAX" and argmax_output:
                c.write(f"{indent}// Class decision is argmax(logits); softmax is monotone and unnecessary for top-1.\n")
            else: raise ValueError(f"unsupported runtime op {code}")
            for index in stored_outputs:
                allocation = program.storage_allocation(index)
                if not allocation.temporary:
                    continue
                slot = allocation.slot
                if program.storage_slots[slot].elements == elements(tensors[index]):
                    c.write(f"    buffer_{slot}={local(index)};\n")
                else:
                    c.write(f"    std::copy_n({local(index)}.begin(),{local(index)}.size(),buffer_{slot}.begin());\n")
            if stored_outputs:
                c.write("  }\n")
        if argmax_output:
            count=elements(tensors[decision_tensor])
            decision = ptr(decision_tensor)
            c.write(f"  const auto& logits={decision};\n")
            c.write(f"  std::uint8_t result=0;for(std::size_t i=1;i<{count};++i)if(logits[i]>logits[result])result=static_cast<std::uint8_t>(i);\n")
            c.write("  return result;\n}\n}\n")
        elif output_elements == 1:
            c.write(f"  return {ptr(output_tensor)}[0];\n}}\n}}\n")
        else:
            c.write(f"  std::array<std::int8_t,{output_elements}> result{{}};"
                    f"std::copy_n({ptr(output_tensor)}.begin(),{output_elements},result.begin());return result;\n}}\n}}\n")
    print(f"generated {header} and {source}")
