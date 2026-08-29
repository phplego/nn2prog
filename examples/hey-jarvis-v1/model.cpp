#include "model.h"
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
template<std::size_t W,std::size_t B,std::size_t O,typename Input,typename Output>
void dense(const Input& in,const std::array<std::uint8_t,W>& weights,const std::array<std::uint8_t,B>& bias,
           Output& out,int input_zero,int output_zero,const std::array<std::int32_t,O>& mult,const std::array<int,O>& shift,int amin,int amax) {
  static_assert(W%O==0 && B==O*4);constexpr std::size_t N=W/O;
  for(std::size_t oc=0;oc<O;++oc){ std::int32_t acc=load_i32(bias.data()+oc*4);
    for(std::size_t i=0;i<N;++i) acc += static_cast<std::int32_t>(signed_byte(weights[oc*N+i]))*(static_cast<int>(in[i])-input_zero);
    acc=requantize(acc,mult[oc],shift[oc])+output_zero; out[oc]=static_cast<std::int8_t>(std::clamp<std::int32_t>(acc,amin,amax)); }
}
template<std::size_t IH,std::size_t IW,std::size_t IC,std::size_t FH,std::size_t FW,std::size_t OC,
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
constexpr std::array<std::int32_t,32> op17_mult = {1818415719,1483719127,2027094634,1212786360,1718181835,1890243475,1666664234,1127797955,1190529712,1544064846,1714947475,1287927853,1806339417,1692158998,1555047281,1131611751,1296262503,1221966222,1465325713,1414902161,1445545554,1953942482,1767046067,1750196408,1653078050,1560561113,1555588356,1713946075,1320497782,1613644991,1568973858,1544728783};
constexpr std::array<int,32> op17_shift = {-12,-12,-12,-11,-12,-12,-12,-11,-12,-12,-12,-11,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12,-12};
constexpr std::array<std::int32_t,24> op33_mult = {1399627741,1178697116,2080317620,1536423034,1252813837,1206407858,1357344291,1610718035,1495897733,1220699284,1618440846,1982019715,1199025935,1553210105,1552882344,1551360338,1138580344,1611095757,1514994267,1214045748,1513460876,1805294771,1754005327,1683740165};
constexpr std::array<int,24> op33_shift = {-8,-7,-8,-8,-7,-8,-8,-8,-8,-8,-8,-9,-7,-9,-8,-8,-8,-8,-8,-8,-8,-8,-7,-8};
constexpr std::array<std::int32_t,24> op34_mult = {1777062773,1316607383,1729026964,1575366115,1201319964,1374747948,1667213728,1783066206,1815061291,1406201729,1360405040,1970809119,1755323316,1523937948,1409824250,1449061316,1182251867,1846497651,1544617708,2065917223,1274509655,1288035487,1307558207,1546254288};
constexpr std::array<int,24> op34_shift = {-7,-7,-8,-7,-8,-8,-8,-8,-8,-8,-7,-8,-8,-8,-7,-7,-7,-7,-7,-8,-6,-7,-7,-7};
constexpr std::array<std::int32_t,24> op38_mult = {1706777101,1873406234,2012101104,1273944939,1874530938,2093110320,2115489428,1113632564,1929068525,1118645908,1977477077,2026118332,1362267410,1479844932,1883335489,1362809301,1869930466,2039495099,1784865269,2029682601,1227774834,2066013495,1581256012,1558376584};
constexpr std::array<int,24> op38_shift = {-10,-8,-9,-9,-9,-9,-9,-8,-11,-8,-9,-9,-8,-8,-9,-9,-9,-10,-8,-9,-8,-9,-8,-9};
constexpr std::array<std::int32_t,24> op39_mult = {2036642514,2103657957,1272888330,1471705826,1387307099,1766666955,1427078537,1747486700,1989767676,1551687029,1326590869,1212605777,1950267401,1385471908,1141936396,1385997787,1360400373,2069319851,1136793732,1299699793,1336116013,1084940561,1262404764,1915630121};
constexpr std::array<int,24> op39_shift = {-7,-8,-8,-8,-7,-8,-7,-6,-8,-8,-7,-6,-7,-7,-7,-7,-8,-8,-7,-8,-7,-7,-6,-8};
constexpr std::array<std::int32_t,24> op43_mult = {1191063693,1093752679,1716820949,1440441340,1303593875,1752914433,1542674267,1210121652,1647639843,1730190799,1571773508,1438375195,1792944068,1117433982,1955425599,1119241213,1367181391,1509820004,1572802169,1492566536,1377592030,1709248838,1235076029,1858573491};
constexpr std::array<int,24> op43_shift = {-9,-9,-8,-8,-8,-8,-8,-8,-9,-8,-10,-8,-9,-9,-10,-7,-9,-8,-9,-8,-8,-8,-8,-12};
constexpr std::array<std::int32_t,24> op47_mult = {1268069779,1400863170,1644009443,1934074523,1311418159,1943288851,1765326138,2026994369,1282678434,1819666092,1637420776,1656126478,2017139701,1180977674,2118600526,1263022430,1308209401,1546190026,1186130443,1227300313,1780594308,1440696741,1108540969,1260027087};
constexpr std::array<int,24> op47_shift = {-9,-9,-9,-11,-8,-13,-9,-10,-9,-9,-9,-9,-9,-8,-10,-9,-9,-9,-8,-8,-9,-11,-8,-8};
constexpr std::array<std::int32_t,32> op49_mult = {1412947805,1389520900,1409433169,1743777681,2132213669,1457362429,1289284382,1356246502,1309980581,1163973553,1515821050,1388789386,1598355310,1077066144,2007216296,1623624160,1579029082,1388249410,1149741597,1889317930,1370454674,2144424957,1379471248,1132579200,1959858646,1755932743,1501098301,1201986510,1342456271,1243698713,1448474585,1275728868};
constexpr std::array<int,32> op49_shift = {-8,-8,-8,-8,-9,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-9,-8,-9,-8,-8,-9,-8,-8,-8,-8,-8,-8,-8};
constexpr std::array<std::int32_t,24> op50_mult = {1514180639,1312961872,1588772594,1698615280,1335420909,2112661424,1251529026,1957350756,1362006790,1761116346,1925226595,1846195857,1145551332,1112073747,2128327665,2116464917,1317600159,1926768534,1882543445,1499263494,1269938781,1487461719,1418350665,1578360698};
constexpr std::array<int,24> op50_shift = {-8,-8,-7,-7,-7,-9,-8,-9,-7,-8,-8,-9,-7,-7,-8,-8,-9,-8,-8,-8,-8,-8,-8,-8};
constexpr std::array<std::int32_t,24> op51_mult = {1972607167,2022648434,1107672055,1250021901,1262438954,1236879418,1585951290,1169932304,1245156203,1187508266,1540514831,1360940667,2007402967,1347876104,2064355872,1498928796,1761357193,1888273408,1176179887,1422841231,1680881939,1137948855,1299912618,1681861235};
constexpr std::array<int,24> op51_shift = {-7,-7,-6,-7,-7,-7,-7,-7,-7,-7,-8,-7,-9,-6,-7,-7,-8,-7,-6,-7,-10,-7,-7,-7};
constexpr std::array<std::int32_t,24> op55_mult = {1757390214,1868159348,1194432736,1950241709,1224445539,1584246129,1495868808,1528214224,1214339521,1728256323,1685096951,1965453322,1078897143,1756707304,2053784526,1779040484,1675393762,1733545968,1588387746,1132996523,1241979219,1278465408,2096589577,1834380480};
constexpr std::array<int,24> op55_shift = {-11,-10,-10,-10,-9,-10,-11,-11,-10,-11,-11,-10,-10,-10,-10,-10,-10,-9,-10,-10,-11,-10,-11,-10};
constexpr std::array<std::int32_t,24> op56_mult = {1958337137,1283536292,1487515620,1892444093,1602391738,1128568231,1201123938,1119497417,2052379383,1140279507,1357987696,2132385841,1150638366,1336470426,2097131065,1794097716,2057525545,1330758150,1628078000,2022189384,1379976165,1704102291,1760801413,1841490517};
constexpr std::array<int,24> op56_shift = {-8,-8,-8,-12,-9,-8,-7,-7,-9,-8,-9,-8,-7,-8,-8,-8,-8,-8,-8,-8,-9,-9,-8,-8};
constexpr std::array<std::int32_t,24> op60_mult = {1434543589,1245945060,1263733401,1674142346,1910326580,1118825653,1641066451,1084799080,1692229945,1508075150,1243614746,1162306925,1352677748,1533317780,2042280280,1971905485,1966957180,1545394665,1479336182,1701265354,1582031384,1636301295,1454380407,1334044168};
constexpr std::array<int,24> op60_shift = {-9,-8,-9,-10,-8,-8,-8,-9,-8,-8,-8,-8,-7,-8,-7,-9,-9,-9,-8,-8,-8,-9,-8,-7};
constexpr std::array<std::int32_t,24> op64_mult = {1229643925,1185059821,1307768889,1928290973,1464501351,1222820784,1375787401,1240220186,1713284802,1257820558,1265648355,1910053853,2111423237,2143104858,1277134644,1360241888,1238528287,1977936231,1497970509,1739002109,2075615133,2067211689,1449541584,2036459790};
constexpr std::array<int,24> op64_shift = {-8,-9,-8,-10,-11,-9,-9,-8,-9,-8,-10,-9,-9,-9,-9,-10,-7,-9,-9,-9,-9,-9,-9,-9};
constexpr std::array<std::int32_t,64> op66_mult = {1110149903,1131949952,1209281594,1241175605,1170519949,2058149406,1085219133,1956556236,2039044295,1261691288,1333963536,2134963599,1122476778,1991641292,1278904638,1363505287,2102579907,1444220385,1351421892,1552072816,1733078857,1911921674,1516600000,1439202478,1498896749,1405023394,1835576582,2014706325,1284888951,1788753206,1387264718,1568645126,1597969314,1941811331,1880053471,1106934248,1840141939,1811765537,1646087511,2004876753,1336467589,1653839884,1250111730,1094497937,1226234805,1757478676,1884134263,1850531647,1997924728,1219644297,1510608283,1446877757,1869127584,1195663165,1169682579,1250177391,1538344189,1185779364,1102362467,2126549170,1906649835,1853549446,1296919752,1264769957};
constexpr std::array<int,64> op66_shift = {-8,-7,-8,-7,-8,-8,-6,-8,-8,-8,-8,-9,-7,-9,-8,-8,-8,-7,-7,-8,-7,-7,-8,-8,-7,-8,-8,-9,-7,-8,-7,-7,-8,-8,-8,-8,-8,-7,-8,-8,-8,-8,-7,-7,-7,-8,-8,-7,-8,-7,-8,-8,-8,-7,-7,-7,-8,-7,-7,-8,-9,-8,-8,-7};
constexpr std::array<std::int32_t,24> op67_mult = {1413093413,1336440450,1508765746,1170241209,1195248433,1293861571,2024482614,1343946711,1382282185,1195369341,1140287266,1117316043,1629467437,1175585374,1099357041,2115093270,2008376937,1822747462,2034168958,1603026469,1149780142,1111159898,1474880769,1326199644};
constexpr std::array<int,24> op67_shift = {-11,-11,-10,-10,-9,-10,-11,-10,-14,-11,-11,-11,-10,-11,-12,-11,-14,-10,-11,-11,-10,-10,-10,-10};
constexpr std::array<std::int32_t,24> op68_mult = {1252150408,1898683393,1110740658,2007188841,1716631981,1220066698,1157596825,1699125717,1283300889,1099484092,1210259878,1495624467,1073958920,2146221227,1734010051,1311936715,1307149938,1975599629,1521292626,1726704709,1100463057,1163068351,1093938779,1222215729};
constexpr std::array<int,24> op68_shift = {-9,-8,-8,-9,-9,-8,-9,-9,-9,-10,-9,-10,-8,-9,-9,-9,-9,-9,-9,-8,-9,-8,-11,-9};
constexpr std::array<std::int32_t,24> op72_mult = {1864470116,1373684861,1936427011,1629721859,1364868163,1488115299,1205979087,1227597359,1074479849,1334730436,1839898885,2005998669,1242293418,1326728919,1093621022,1649945565,1088928031,1134965059,1343553487,1683109677,1403866330,2110901483,1222133660,1363279149};
constexpr std::array<int,24> op72_shift = {-10,-9,-10,-9,-14,-11,-10,-10,-9,-10,-9,-10,-9,-10,-10,-11,-10,-8,-9,-9,-10,-11,-9,-10};
constexpr std::array<std::int32_t,24> op73_mult = {1429352920,1912333868,1292929812,1589472976,1194301858,2090304226,1104747849,1520087547,1721862475,1632685811,1555108126,1825590962,1133406718,1594296016,1171008064,1415488266,1662154027,2097202882,1121178533,1242694891,1226934497,1564561113,1348029033,1459068978};
constexpr std::array<int,24> op73_shift = {-9,-9,-9,-8,-8,-10,-9,-9,-8,-9,-9,-9,-7,-8,-8,-8,-9,-9,-8,-9,-9,-8,-9,-9};
constexpr std::array<std::int32_t,24> op77_mult = {1112466320,1704533932,1618829202,1909484085,2048261978,1165669624,1954876691,2087171201,1289759114,1177352426,1235044931,1549217399,1203521152,1844497796,1287786447,1609662123,1764243594,1369199421,1247852245,1793596531,1504980238,1106125257,1163685673,1531117707};
constexpr std::array<int,24> op77_shift = {-8,-9,-10,-15,-9,-8,-9,-9,-8,-9,-7,-9,-9,-9,-7,-11,-9,-10,-9,-9,-9,-11,-9,-8};
constexpr std::array<std::int32_t,24> op81_mult = {1636527628,1506754996,2146860426,1619618125,2001410819,1276399496,1347140706,1578706071,1612600879,1799001612,1253725358,1515996449,1551202399,1766049252,1756217011,1260105475,1337180306,1591417386,2032421395,1799182444,1282627931,1677013158,1660993744,1375361762};
constexpr std::array<int,24> op81_shift = {-11,-10,-11,-10,-13,-10,-10,-11,-10,-10,-11,-11,-10,-11,-11,-10,-10,-10,-11,-14,-11,-11,-11,-11};
constexpr std::array<std::int32_t,96> op83_mult = {2036603110,1128896039,1477912050,1165239912,1184400295,1532183630,1403808087,2046065890,1883877125,2058417919,1426163281,1214281222,1501989762,1773312984,1296503597,1805778887,1588460510,1443559181,1314460818,1318155433,2026714035,1656024605,1116713440,1227836965,1575239845,1832388635,2031335946,2122649686,1914068226,1814960731,1141634980,1149826347,1241081519,1728294727,1576276712,1144350924,1077429038,1398282121,1569645996,2085594337,1567078452,1103302500,1610559578,1293059944,1307143622,1949039324,1785242409,2117403734,1603648293,1733058491,1325524925,1191056560,1509234925,1803157111,2035165957,1721037390,1536881356,1395160542,1664288375,1930101220,1651596934,1274591760,1656859671,1221443008,2013222761,1294334862,1780268541,1776480957,1607138153,1682775652,2070029512,1754036575,1446257048,1979935628,1111087126,1671604188,1183172323,1605948919,1628652287,1422457137,1137084086,1074972357,1319123865,1412357959,1497886930,1266205876,1880815957,1084515471,1613162723,1163736906,1429008719,1483515859,1118707396,1156092288,1389287231,1416638983};
constexpr std::array<int,96> op83_shift = {-7,-6,-6,-9,-6,-5,-6,-6,-8,-8,-8,-6,-6,-7,-6,-6,-10,-7,-6,-5,-7,-6,-6,-5,-6,-7,-7,-7,-7,-7,-11,-7,-6,-7,-6,-6,-5,-7,-8,-7,-7,-6,-6,-6,-5,-6,-7,-7,-6,-6,-6,-7,-7,-7,-6,-6,-7,-6,-6,-6,-6,-7,-6,-5,-7,-5,-6,-7,-7,-7,-7,-6,-6,-7,-6,-7,-8,-6,-12,-7,-6,-10,-6,-6,-6,-6,-6,-6,-5,-6,-5,-6,-5,-10,-6,-6};
constexpr std::array<std::int32_t,1> op87_mult = {1262285017};
constexpr std::array<int,1> op87_shift = {-11};
constexpr std::array<std::int8_t,256> logistic_lut = {-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-127,-127,-127,-127,-127,-127,-126,-126,-125,-125,-124,-123,-122,-121,-120,-118,-116,-113,-110,-107,-103,-98,-92,-85,-77,-69,-59,-49,-37,-25,-13,0,13,25,37,49,59,69,77,85,92,98,103,107,110,113,116,118,120,121,122,123,124,125,125,126,126,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127,127};
}

void Model::reset(){
  state_stream_11_states.fill(static_cast<std::int8_t>(-128));
  state_stream_12_states.fill(static_cast<std::int8_t>(-128));
  state_stream_13_states.fill(static_cast<std::int8_t>(-128));
  state_stream_14_states.fill(static_cast<std::int8_t>(-128));
  state_stream_15_states.fill(static_cast<std::int8_t>(-128));
  state_stream_16_states.fill(static_cast<std::int8_t>(-128));
  state_stream_17_states.fill(static_cast<std::int8_t>(-128));
  state_stream_18_states.fill(static_cast<std::int8_t>(-128));
  state_stream_19_states.fill(static_cast<std::int8_t>(-128));
  state_stream_20_states.fill(static_cast<std::int8_t>(-128));
  state_stream_21_states.fill(static_cast<std::int8_t>(-128));
}

std::size_t Model::working_memory_bytes(){return 7473;}

NN2PROG_ESP32_IRAM std::uint8_t Model::invoke(const std::array<std::int8_t,40>& input){
  std::array<std::int8_t,7104> buffer_0;
  std::array<std::int8_t,120> buffer_1;
  std::array<std::int8_t,120> buffer_2;
  std::array<std::int8_t,96> buffer_3;
  std::array<std::uint8_t,1> buffer_4;
  // op 12: RESHAPE
  const auto& tensor_71=input;
  // op 13: READ_VARIABLE
  const auto& tensor_72=state_stream_11_states;
  // op 14: STRIDED_SLICE
  ArrayView<const std::int8_t,160> tensor_73(tensor_72.data()+40);
  {
    ArrayView<std::int8_t,200> tensor_74(buffer_0.data());
    // op 15: CONCATENATION
    std::copy_n(tensor_73.begin(),160,tensor_74.begin()+0);
    std::copy_n(tensor_71.begin(),40,tensor_74.begin()+160);
  }
  // op 16: ASSIGN_VARIABLE
  std::copy_n(ArrayView<const std::int8_t,200>(buffer_0.data()).begin(),ArrayView<const std::int8_t,200>(buffer_0.data()).size(),state_stream_11_states.begin());
  {
    ArrayView<std::int8_t,32> tensor_75(buffer_1.data());
    // op 17: CONV_2D
    dense(ArrayView<const std::int8_t,200>(buffer_0.data()),sg0_tensor59_bytes,sg0_tensor58_bytes,tensor_75,-128,21,op17_mult,op17_shift,-128,127);
  }
  // op 18: RESHAPE
  const auto& tensor_76=ArrayView<const std::int8_t,32>(buffer_1.data());
  // op 19: READ_VARIABLE
  const auto& tensor_77=state_stream_12_states;
  // op 20: READ_VARIABLE
  const auto& tensor_78=state_stream_13_states;
  // op 21: READ_VARIABLE
  const auto& tensor_79=state_stream_14_states;
  // op 22: READ_VARIABLE
  const auto& tensor_80=state_stream_15_states;
  // op 23: READ_VARIABLE
  const auto& tensor_81=state_stream_16_states;
  // op 24: READ_VARIABLE
  const auto& tensor_82=state_stream_17_states;
  // op 25: READ_VARIABLE
  const auto& tensor_83=state_stream_18_states;
  // op 26: READ_VARIABLE
  const auto& tensor_84=state_stream_19_states;
  // op 27: READ_VARIABLE
  const auto& tensor_85=state_stream_20_states;
  // op 28: READ_VARIABLE
  const auto& tensor_86=state_stream_21_states;
  // op 29: STRIDED_SLICE
  ArrayView<const std::int8_t,7008> tensor_87(tensor_86.data()+96);
  {
    std::array<std::int8_t,32> tensor_88;
    // op 30: MUL
    for(std::size_t i=0;i<tensor_88.size();++i){int av=static_cast<int>(tensor_76[i])-(21);int bv=static_cast<int>(signed_byte(sg0_tensor57_bytes[i%4]))-(-128);int v=requantize(av*bv,1237366504,-7)+(29);tensor_88[i]=static_cast<std::int8_t>(std::clamp(v,-128,127));}
    std::copy_n(tensor_88.begin(),tensor_88.size(),buffer_1.begin());
  }
  {
    std::array<std::int8_t,32> tensor_89;
    // op 31: ADD
    for(std::size_t i=0;i<tensor_89.size();++i){int av=(static_cast<int>(ArrayView<const std::int8_t,32>(buffer_1.data())[i])-(29))*(1<<20);int bv=(static_cast<int>(signed_byte(sg0_tensor56_bytes[i%4]))-(-128))*(1<<20);int x=requantize(av,1073741824,0)+requantize(bv,1679336654,-6);int v=requantize(x,1402506963,-17)+(-128);tensor_89[i]=static_cast<std::int8_t>(std::clamp(v,-128,127));}
    std::copy_n(tensor_89.begin(),tensor_89.size(),buffer_1.begin());
  }
  // op 32: RESHAPE
  const auto& tensor_90=ArrayView<const std::int8_t,32>(buffer_1.data());
  {
    ArrayView<std::int8_t,24> tensor_91(buffer_0.data());
    // op 33: CONV_2D
    dense(tensor_90,sg0_tensor55_bytes,sg0_tensor54_bytes,tensor_91,-128,-128,op33_mult,op33_shift,-128,127);
  }
  {
    ArrayView<std::int8_t,24> tensor_92(buffer_2.data());
    // op 34: CONV_2D
    dense(tensor_90,sg0_tensor53_bytes,sg0_tensor52_bytes,tensor_92,-128,-128,op34_mult,op34_shift,-128,127);
  }
  {
    ArrayView<std::int8_t,72> tensor_93(buffer_3.data());
    // op 35: CONCATENATION
    std::copy_n(tensor_77.begin(),48,tensor_93.begin()+0);
    std::copy_n(ArrayView<const std::int8_t,24>(buffer_2.data()).begin(),24,tensor_93.begin()+48);
  }
  // op 36: STRIDED_SLICE
  ArrayView<const std::int8_t,48> tensor_94(ArrayView<const std::int8_t,72>(buffer_3.data()).data()+24);
  // op 37: ASSIGN_VARIABLE
  std::copy_n(tensor_94.begin(),tensor_94.size(),state_stream_12_states.begin());
  {
    ArrayView<std::int8_t,24> tensor_95(buffer_2.data());
    // op 38: CONV_2D
    dense(ArrayView<const std::int8_t,72>(buffer_3.data()),sg0_tensor51_bytes,sg0_tensor50_bytes,tensor_95,-128,-128,op38_mult,op38_shift,-128,127);
  }
  {
    ArrayView<std::int8_t,24> tensor_96(buffer_3.data());
    // op 39: CONV_2D
    dense(tensor_90,sg0_tensor49_bytes,sg0_tensor48_bytes,tensor_96,-128,-128,op39_mult,op39_shift,-128,127);
  }
  {
    ArrayView<std::int8_t,72> tensor_97(buffer_1.data());
    // op 40: CONCATENATION
    std::copy_n(tensor_78.begin(),48,tensor_97.begin()+0);
    std::copy_n(ArrayView<const std::int8_t,24>(buffer_3.data()).begin(),24,tensor_97.begin()+48);
  }
  // op 41: STRIDED_SLICE
  ArrayView<const std::int8_t,48> tensor_98(ArrayView<const std::int8_t,72>(buffer_1.data()).data()+24);
  // op 42: ASSIGN_VARIABLE
  std::copy_n(tensor_98.begin(),tensor_98.size(),state_stream_13_states.begin());
  {
    ArrayView<std::int8_t,24> tensor_99(buffer_3.data());
    // op 43: CONV_2D
    dense(ArrayView<const std::int8_t,72>(buffer_1.data()),sg0_tensor47_bytes,sg0_tensor46_bytes,tensor_99,-128,-128,op43_mult,op43_shift,-128,127);
  }
  {
    ArrayView<std::int8_t,72> tensor_100(buffer_1.data());
    // op 44: CONCATENATION
    std::copy_n(tensor_79.begin(),48,tensor_100.begin()+0);
    std::copy_n(ArrayView<const std::int8_t,24>(buffer_3.data()).begin(),24,tensor_100.begin()+48);
  }
  // op 45: STRIDED_SLICE
  ArrayView<const std::int8_t,48> tensor_101(ArrayView<const std::int8_t,72>(buffer_1.data()).data()+24);
  // op 46: ASSIGN_VARIABLE
  std::copy_n(tensor_101.begin(),tensor_101.size(),state_stream_14_states.begin());
  {
    ArrayView<std::int8_t,24> tensor_102(buffer_3.data());
    // op 47: CONV_2D
    dense(ArrayView<const std::int8_t,72>(buffer_1.data()),sg0_tensor45_bytes,sg0_tensor44_bytes,tensor_102,-128,-128,op47_mult,op47_shift,-128,127);
  }
  {
    ArrayView<std::int8_t,72> tensor_103(buffer_1.data());
    // op 48: CONCATENATION
    std::copy_n(ArrayView<const std::int8_t,24>(buffer_0.data()).begin(),24,tensor_103.begin()+0);
    std::copy_n(ArrayView<const std::int8_t,24>(buffer_2.data()).begin(),24,tensor_103.begin()+24);
    std::copy_n(ArrayView<const std::int8_t,24>(buffer_3.data()).begin(),24,tensor_103.begin()+48);
  }
  {
    ArrayView<std::int8_t,32> tensor_104(buffer_2.data());
    // op 49: CONV_2D
    dense(ArrayView<const std::int8_t,72>(buffer_1.data()),sg0_tensor43_bytes,sg0_tensor42_bytes,tensor_104,-128,-128,op49_mult,op49_shift,-128,127);
  }
  {
    ArrayView<std::int8_t,24> tensor_105(buffer_1.data());
    // op 50: CONV_2D
    dense(ArrayView<const std::int8_t,32>(buffer_2.data()),sg0_tensor41_bytes,sg0_tensor40_bytes,tensor_105,-128,-128,op50_mult,op50_shift,-128,127);
  }
  {
    ArrayView<std::int8_t,24> tensor_106(buffer_3.data());
    // op 51: CONV_2D
    dense(ArrayView<const std::int8_t,32>(buffer_2.data()),sg0_tensor39_bytes,sg0_tensor38_bytes,tensor_106,-128,-128,op51_mult,op51_shift,-128,127);
  }
  {
    ArrayView<std::int8_t,120> tensor_107(buffer_0.data());
    // op 52: CONCATENATION
    std::copy_n(tensor_80.begin(),96,tensor_107.begin()+0);
    std::copy_n(ArrayView<const std::int8_t,24>(buffer_3.data()).begin(),24,tensor_107.begin()+96);
  }
  // op 53: STRIDED_SLICE
  ArrayView<const std::int8_t,96> tensor_108(ArrayView<const std::int8_t,120>(buffer_0.data()).data()+24);
  // op 54: ASSIGN_VARIABLE
  std::copy_n(tensor_108.begin(),tensor_108.size(),state_stream_15_states.begin());
  {
    ArrayView<std::int8_t,24> tensor_109(buffer_3.data());
    // op 55: CONV_2D
    dense(ArrayView<const std::int8_t,120>(buffer_0.data()),sg0_tensor37_bytes,sg0_tensor36_bytes,tensor_109,-128,-128,op55_mult,op55_shift,-128,127);
  }
  {
    ArrayView<std::int8_t,24> tensor_110(buffer_0.data());
    // op 56: CONV_2D
    dense(ArrayView<const std::int8_t,32>(buffer_2.data()),sg0_tensor35_bytes,sg0_tensor34_bytes,tensor_110,-128,-128,op56_mult,op56_shift,-128,127);
  }
  {
    auto& tensor_111=buffer_2;
    // op 57: CONCATENATION
    std::copy_n(tensor_81.begin(),96,tensor_111.begin()+0);
    std::copy_n(ArrayView<const std::int8_t,24>(buffer_0.data()).begin(),24,tensor_111.begin()+96);
  }
  // op 58: STRIDED_SLICE
  ArrayView<const std::int8_t,96> tensor_112(buffer_2.data()+24);
  // op 59: ASSIGN_VARIABLE
  std::copy_n(tensor_112.begin(),tensor_112.size(),state_stream_16_states.begin());
  {
    ArrayView<std::int8_t,24> tensor_113(buffer_0.data());
    // op 60: CONV_2D
    dense(buffer_2,sg0_tensor33_bytes,sg0_tensor32_bytes,tensor_113,-128,-128,op60_mult,op60_shift,-128,127);
  }
  {
    auto& tensor_114=buffer_2;
    // op 61: CONCATENATION
    std::copy_n(tensor_82.begin(),96,tensor_114.begin()+0);
    std::copy_n(ArrayView<const std::int8_t,24>(buffer_0.data()).begin(),24,tensor_114.begin()+96);
  }
  // op 62: STRIDED_SLICE
  ArrayView<const std::int8_t,96> tensor_115(buffer_2.data()+24);
  // op 63: ASSIGN_VARIABLE
  std::copy_n(tensor_115.begin(),tensor_115.size(),state_stream_17_states.begin());
  {
    ArrayView<std::int8_t,24> tensor_116(buffer_0.data());
    // op 64: CONV_2D
    dense(buffer_2,sg0_tensor31_bytes,sg0_tensor30_bytes,tensor_116,-128,-128,op64_mult,op64_shift,-128,127);
  }
  {
    ArrayView<std::int8_t,72> tensor_117(buffer_2.data());
    // op 65: CONCATENATION
    std::copy_n(ArrayView<const std::int8_t,24>(buffer_1.data()).begin(),24,tensor_117.begin()+0);
    std::copy_n(ArrayView<const std::int8_t,24>(buffer_3.data()).begin(),24,tensor_117.begin()+24);
    std::copy_n(ArrayView<const std::int8_t,24>(buffer_0.data()).begin(),24,tensor_117.begin()+48);
  }
  {
    ArrayView<std::int8_t,64> tensor_118(buffer_1.data());
    // op 66: CONV_2D
    dense(ArrayView<const std::int8_t,72>(buffer_2.data()),sg0_tensor29_bytes,sg0_tensor28_bytes,tensor_118,-128,-128,op66_mult,op66_shift,-128,127);
  }
  {
    ArrayView<std::int8_t,24> tensor_119(buffer_3.data());
    // op 67: CONV_2D
    dense(ArrayView<const std::int8_t,64>(buffer_1.data()),sg0_tensor27_bytes,sg0_tensor26_bytes,tensor_119,-128,-128,op67_mult,op67_shift,-128,127);
  }
  {
    ArrayView<std::int8_t,24> tensor_120(buffer_2.data());
    // op 68: CONV_2D
    dense(ArrayView<const std::int8_t,64>(buffer_1.data()),sg0_tensor25_bytes,sg0_tensor24_bytes,tensor_120,-128,-128,op68_mult,op68_shift,-128,127);
  }
  {
    ArrayView<std::int8_t,120> tensor_121(buffer_0.data());
    // op 69: CONCATENATION
    std::copy_n(tensor_83.begin(),96,tensor_121.begin()+0);
    std::copy_n(ArrayView<const std::int8_t,24>(buffer_2.data()).begin(),24,tensor_121.begin()+96);
  }
  // op 70: STRIDED_SLICE
  ArrayView<const std::int8_t,96> tensor_122(ArrayView<const std::int8_t,120>(buffer_0.data()).data()+24);
  // op 71: ASSIGN_VARIABLE
  std::copy_n(tensor_122.begin(),tensor_122.size(),state_stream_18_states.begin());
  {
    ArrayView<std::int8_t,24> tensor_123(buffer_2.data());
    // op 72: CONV_2D
    dense(ArrayView<const std::int8_t,120>(buffer_0.data()),sg0_tensor23_bytes,sg0_tensor22_bytes,tensor_123,-128,-128,op72_mult,op72_shift,-128,127);
  }
  {
    ArrayView<std::int8_t,24> tensor_124(buffer_0.data());
    // op 73: CONV_2D
    dense(ArrayView<const std::int8_t,64>(buffer_1.data()),sg0_tensor21_bytes,sg0_tensor20_bytes,tensor_124,-128,-128,op73_mult,op73_shift,-128,127);
  }
  {
    auto& tensor_125=buffer_1;
    // op 74: CONCATENATION
    std::copy_n(tensor_84.begin(),96,tensor_125.begin()+0);
    std::copy_n(ArrayView<const std::int8_t,24>(buffer_0.data()).begin(),24,tensor_125.begin()+96);
  }
  // op 75: STRIDED_SLICE
  ArrayView<const std::int8_t,96> tensor_126(buffer_1.data()+24);
  // op 76: ASSIGN_VARIABLE
  std::copy_n(tensor_126.begin(),tensor_126.size(),state_stream_19_states.begin());
  {
    ArrayView<std::int8_t,24> tensor_127(buffer_0.data());
    // op 77: CONV_2D
    dense(buffer_1,sg0_tensor19_bytes,sg0_tensor18_bytes,tensor_127,-128,-128,op77_mult,op77_shift,-128,127);
  }
  {
    auto& tensor_128=buffer_1;
    // op 78: CONCATENATION
    std::copy_n(tensor_85.begin(),96,tensor_128.begin()+0);
    std::copy_n(ArrayView<const std::int8_t,24>(buffer_0.data()).begin(),24,tensor_128.begin()+96);
  }
  // op 79: STRIDED_SLICE
  ArrayView<const std::int8_t,96> tensor_129(buffer_1.data()+24);
  // op 80: ASSIGN_VARIABLE
  std::copy_n(tensor_129.begin(),tensor_129.size(),state_stream_20_states.begin());
  {
    ArrayView<std::int8_t,24> tensor_130(buffer_0.data());
    // op 81: CONV_2D
    dense(buffer_1,sg0_tensor17_bytes,sg0_tensor16_bytes,tensor_130,-128,-128,op81_mult,op81_shift,-128,127);
  }
  {
    ArrayView<std::int8_t,72> tensor_131(buffer_1.data());
    // op 82: CONCATENATION
    std::copy_n(ArrayView<const std::int8_t,24>(buffer_3.data()).begin(),24,tensor_131.begin()+0);
    std::copy_n(ArrayView<const std::int8_t,24>(buffer_2.data()).begin(),24,tensor_131.begin()+24);
    std::copy_n(ArrayView<const std::int8_t,24>(buffer_0.data()).begin(),24,tensor_131.begin()+48);
  }
  {
    auto& tensor_132=buffer_3;
    // op 83: CONV_2D
    dense(ArrayView<const std::int8_t,72>(buffer_1.data()),sg0_tensor15_bytes,sg0_tensor14_bytes,tensor_132,-128,-128,op83_mult,op83_shift,-128,127);
  }
  {
    auto& tensor_133=buffer_0;
    // op 84: CONCATENATION
    std::copy_n(tensor_87.begin(),7008,tensor_133.begin()+0);
    std::copy_n(buffer_3.begin(),96,tensor_133.begin()+7008);
  }
  // op 85: ASSIGN_VARIABLE
  std::copy_n(buffer_0.begin(),buffer_0.size(),state_stream_21_states.begin());
  // op 86: RESHAPE
  const auto& tensor_134=buffer_0;
  {
    ArrayView<std::int8_t,1> tensor_135(buffer_3.data());
    // op 87: FULLY_CONNECTED
    dense(tensor_134,sg0_tensor13_bytes,sg0_tensor12_bytes,tensor_135,-128,23,op87_mult,op87_shift,-128,127);
  }
  {
    std::array<std::int8_t,1> tensor_136;
    // op 88: LOGISTIC
    tensor_136[0]=logistic_lut[static_cast<std::uint8_t>(static_cast<int>(ArrayView<const std::int8_t,1>(buffer_3.data())[0])+128)];
    std::copy_n(tensor_136.begin(),tensor_136.size(),buffer_3.begin());
  }
  {
    auto& tensor_137=buffer_4;
    // op 89: QUANTIZE
    tensor_137[0]=static_cast<std::uint8_t>(static_cast<int>(ArrayView<const std::int8_t,1>(buffer_3.data())[0])+128);
  }
  return buffer_4[0];
}
}
