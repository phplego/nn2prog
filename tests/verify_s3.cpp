#include "model.cpp"
#include <cstdio>
using namespace nn2prog::generated;

template<std::size_t... Lane>
bool check_lanes(const std::uint8_t* raw,const std::array<int,16>& expected,std::index_sequence<Lane...>) {
  return ((s3_lane20_fixed<Lane>(raw)==expected[Lane]) && ...);
}

std::int32_t reference_high_mul(std::int32_t a,std::int32_t b) {
  if(a==INT32_MIN && b==INT32_MIN)return INT32_MAX;
  const std::int64_t product=std::int64_t(a)*b;
  const std::int64_t nudge=product>=0 ? (1LL<<30) : 1-(1LL<<30);
  return std::int32_t((product+nudge)/(1LL<<31));
}

int main() {
  constexpr std::array<std::int32_t,1> positive{{123}}, negative{{-123}};
  constexpr std::array<int,1> good{{-31}}, zero{{0}}, left{{1}}, bad{{-32}};
  static_assert(s3_right_shift_only(positive,good));
  static_assert(!s3_right_shift_only(negative,good));
  static_assert(!s3_right_shift_only(positive,zero));
  static_assert(!s3_right_shift_only(positive,left));
  static_assert(!s3_right_shift_only(positive,bad));
  std::uint32_t rng=123;
  for(int i=0;i<1000000;++i) {
    rng=rng*1664525u+1013904223u; auto x=std::int32_t(rng);
    rng=rng*1664525u+1013904223u; auto m=std::int32_t(rng&0x7fffffffu);
    if(saturating_high_mul(x,std::int32_t(rng))!=reference_high_mul(x,std::int32_t(rng)))return 4;
    int shift=-1-i%31;
    if(s3_output_scale<true>(x,m,shift)!=requantize(x,m,shift))return 1;
  }
  for(auto x:{INT32_MIN,INT32_MIN+1,-1,0,1,INT32_MAX})
    for(auto y:{INT32_MIN,INT32_MIN+1,-1,0,1,INT32_MAX})
      if(saturating_high_mul(x,y)!=reference_high_mul(x,y))return 5;
  for(auto x:{INT32_MIN,INT32_MIN+1,-1,0,1,INT32_MAX})
    for(auto m:{0,1,1073741824,INT32_MAX})for(int shift=-31;shift<0;++shift)
      if(s3_output_scale<true>(x,m,shift)!=requantize(x,m,shift))return 2;
  alignas(16) std::array<std::int8_t,80> input{};
  alignas(16) std::array<std::int8_t,80*16> weights{};
  alignas(16) std::array<std::uint8_t,64> raw{};
  for(unsigned i=0;i<input.size();++i)input[i]=int(i%256)-128;
  for(unsigned i=0;i<weights.size();++i)weights[i]=int(i%31)-15;
  s3_pointwise_sums<80>(input.data(),weights.data(),raw.data());
  std::array<int,16> expected{};
  for(unsigned c=0;c<16;++c) {
    for(unsigned i=0;i<80;++i)expected[c]+=int(input[i])*weights[i*16+c];
  }
  if(!check_lanes(raw.data(),expected,std::make_index_sequence<16>{}))return 3;
  std::puts("S3 arithmetic and packed layout: PASS");
}
