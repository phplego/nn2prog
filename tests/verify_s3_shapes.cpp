// Compare complete S3 kernel outputs against scalar computations.
// Compile with the generated model directory on the include path.
#include "model.cpp"
#include <cstdio>
namespace nn2prog::generated {
template<std::size_t C,int Stride,int Pad> bool depthwise_tail() {
  constexpr int H=5,W=7,OH=(H+2*Pad-3)/Stride+1,OW=(W+2*Pad-3)/Stride+1;
  constexpr std::size_t P=(C+15)/16*16;
  alignas(16) std::array<std::int8_t,H*W*C> input;
  alignas(16) std::array<std::int8_t,9*P> weights{};
  std::array<std::int32_t,C> bias{},mult;
  std::array<int,C> shift;
  std::array<std::int8_t,OH*OW*C> output;
  constexpr int iz=-17,oz=3;
  for(std::size_t c=0;c<C;++c){
    mult[c]=1234567890;shift[c]=-int(c%8)-1;
    for(int tap=0;tap<9;++tap){
      weights[tap*P+c]=int((tap*73+c*101)%63)-31;
      bias[c]-=iz*int(weights[tap*P+c]);
    }
  }
  for(int sample=0;sample<12;++sample){
    for(std::size_t i=0;i<input.size();++i)
      input[i]=sample==0?-128:sample==1?127:sample==2?0:int((i*73+sample*101)%256)-128;
    esp32s3_depthwise_3x3<H,W,C,OH,OW>(input,weights,bias,output,Stride,Stride,Pad,Pad,iz,oz,mult,shift,-128,127);
    for(int y=0;y<OH;++y)for(int x=0;x<OW;++x)for(std::size_t c=0;c<C;++c){
      int acc=0;
      for(int fy=0;fy<3;++fy)for(int fx=0;fx<3;++fx){
        int iy=y*Stride+fy-Pad,ix=x*Stride+fx-Pad;
        int v=iy>=0&&iy<H&&ix>=0&&ix<W?input[(iy*W+ix)*C+c]:iz;
        acc+=(v-iz)*weights[(fy*3+fx)*P+c];
      }
      int ref=std::clamp<std::int32_t>(requantize(acc,mult[c],shift[c])+oz,-128,127);
      if(output[(y*OW+x)*C+c]!=ref){
        std::printf("mismatch,C=%u,stride=%d,pad=%d,sample=%d,y=%d,x=%d,c=%u,acc=%d,got=%d,ref=%d\n",
          unsigned(C),Stride,Pad,sample,y,x,unsigned(c),acc,int(output[(y*OW+x)*C+c]),ref);
        return false;
      }
    }
  }
  return true;
}
template<std::size_t IC,std::size_t OC=32> bool pointwise_order() {
  constexpr std::size_t P=(IC+15)/16*16;
  alignas(16) std::array<std::int8_t,6*IC> input;
  alignas(16) std::array<std::int8_t,OC*P> weights{};
  std::array<std::int32_t,OC> bias{},mult;
  std::array<int,OC> shift;
  std::array<std::int8_t,3*OC> output;
  for(std::size_t c=0;c<OC;++c){
    mult[c]=1234567890;shift[c]=-8;
    for(std::size_t i=0;i<IC;++i)weights[c/16*P*16+i*16+c%16]=int((c*73+i*101)%31)-15;
  }
  for(int sample=0;sample<12;++sample){
    for(std::size_t i=0;i<input.size();++i)
      input[i]=sample==0?-128:sample==1?127:sample==2?0:int((i*73+sample*101)%256)-128;
    esp32s3_conv_1x1_qacc<1,6,IC,OC,1,3>(input,weights,bias,output,1,2,3,mult,shift,-128,127);
    for(std::size_t x=0;x<3;++x)for(std::size_t c=0;c<OC;++c){
      int acc=0;for(std::size_t i=0;i<IC;++i)acc+=int(input[x*2*IC+i])*weights[c/16*P*16+i*16+c%16];
      if(output[x*OC+c]!=std::clamp<std::int32_t>(requantize(acc,mult[c],shift[c])+3,-128,127))return false;
    }
  }
  return true;
}
bool run_tests(){
  bool ok=true;
#define DW(C) {bool pass=depthwise_tail<C,1,1>()&&depthwise_tail<C,2,0>();std::printf("depthwise,%d,%s\n",C,pass?"PASS":"FAIL");ok&=pass;}
  DW(1) DW(3) DW(4) DW(7) DW(8) DW(15) DW(16) DW(17) DW(24) DW(31) DW(32) DW(33)
#undef DW
#define PW(C) {bool pass=pointwise_order<C>();std::printf("pointwise,%d,%s\n",C,pass?"PASS":"FAIL");ok&=pass;}
  PW(3) PW(8) PW(16) PW(17) PW(128)
  {bool pass=pointwise_order<256,128>();std::printf("pointwise,multi-tile,%s\n",pass?"PASS":"FAIL");ok&=pass;}
#undef PW
  std::printf("benchmark_complete,%s\n",ok?"PASS":"FAIL");return ok;
}
}
#ifdef ESP_PLATFORM
extern "C" void app_main(){nn2prog::generated::run_tests();}
#else
int main(){return nn2prog::generated::run_tests()?0:1;}
#endif
