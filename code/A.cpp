/*
マラソンマッチテンプレート
*/
//#define NDEBUG
#define ONLINE_JUDGE
#ifndef ONLINE_JUDGE
#define OPTUNA
#endif

#include<bits/stdc++.h>
using namespace std;
using ll=long long int;
//using Int=__int128;
#define mask(x) ((1LL<<x)-1)
#define ALL(A) A.begin(),A.end()
template<typename T1,typename T2> bool chmin(T1 &a,T2 b){if(a<=b)return 0; a=b; return 1;}
template<typename T1,typename T2> bool chmax(T1 &a,T2 b){if(a>=b)return 0; a=b; return 1;}
template<typename T> int bitUP(T x,int a){return (x>>a)&1;}
//→　↓　←　↑ 
int dh[4]={0,1,0,-1}, dw[4]={1,0,-1,0};
//左上から時計回り
//int dh[8]={-1,0,1,1,1,0,-1,-1}, dw[8]={1,1,1,0,-1,-1,-1,0};
long double EPS = 1e-6;
const ll INF=(1LL<<62);
const int MAX=(1<<30);
using pii=pair<int,int>;
using pil=pair<int,ll>;
using pli=pair<ll,int>;
using pll=pair<ll,ll>;
using psi=pair<string,int>;
using pis=pair<int,string>;
using psl=pair<string,ll>;
using pls=pair<ll,string>;
using pss=pair<string,string>;

using Graph=vector<vector<int>>;

using i8=int8_t;
using i16=int16_t;
using i32=int32_t;
using i64=int64_t;

using u8=uint8_t;
using u16=uint16_t;
using u32=uint32_t;
using u64=uint64_t;


///++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
///++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
///++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

namespace Param{
    bool yama=true; //trueなら焼きなましが山登りになる
    constexpr float startTemp=5e-2; //最初の温度
    constexpr float endTemp=0.0; //最後の温度
    constexpr int YAKI_TIME_LIMIT=4500;
}

void FastIO(){
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout << fixed << setprecision(20);
}


struct Xor32{
    using u32=uint32_t;
 
    u32 x=1234567;
    inline u32 rnd_make(){
        x^=(x<<13);
        x=x^(x>>17);
        return x=x^(x<<5);
    }
    inline u32 operator()(){
        return rnd_make();
    }
    //[a,b]
    inline int operator()(int a,int b){
        int dis=b-a+1;
        int add=rnd_make()%dis;
        return a+add;
    }
};

struct Timer{
    chrono::high_resolution_clock::time_point st;
    Timer(){start();}
    void start(){
        st=chrono::high_resolution_clock::now();
    }
    int span(){
        auto now=chrono::high_resolution_clock::now();
        return chrono::duration_cast<chrono::milliseconds>(now-st).count();
    }
};
Timer TIME;
Xor32 Rand;
//http://gasin.hatenadiary.jp/entry/2019/09/03/162613
struct yakinamashi{
    const float startTemp=Param::startTemp; //差の最大値(あくまでも参考)
    const float endTemp=Param::endTemp; //差の最小値(あくまでも参考)
    const float endTime=Param::YAKI_TIME_LIMIT;
    float temp=startTemp;
    yakinamashi(){}
    inline bool operator()(float plus){
        if(plus>0) return true;
        if(Param::yama) return false;
        temp=startTemp+(endTemp-startTemp)*TIME.span()/endTime;
        float prob=exp(plus/temp);
        if(prob>float(Rand()&mask(30))/mask(30)) return true;
        else return false;
    }
};
yakinamashi Yaki;

//https://koturn.hatenablog.com/entry/2018/08/02/000000
template<typename T, std::size_t N, std::size_t... Extents>
struct extents_expander
  : extents_expander<std::array<T, N>, Extents...>
{};  // struct extents_expander
 
template<typename T, std::size_t N>
struct extents_expander<T, N>
{
  using type = std::array<T, N>;
};  // struct extents_expander
 
template<typename T, std::size_t... Extents>
struct ndarray_helper
{
  using type = typename extents_expander<T, Extents...>::type;
};  // struct ndarray_helper
 
template<typename T, std::size_t N, std::size_t... Extents>
struct ndarray_helper<T[N], Extents...>
{
  using type = typename ndarray_helper<T, N, Extents...>::type;
};  // struct ndarray_helper
 
template<typename T>
using NdArray = typename ndarray_helper<T>::type;

struct is_range_impl
{
  template<typename T>
  static auto
  check(T&& obj) -> decltype(std::begin(obj), std::end(obj), std::true_type{});
 
  template<typename T>
  static auto
  check(...) -> std::false_type;
};  // struct is_range_impl
 
template<typename T>
class is_range :
  public decltype(is_range_impl::check<T>(std::declval<T>()))
{};  // class is_range
 
 
template<
  typename R,
  typename T,
  typename std::enable_if<
    is_range<R>::value && !is_range<typename std::iterator_traits<decltype(std::begin(std::declval<R>()))>::value_type>::value,
    std::nullptr_t
  >::type = nullptr
>
static inline void
fill(R&& range, T&& value) noexcept
{
  std::fill(std::begin(range), std::end(range), std::forward<T>(value));
}
 
 
template<
  typename R,
  typename T,
  typename std::enable_if<
    is_range<R>::value && is_range<typename std::iterator_traits<decltype(std::begin(std::declval<R>()))>::value_type>::value,
    std::nullptr_t
  >::type = nullptr
>
static inline void
fill(R&& range, T&& value) noexcept
{
  for (auto&& e : range) {
    fill(std::forward<decltype(e)>(e), std::forward<T>(value));
  }
}

#define Fill(v,from,to,value) for(int s=from;s<to;s++) v[s]=value
#define FillParity(v,from,to,parity,value) for(int s=from;s<to;s++) if(s%2==parity) v[s]=value

///++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
///++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
///++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

//https://atcoder.jp/contests/asprocon9/submissions/34659956
#ifndef OPTUNA 
#define REGIST_PARAM(name, type, defaultValue) constexpr type name = defaultValue
#else
#define REGIST_PARAM(name, type, defaultValue) type name = defaultValue
#endif

//parameterは極力ここに置く
namespace OP{

};

void solve(){

}
 
int main(const int argc,const char** argv){
#ifdef OPTUNA
    if(argc!=1){
        OP::min_all=stoi(argv[1]);
        OP::kyuzitu=stoi(argv[2]);
        cerr<<argv[1]<<endl;
    }
#endif
    FastIO();
    int T=1;
    //cin>>T;
    while(T--) solve();
}