/*
マラソンマッチテンプレート
todo:

*/
#define NDEBUG

//#define ONLINE_JUDGE
#ifndef ONLINE_JUDGE
//#define OPTUNA
#endif

#ifdef ONLINE_JUDGE
#define NDEBUG
#endif

#pragma GCC target("avx2")
#pragma GCC optimize("O3")
#pragma GCC optimize("unroll-loops")
#include<bits/stdc++.h>
using namespace std;
using ll=long long int;
//using Int=__int128;
#define mask(x) ((1LL<<x)-1)
#define ALL(A) A.begin(),A.end()
template<typename T1,typename T2> bool chmin(T1 &a,T2 b){if(a<=b)return 0; a=b; return 1;}
template<typename T1,typename T2> bool chmax(T1 &a,T2 b){if(a>=b)return 0; a=b; return 1;}
template<typename T> int bitUP(T x,int a){return (x>>a)&1;}
//← ↑ →　↓ 
int dh[4]={0,-1,0,1}, dw[4]={-1,0,1,0};
//上から時計回り
//int dx[8]={0,1,1,1,0,-1,-1,-1}, dy[8]={1,1,0,-1,-1,-1,0,1};
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
    //[a,b)
    inline int operator()(int a,int b){
        int dis=b-a;
        int add=rnd_make()%dis;
        return a+add;
    }
    //[0,b)
    inline int operator()(int b){
        return rnd_make()%b;
    }
};

struct Xor64{
    u64 x=1234567;
    inline u64 rnd_make(){
    	x ^= x << 13;
    	x ^= x >> 7;
    	x ^= x << 17;
    	return x;
    }
    inline u64 operator()(){
        return rnd_make();
    }
};

struct Timer{
    chrono::high_resolution_clock::time_point st;
    Timer(){start();}
    void start(){
        st=chrono::high_resolution_clock::now();
    }
    int span()const{
        auto now=chrono::high_resolution_clock::now();
        return chrono::duration_cast<chrono::milliseconds>(now-st).count();
    }
};
struct TestTimer{
    chrono::high_resolution_clock::time_point st;
    unordered_map<string,ll> sum_time;
    unordered_map<string,chrono::high_resolution_clock::time_point> start_time;

    TestTimer(){}
    void start(const string &s){
#ifndef ONLINE_JUDGE
        start_time[s]=chrono::high_resolution_clock::now();
#endif
    }
    void end(const string &s){
#ifndef ONLINE_JUDGE
        auto now=chrono::high_resolution_clock::now();
        sum_time[s]+=chrono::duration_cast<chrono::nanoseconds>(now-start_time[s]).count();
#endif
    }
    void output()const{
#ifndef ONLINE_JUDGE
        for(auto m:sum_time){
            cerr<<m.first<<": "<<m.second/1e6<<"ms"<<endl;
        }
#endif
    }
};
struct TestCounter{
    unordered_map<string,ll> cnt;

    TestCounter(){}

    void count(const string &s){
#ifndef ONLINE_JUDGE
        cnt[s]++;
#endif
    }
    void output()const{
#ifndef ONLINE_JUDGE
        for(auto m:cnt){
            cerr<<m.first<<": "<<m.second<<endl;
        }
#endif
    }    
};
Timer TIME;
Xor32 Rand32;
Xor64 Rand64;
TestTimer testTimer;
TestCounter testCounter;

//https://atcoder.jp/contests/asprocon9/submissions/34659956
template<class T,int CAPACITY>
class DynamicArray{
public:
    array<T,CAPACITY> array_={};
    int size_=0;

    DynamicArray(){}

    DynamicArray(int n){
        resize(n);
    }

    void push_back(const T &e){
        array_[size_++]=e;
    }
    inline T& operator[](int index){
        return array_[index];
    }
	inline const T& operator[](int index) const {
		return array_[index];
	}
    inline int size()const{
        return size_;
    }
    inline T& back(){
        return array_[size_-1];
    }
	inline auto begin() -> decltype(array_.begin()) {
		return array_.begin();
	}

	inline auto end() -> decltype(array_.begin()) {

		return array_.begin() + size_;
	}

	inline auto begin() const -> decltype(array_.begin()) {
		return array_.begin();
	}

	inline auto end() const -> decltype(array_.begin()) {

		return array_.begin() + size_;
	}
    inline void resize(int new_size){
        size_=new_size;
    }
    void operator=(const DynamicArray &e){
        for(int i=0;i<e.size_;i++){
            array_[i]=e[i];
        }
        size_=e.size_;
    }
    void clear(){
        size_=0;
    }
    //O(1)
    //末尾と入れ替える。順序が保持されないが高速
    void swap_remove(int idx){
        array_[idx]=array_[size_-1];
        size_--;
    }
    //O(size)
    //順序を気にしない場合、swap_removeの方がいい
    void remove(int idx){
        for(int i=idx;i<size_-1;i++){
            array_[i]=array_[i+1];
        }
        size_--;
    }
    void fill(T x){
        for(int i=0;i<size_;i++){
            array_[i]=x;
        }
    }
};

//[0,n)の集合を管理
//値の追加・削除・存在確認: O(1)
//空間計算量: O(n)
//重複は許さない
//https://topcoder-tomerun.hatenablog.jp/entry/2021/06/12/134643
template<int CAPACITY>
struct IntSet{
    DynamicArray<int,CAPACITY> set_;
    array<int,CAPACITY> pos_;

    IntSet(){
        fill(set_,-1);
    }
    void insert(int v){
        if(pos_[v]!=-1) return;
        pos_[v]=set_.size();
        set_.push_back(v);
    }

    void remove(int v){
        assert(pos_[v]!=-1);
        set_[pos_[v]]=set_.back();
        set_.pop_back();
        pos_[v]=-1;
    }

    bool contains(int v)const{
        return pos_[v]!=-1;
    }

    int size()const{
        return set_.size();
    }

    int random()const{
        return set_[Rand32(set_.size())];
    }

    int random_extract(){
        int v=set_[Rand32(set_.size())];
        remove(v);
        return v;
    }
};

//https://atcoder.jp/contests/asprocon9/submissions/34659956
#ifndef OPTUNA 
#define REGIST_PARAM(name, type, defaultValue) constexpr type name = defaultValue
#else
#define REGIST_PARAM(name, type, defaultValue) type name = defaultValue
#endif


///++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
///++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
///++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

namespace OP{
    REGIST_PARAM(startTemp,float, 30000);
    REGIST_PARAM(endTemp,float,0);
    REGIST_PARAM(yama,bool,false);
    REGIST_PARAM(TIME_END,int,4960);
};


//http://gasin.hatenadiary.jp/entry/2019/09/03/162613
struct yakinamashi{
    const float startTemp=OP::startTemp; //差の最大値(あくまでも参考)
    const float endTemp=OP::endTemp; //差の最小値(あくまでも参考)
    const float endTime=OP::TIME_END;
    float temp=startTemp;
    yakinamashi(){}
    inline bool operator()(float plus){
        if(plus>0) return true;
        if(OP::yama) return false;
        if(TIME.span()>=endTime) temp=endTemp;
        else temp=startTemp+(endTemp-startTemp)*TIME.span()/endTime;
        float prob=exp(plus/temp);
        if(prob>float(Rand32()&mask(30))/mask(30)) return true;
        else return false;
    }
};
yakinamashi Yaki;

///++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
///++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
///++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

//要素数n 初期値x
template<typename T>
inline vector<T> vmake(size_t n,T x){
	return vector<T>(n,x);
}

//a,b,c,x data[a][b][c] 初期値x
template<typename... Args>
auto vmake(size_t n,Args... args){
	auto v=vmake(args...);
	return vector<decltype(v)>(n,move(v));
}

template<class T,class U>
T linear_function(U x,U start_x,U end_x,T start_value,T end_value){
    if(x>=end_x) return end_value;
    return start_value+(end_value-start_value)*(x-start_x)/(end_x-start_x);
}

const int MAX_H=10,MAX_W=10;
bool OutGrid(int h,int w){
    return h<0 or w<0 or h>=MAX_H or w>=MAX_W;
}

template<class T,int CAPACITY>
struct GlobalArray{
    array<T,CAPACITY> array_;
    int used_size_=0;

    int allocation(int size){
        int start=used_size_;
        used_size_+=size;
        return start;
    }
    inline T& operator[](int index){
        return array_[index];
    }
	inline const T& operator[](int index) const {
		return array_[index];
	}
	inline auto begin() -> decltype(array_.begin()) {
		return array_.begin();
	}

	inline auto end() -> decltype(array_.begin()) {
		return array_.begin() + used_size_;
	}

	inline auto begin() const -> decltype(array_.begin()) {
		return array_.begin();
	}

	inline auto end() const -> decltype(array_.begin()) {
		return array_.begin() + used_size_;
	}

};

template<class T,int CAPACITY>
struct LocalArray{
    GlobalArray<T,CAPACITY> &array_;
    int start_;
    int size_;
    LocalArray(GlobalArray<T,CAPACITY> &array,int size):array_(array),size_(size){
        start_=array_.allocation(size_);
    }
    void push_back(const T &e){
        assert(array_.used_size_==start_+size_);
        array_[start_+size_]=e;
        size_++;
        array_.used_size_++;
    }
    inline T& operator[](int index){
        return array_[start_+index];
    }
	inline const T& operator[](int index) const {
		return array_[start_+index];
	}
    inline int size()const{
        return size_;
    }
    inline T& back(){
        return array_[start_+size_-1];
    }
	inline auto begin() -> decltype(array_.begin()) {
		return array_.begin()+start_;
	}

	inline auto end() -> decltype(array_.begin()) {
		return array_.begin() + start_ + size_;
	}

	inline auto begin() const -> decltype(array_.begin()) {
		return array_.begin()+start_;
	}

	inline auto end() const -> decltype(array_.begin()) {
		return array_.begin() + start_ + size_;
	}
};

void local_array_test(){
    GlobalArray<int,100> global_array;
    LocalArray<int,100> local_array(global_array,10);
    LocalArray<int,100> local_array2(global_array,0);
    local_array2.push_back(1);
    local_array2.push_back(2);
    for(int i:local_array){
        cout<<i<<endl;
    }
    cout<<"#####"<<endl;
    for(int i:local_array2){
        cout<<i<<endl;
    }
}



void solve(){
    
}
 
int main(const int argc,const char** argv){
#ifndef ONLINE_JUDGE
    if(argc!=1){
        /*OP::min_all=stoi(argv[1]);
        OP::kyuzitu=stoi(argv[2]);*/
        cerr<<argv[1]<<endl;
    }
#endif
    FastIO();
    int T=1;
    //cin>>T;
    while(T--) solve();
}
