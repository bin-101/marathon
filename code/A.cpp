/*
マラソンマッチテンプレート
todo:

*/
#define NDEBUG

#define ONLINE_JUDGE
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
//→　↓　←　↑ 
//int dh[4]={0,1,0,-1}, dw[4]={1,0,-1,0};
//上から時計回り
int dx[8]={0,1,1,1,0,-1,-1,-1}, dy[8]={1,1,0,-1,-1,-1,0,1};
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

//https://atcoder.jp/contests/ahc005/submissions/24829813
//https://future-architect.github.io/articles/20211201a/
//improvedを使う必要があるのか？
//inf_で足し算をしてしまうとまずい
//inf_で初期化をする
template<class T,int SIZE>
class TSP{
public:
    DynamicArray<DynamicArray<T,SIZE>,SIZE> distance_;
    DynamicArray<int,SIZE> ans_;
    int n_;
    DynamicArray<int,SIZE> reuse_;//使いまわす
    DynamicArray<int,SIZE> perm_;//使いまわす　常に[0,n_-1]の順列
    DynamicArray<DynamicArray<int,SIZE>,SIZE> order_; //[i][j]: 頂点iからj番目に近い頂点
    DynamicArray<int,SIZE> place_; //ans_[place_[i]]=i
    T inf_;

    TSP():inf_(numeric_limits<T>::max()){

    }

    TSP(int n):n_(n),inf_(numeric_limits<T>::max()){
        distance_.resize(n_);
        for(int i=0;i<n_;i++){
            distance_[i].resize(n_);
        }
        ans_.resize(n_);
        reuse_.resize(n_);
        perm_.resize(n_);
        iota(perm_.begin(), perm_.end(),0);
        
        order_.resize(n_);
        place_.resize(n_);
        for(int i=0;i<n_;i++){
            order_[i].resize(n_);
            iota(order_[i].begin(),order_[i].end(),0);
            order_[i].swap_remove(i);
        }
    }
    void init(int n){
        n_=n;
        assert(n_);
        distance_.resize(n_);
        for(int i=0;i<n_;i++){
            distance_[i].resize(n_);
        }
        ans_.resize(n_);
        reuse_.resize(n_);
        perm_.resize(n_);
        iota(perm_.begin(), perm_.end(),0);
        
        order_.resize(n_);
        place_.resize(n_);
        for(int i=0;i<n_;i++){
            order_[i].resize(n_);
            iota(order_[i].begin(),order_[i].end(),0);
            order_[i].swap_remove(i);
        }
        assert(n_);
    }

    void set(int from,int to,T dist){
        distance_[from][to]=dist;
    }
    //O(N^2logN)
    //order_を構築
    void build_order(){
        for(int i=0;i<n_;i++){
            sort(order_[i].begin(),order_[i].end(),
                [&](int j,int k){
                    return dist(i,j)<dist(i,k);
                });
        }
    }
    //[from,to)
    void build_place(int from,int to){
        for(int i=from;i<to;i++){
            place_[ans_[i]]=i;
        }
    }
    //局所最適まで探索
    //変化量を返す
    T local_search(T now_dist){
        T diff=0;
        int stage=Rand32(2);
        bool pre_fault=0;
        while(true){
            bool updated=false;
            for(int a1=0;a1<n_;a1++){
                int a2=next(a1);
                T d12=dist(ans_[a1],ans_[a2]);
                int v1=ans_[a1];
                int v2=ans_[a2];
                //ans_[a1]から近い順にイテレート
                if(stage==0){
                    for(int o1=0;o1<n_;o1++){
                        int v3=order_[v1][o1];
                        if(d12<=dist(v1,v3)) break;
                        //cerr<<a1<<" "<<place_[v3]<<endl;
                        T change_dist=calc_two_opt_score(a1,place_[v3]);
                        if(change_dist<0){
                            diff+=change_dist;
                            //cerr<<change_dist<<endl;
                            swap_two_opt(a1,place_[v3]);
                            updated=true;
                            assert(is_perm(ans_));

                            assert(calc_sum_dist(ans_)==now_dist+diff);

                        }
                    }
                }
                //ans_[a2]から近い順にイテレート
                if(stage==1){
                    for(int o2=0;o2<n_;o2++){
                        int v4=order_[v2][o2];
                        if(d12<=dist(v2,v4)) break;
                        T change_dist=calc_two_opt_score(a1,prev(place_[v4]));
                        //cerr<<change_dist<<" "<<a1<<" "<<prev(place_[v4])<<endl;
                        if(change_dist<0){
                            diff+=change_dist;
                            swap_two_opt(a1,prev(place_[v4]));
                            updated=true;
                            assert(is_perm(ans_));
                            assert(calc_sum_dist(ans_)==now_dist+diff);
                        }
                    }
                }
            }
            stage^=1;
            if(not updated){
                if(pre_fault) return diff;
                pre_fault=true;
            }else{
                pre_fault=false;
            }

        }
    }
    void solve(int time_limit){
        if(n_<10){
            solve_perm(time_limit);
            return;
        }
        assert(n_);
        DynamicArray<int,SIZE> best_perm;
        T now_dist=greedy();
        assert(is_perm(ans_));
        assert(calc_sum_dist(ans_)==now_dist);
        best_perm=ans_;
        T best_dist=now_dist;
        //cerr<<best_dist<<endl;

        while(TIME.span()<time_limit){
            now_dist+=local_search(now_dist);
            assert(is_perm(ans_));
            //cerr<<now_dist<<endl;
            assert(calc_sum_dist(ans_)==now_dist);
            //cerr<<best_dist<<endl;
            if(now_dist<best_dist){
                best_perm=ans_;
                best_dist=now_dist;
                testCounter.count("tsp_update");
            }
            now_dist+=double_bridge();
            assert(is_perm(ans_));
            //cerr<<now_dist<<endl;
            //cerr<<calc_sum_dist(ans_)<<endl;
            assert(calc_sum_dist(ans_)==now_dist);
            testCounter.count("tsp_try");
            assert(true_place());
        }

        ans_=best_perm;
    }
    bool true_place(){
        for(int i=0;i<n_;i++){
            if(ans_[place_[i]]!=i) return false;
        }
        return true;
    }

    //O(n!)
    void solve_perm(int time_limit){
        T best_score=numeric_limits<T>::max();

        iota(reuse_.begin(),reuse_.end(),0);

        do{
            T sum_dist=calc_sum_dist(reuse_);
            if(sum_dist<best_score){
                best_score=sum_dist;
                ans_=reuse_;
            }
        }while(next_permutation(reuse_.begin(),reuse_.end()) and TIME.span()<time_limit);
    }

    T greedy(){
        assert(n_);
        reuse_.fill(0);
        ans_[0]=0;
        reuse_[0]=1;
        int now=0;
        T sum_dist=0;
        for(int i=1;i<n_;i++){
            int to=-1;
            for(int o1=0;o1<n_;o1++){
                int v2=order_[now][o1];
                if(not reuse_[v2]){
                    to=v2;
                    break;
                }
            }
            assert(to!=-1);
            sum_dist+=dist(now,to);
            reuse_[to]=1;
            ans_[i]=to;
            now=to;
        }
        build_place(0,n_);
        return sum_dist+dist_ans(n_-1,0);
    }
    T double_bridge(){
        //cerr<<n_<<endl;
        random_choose(4);
        //cerr<<n_<<endl;

        T change_dist=0;

        for(int i=0;i<4;i++){
            change_dist-=dist(ans_[reuse_[i]],ans_[next(reuse_[i])]);
            change_dist+=dist(ans_[reuse_[i]],ans_[next(reuse_[(i+2)%4])]);
        }
        //assert(is_perm(ans_));

        copy(ans_.begin(),ans_.begin()+reuse_[0]+1,perm_.begin());
        copy(ans_.begin()+reuse_[2]+1,ans_.begin()+reuse_[3]+1,perm_.begin()+reuse_[0]+1);
        copy(ans_.begin()+reuse_[1]+1,ans_.begin()+reuse_[2]+1,perm_.begin()+reuse_[0]+reuse_[3]-reuse_[2]+1);
        copy(ans_.begin()+reuse_[0]+1,ans_.begin()+reuse_[1]+1,perm_.begin()+reuse_[0]+reuse_[3]-reuse_[1]+1);
        copy(ans_.begin()+reuse_[3]+1,ans_.end(),perm_.begin()+reuse_[3]+1);

        ans_=perm_;
        assert(is_perm(ans_));
        build_place(0,n_);

        return change_dist;
    }

    inline T dist(int i,int j){
        return distance_[i][j];
    }
    inline T dist(int i){
        return distance_[ans_[i]][ans_[next(i)]];
    }
    inline T dist_ans(int a1,int a2){
        return dist(ans_[a1],ans_[a2]);
    }

    T dist(int i,int j,const DynamicArray<int,SIZE> &perm){
        return distance_[perm[i]][perm[j]];
    }

    T calc_sum_dist(const DynamicArray<int,SIZE> &perm){
        T sum_dist=dist(perm[n_-1],perm[0]);
        for(int i=1;i<n_;i++){
            sum_dist+=dist(perm[i-1],perm[i]);
        }
        return sum_dist;
    }
    //reuse_[0:k-1]に相異なるn_未満の自然数を昇順に代入
    void random_choose(int k){
        assert(k<=n_);
        for(int i=0;i<k;i++){
            int idx=i+Rand32(n_-i);
            reuse_[i]=perm_[idx];
            swap(perm_[i],perm_[idx]);
        }
        sort(reuse_.begin(),reuse_.begin()+k);
    }
    //debug用 O(n_)
    bool is_perm(const DynamicArray<int,SIZE> &v){
        reuse_.fill(0);
        for(int i=0;i<n_;i++){
            if(reuse_[v[i]]) return false;
            reuse_[v[i]]=true;
        }
        return true;
    }
    int next(int a){
        return a==n_-1?0:a+1;
    }
    int prev(int a){
        return a==0?n_-1:a-1;
    }
    //ans_[a1]-ans_[a1+1] ans[a2]_-ans_[a2+1]
    T calc_two_opt_score(int a1,int a2){
        if(a1==a2) return 0;
        T res=-dist(a1)-dist(a2);
        res+=dist_ans(a1,a2)+dist_ans(next(a1),next(a2));
        return res;
    }
    void swap_two_opt(int a1,int a2){
        assert(a1!=a2);
        if(a1>a2){
            swap(a1,a2);
        }
        reverse(ans_.begin()+a1+1,ans_.begin()+a2+1);
        build_place(a1+1,a2+1);
        //cerr<<-1<<endl;
    }
};

constexpr int MAX_SIZE=2500+1;
TSP<ll,2500> tsp;
void solve(){
    int n=0;
    vector<int> id(MAX_SIZE);
    vector<double> x(MAX_SIZE),y(MAX_SIZE);
    while(cin>>id[n]){
        cin>>x[n]>>y[n];
        n++;
    }
    tsp.init(n);
    assert(tsp.n_);
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            tsp.set(i,j,sqrt((x[i]-x[j])*(x[i]-x[j])+(y[i]-y[j])*(y[i]-y[j]))*100);
        }
    }
    assert(tsp.n_);
    tsp.build_order();
    assert(tsp.n_);
    tsp.solve(60*1000);

    for(int i=0;i<n;i++){
        cout<<id[tsp.ans_[i]]<<endl;
    }
    cerr<<tsp.calc_sum_dist(tsp.ans_)<<endl;
    testCounter.output();
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
