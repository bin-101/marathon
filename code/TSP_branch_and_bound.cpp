/*
TSPに対する分枝限定法
verify: https://atcoder.jp/contests/tessoku-book/submissions/46390189
*/
//#define NDEBUG

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
#define derr if(0) cerr
#define all(c) ((c).begin()), ((c).end())
#define dout if(1)cout
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
        assert(size_>=0 and size_<CAPACITY);
        array_[size_++]=e;
    }
    T pop_back(){
        assert(size_>=1 and size_<=CAPACITY);
        size_--;
        return array_[size_];
    }
    inline T& operator[](int index){
        assert(index>=0 and index<size_);
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
    friend ostream &operator<<(ostream &os,const DynamicArray<T,CAPACITY> &a){
        for(int i=0;i<a.size();i++){
            if(i) cout<<" ";
            cout<<a[i];
        }
    	return os;
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
//vector cout
template<typename T>
inline ostream &operator<<(ostream &os,const vector<T> &v) {
    bool sp=true;
    if(string(typeid(T).name())=="c") sp=false;
    for(size_t i=0;i<v.size();i++){
        if(i and sp) os<<" ";
        os<<v[i];
    }
    return os;
}
//vector<vector> cout
template<typename T>
inline ostream &operator<<(ostream &os,const vector<vector<T>> &v) {
    for(size_t i=0;i<v.size();i++){
        os<<v[i];
        if(i+1!=v.size()) os<<"\n";
    }
    return os;
}

template<class T,class U>
T linear_function(U x,U start_x,U end_x,T start_value,T end_value){
    if(x>=end_x) return end_value;
    return start_value+(end_value-start_value)*(x-start_x)/(end_x-start_x);
}

struct UnionFind{
    vector<int> par;
    UnionFind(int N) : par(N,-1){}
    
    //rootを探す
    int find(int x){
        if(par[x]<0) return x;
        else return par[x]=find(par[x]);
    }
    //集合の要素数
    int usize(int x) {return -par[find(x)];}
    //xとyが繋がっていたらfalseを返す
    bool unite(int x,int y){
        x=find(x);
        y=find(y);
        if(x==y) return false;
        if(usize(x)<usize(y)) swap(x,y);
        par[x]+=par[y];
        par[y]=x;
        return true;
    }

    bool same(int x,int y) {return find(x)==find(y);}

    void clear(){
        par.assign(par.size(),-1);
    }
};
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

//https://gist.github.com/spaghetti-source/c31558e07adcd2ced2d6
//chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/http://akira.ruc.dk/~keld/research/LKH/LKH-2.0/DOC/LKH_REPORT.pdf
//n=2のときバグるが、辺を2回張ったら大丈夫だった
template<class T,int SIZE>
class TSPBB{
public:
    struct Edge{
        int from,to;
        T cost;
    };

    int n_;
    int m_;
    T inf_;
    T best_dist_;
    vector<bool> best_solution_; //size:m 使う辺をtrue
    vector<vector<int>> adj_;
    vector<Edge> edges;

    void add_edge(int from,int to,T cost){
        adj_[from].push_back(edges.size());
        adj_[to].push_back(edges.size());
        edges.push_back({from,to,cost});
        m_++;
    }
    void output(int start=0){
        vector<bool> used(n_,false);
        used[start]=true;
        vector<int> tour={start};
        while(tour.size()<n_){
            for(int i=0;i<m_;i++){
                if(best_solution_[i]==false) continue;
                if(tour.back()==edges[i].from and used[edges[i].to]==false){
                    tour.push_back(edges[i].to);
                }else if(tour.back()==edges[i].to and used[edges[i].from]==false){
                    tour.push_back(edges[i].from);
                }
                used[tour.back()]=true;
            }
        }
        cout<<tour<<endl;
    }

    struct State{
        vector<bool> used_edge_1tree_;
        vector<int> state_edge_;//-1:不使用確定 0:未確定 1:使用確定
        vector<T> penalty_;
        vector<int> tree1_degree_;
        vector<int> solution_degree_;
        T lower_bound_;
        State(int n,int m,T inf):used_edge_1tree_(m),state_edge_(m),
            penalty_(n),tree1_degree_(n),solution_degree_(n),lower_bound_(inf){}

        bool operator<(const State &s)const{
            if(lower_bound_==s.lower_bound_){
                int num_dec=0;
                for(int state:state_edge_){
                    num_dec+=abs(state);
                }
                for(int state:s.state_edge_){
                    num_dec-=abs(state);
                }
                return num_dec>0;
            }
            return lower_bound_>s.lower_bound_;
        }
        //O(n_)
        bool is_1tree_tour(){
            for(int d:tree1_degree_){
                if(d!=2) return false;
            }
            return true;
        }
    };

    TSPBB(int n):n_(n),m_(0),inf_(numeric_limits<T>::max()),best_dist_(inf_),adj_(n),uni_(n){
    
    }

    bool is_tour(const vector<bool> &solution)const{
        vector<int> degree(n_,0);
        for(int eid=0;eid<m_;eid++){
            if(solution[eid]){
                degree[edges[eid].from]++;
                degree[edges[eid].to]++;
            }
        }
        for(int v=0;v<n_;v++){
            if(degree[v]!=2) return false;
        }
        return true;
    }
    T calc_sum_dist(const vector<bool> &solution)const{
        T sum_dist=0;
        for(int eid=0;eid<m_;eid++){
            if(solution[eid]){
                sum_dist+=edges[eid].cost;
            }
        }
        return sum_dist;
    }
    //O(nm)
    void set(DynamicArray<int,SIZE> &solution){

        auto used=vmake(n_,n_,false);
        for(int i=0;i<n_;i++){
            int to=i+1;
            if(to==n_) to=0;
            used[solution[i]][solution[to]]=true;
            used[solution[to]][solution[i]]=true;
        }
        best_solution_.resize(m_);
        for(int i=0;i<m_;i++){
            auto &e=edges[i];
            best_solution_[i]=used[e.from][e.to];
        }
        best_dist_=calc_sum_dist(best_solution_);
    }
    DynamicArray<int,SIZE*(SIZE-1)/2> order_edge_;
    UnionFind uni_;

    //update: tree1_degree_・used_edge_1tree_・lower_bound_
    void calc_minimum_1tree(State &s){
        testCounter.count("calc_1tree");
        s.tree1_degree_.assign(n_,0);
        s.used_edge_1tree_.assign(m_,false);
        s.lower_bound_=0;

        auto edge_cost=[&](int i){
            return edges[i].cost+s.penalty_[edges[i].from]+s.penalty_[edges[i].to];
        };

        sort(order_edge_.begin(),order_edge_.end(),
            [&](int i,int j){
                if(s.state_edge_[i]!=s.state_edge_[j]) return s.state_edge_[i]>s.state_edge_[j];
                return edge_cost(i)<edge_cost(j);
            });
        uni_.clear();
        int cnt_used_edge=0;
        for(int edge_id:order_edge_){
            const Edge &e=edges[edge_id];
            bool ok=false;
            if(s.state_edge_[edge_id]==1 and uni_.same(e.from,e.to)){
                s.lower_bound_=inf_;
                return;
            }
            if(s.state_edge_[edge_id]==-1){
                break;
            }
            if(s.state_edge_[edge_id]==0 and (s.solution_degree_[e.from]>=2 or s.solution_degree_[e.to]>=2)) continue;
            if(e.from==0 or e.to==0){
                if(s.tree1_degree_[0]<2) ok=true;
            }else if(uni_.unite(e.from,e.to)) ok=true;
            
            if(ok){
                s.used_edge_1tree_[edge_id]=true;
                s.tree1_degree_[e.from]++;
                s.tree1_degree_[e.to]++;
                s.lower_bound_+=edge_cost(edge_id);
                cnt_used_edge++;
            }
            if(cnt_used_edge==n_) break;
        }
        if(cnt_used_edge<n_){
            s.lower_bound_=inf_;
        }
        return;
    }
    void calc_upper_bound(State &s){

    }
    //minimum 1-tree
    //now_solution_のを計算
    //prim法をする
    //tourならtrueを返す
    //todo: tour作成不可能ならinfを返したい
    bool calc_lower_bound(State &s){
        testCounter.count("calc_lower_bound");
        //s.penalty_.assign(n_,0);
        calc_minimum_1tree(s);
        if(s.is_1tree_tour()){
            return true;
        }
        if(s.lower_bound_==inf_) return false;
        State best_state=s;
        vector<int> pre_degree=s.tree1_degree_;

        bool initial_phase=true;
        for(int iter=0;iter<1;iter++){
            T step_size=10;
            s=best_state;
            int period=n_/2;
            while(step_size>=10){
                T step_size4=step_size*0.4;
                T step_size6=step_size*0.6;
                //次数をもとにpenalty_を更新
                for(int v=0;v<n_;v++){
                    T plus=(s.tree1_degree_[v]-2)*step_size6+(pre_degree[v]-2)*step_size4;
                    s.penalty_[v]+=plus;
                    //derr<<" s.penalty_"<<v<<" "<<s.penalty_[v]<<endl;
                }
                pre_degree=s.tree1_degree_;
                calc_minimum_1tree(s);
                if(s.is_1tree_tour()) return true;
                if(s.lower_bound_>=best_dist_) return false;
                //derr<<best_state.lower_bound_<<" "<<s.lower_bound_<<endl;
                if(best_state.lower_bound_<s.lower_bound_){
                    best_state=s;
                    step_size*=2;
                    //derr<<step_size<<" "<<best_state.lower_bound_<<endl;
                }else{
                    initial_phase=false;
                    step_size/=2;
                }
            }
        }
        s=best_state;
        return false;
    }
    void push(priority_queue<State> &que,State &s){
        if(calc_lower_bound(s)){
            if(chmin(best_dist_,s.lower_bound_)){
                best_solution_=s.used_edge_1tree_;
            }
        }else if(s.lower_bound_<best_dist_){
            que.push(s);
        }
    }

    void branch_and_bound(){
        order_edge_.resize(m_);
        //cerr<<order_edge_.size()<<endl;
        iota(order_edge_.begin(),order_edge_.end(),0);

        priority_queue<State> state_que;
        //stack<State> state_que;
        
        State s(n_,m_,inf_);
        push(state_que,s);

        while(state_que.size()){
            s=state_que.top(); state_que.pop();
            derr<<endl;
            derr<<"new_node"<<endl;
            derr<<" lower_bound:"<<s.lower_bound_<<" "<<"upper_bound:"<<best_dist_<<endl;
            if(s.lower_bound_>=best_dist_) continue;
            testCounter.count("node");
            for (int e = 0; e < m_; ++e) {
                if (s.state_edge_[e] == -1)  
                    derr << "  " << edges[e].from << " " << edges[e].to << " is excluded" << endl;
                if (s.state_edge_[e] ==1)  
                    derr << "  " << edges[e].from << " " << edges[e].to << " is included" << endl;
                if (s.used_edge_1tree_[e]) 
                    derr << "  " << edges[e].from << " " << edges[e].to << " is used" << endl;
            }       

            int branch_vertex=-1;
            vector<int> branch_vs(n_);
            iota(branch_vs.begin(),branch_vs.end(),0);
            vector<int> num_res_edges(n_);
            for(int v=0;v<n_;v++){
                for(int eid:adj_[v]){
                    if(s.state_edge_[eid]==0) num_res_edges[v]++;
                }
            }

            sort(branch_vs.begin(),branch_vs.end(),
                [&](int i,int j){
                    if(s.tree1_degree_[i]<=2 or s.tree1_degree_[j]<=2){
                        return s.tree1_degree_[i]>s.tree1_degree_[j];
                    }
                    if(s.solution_degree_[i]!=s.solution_degree_[j]){
                        return s.solution_degree_[i]>s.solution_degree_[j];
                    }
                    return num_res_edges[i]<num_res_edges[j];
                });

            /*for(int v=0;v<n_;v++){
                if(s.solution_degree_[v]==2){
                    if(s.tree1_degree_[v]!=2){
                        cerr<<s.tree1_degree_[v]<<endl;
                        assert(false);
                    }
                }
                if(s.tree1_degree_[v]>2){
                    branch_vertex=v;
                    break;
                }
            }*/

            branch_vertex=branch_vs[0];
            for(int v=0;v<n_;v++){
                if(num_res_edges[v]<3) branch_vertex=v;
            }
            assert(branch_vertex!=-1);
            derr << "  branching vertex " << branch_vertex << endl;

            //https://reader.elsevier.com/reader/sd/pii/0377221782900157?token=96C4ED129C5EE4A001288AC206E16564B422F0CE3CF60061899027E77B858C2B7BC8221AED9A8BCFCA3CCFDD94F42605&originRegion=us-east-1&originCreation=20221014035825
            //branching strategy
            //todo: ちゃんと読んで書く ソートする

            UnionFind uni(n_);
            for(int eid=0;eid<m_;eid++){
                if(s.state_edge_[eid]==1) uni.unite(edges[eid].from,edges[eid].to);
            }
            vector<int> branch_edges;
            for(int edge_id:adj_[branch_vertex]){
                auto &e=edges[edge_id];
                if(s.used_edge_1tree_[edge_id] and s.state_edge_[edge_id]==0 and uni.same(e.from,e.to)==false){
                    branch_edges.push_back(edge_id);
                }
            }
            sort(branch_edges.begin(),branch_edges.end(),
                [&](int i,int j){
                    auto &ei=edges[i];
                    auto &ej=edges[j];
                    return ei.cost+s.penalty_[ei.from]+s.penalty_[ei.to]<ej.cost+s.penalty_[ej.from]+s.penalty_[ej.to];
                });
            vector<array<int,2>> next_state;
            next_state.push_back({-1,0});
            next_state.push_back({1,-1});
            next_state.push_back({1,1});
            for(int I=0;I<3;I++){
                if(I==2 and s.solution_degree_[branch_vertex]==1) continue;
                State ns=s;
                for(int i=0;i<2;i++){
                    int edge_id=branch_edges[i];
                    ns.state_edge_[edge_id]=next_state[I][i];
                    if(next_state[I][i]==1){
                        ns.solution_degree_[edges[edge_id].from]++;
                        ns.solution_degree_[edges[edge_id].to]++;
                    }
                }
                push(state_que,ns);
            }
        }
    }
    //O(2^n m)
    T solve_dp(){
        assert(n_);
        auto dp=vmake((1<<n_),n_,inf_);
        dp[1][0]=0;
        for(int S=0;S<(1<<n_);S++){
            for(int v=0;v<n_;v++){
                if(dp[S][v]==inf_) continue;
                for(int eid:adj_[v]){
                    int to=-1;
                    if(edges[eid].from!=v) to=edges[eid].from;
                    if(edges[eid].to!=v) to=edges[eid].to;
                    if(bitUP(S,to)) continue;
                    chmin(dp[S|(1<<to)][to],dp[S][v]+edges[eid].cost);
                }
            }
        }
        //derr<<dp<<endl;
        T ret=inf_;
        for(int eid:adj_[0]){
            int S=(1<<n_)-1;
            int to=-1;
            if(edges[eid].from!=0) to=edges[eid].from;
            if(edges[eid].to!=0) to=edges[eid].to;
            //cerr<<dp[S][to]<<" "<<edges[eid].cost<<endl;
            if(dp[S][to]!=inf_) chmin(ret,dp[S][to]+edges[eid].cost);
        }
        return ret;
    }

    void solve(){
        branch_and_bound();
    }

};

constexpr int MAX_SIZE=150;
void solve(){
    int n=100,m;
    TSPBB<ll,MAX_SIZE> tsp(n);
    TSP<ll,MAX_SIZE> tsph(n);


    vector<double> x(n),y(n);
    for(int i=0;i<n;i++){
        //cin>>dummy;
        //cin>>x[i]>>y[i];
        x[i]=Rand32(1e3);
        y[i]=Rand32(1e3);
    };
    assert(tsp.n_);
    for(int i=0;i<n;i++){
        for(int j=i+1;j<n;j++){
            tsp.add_edge(i,j,sqrt((x[i]-x[j])*(x[i]-x[j])+(y[i]-y[j])*(y[i]-y[j]))*10000);
            tsph.set(i,j,sqrt((x[i]-x[j])*(x[i]-x[j])+(y[i]-y[j])*(y[i]-y[j]))*10000);
            tsph.set(j,i,sqrt((x[i]-x[j])*(x[i]-x[j])+(y[i]-y[j])*(y[i]-y[j]))*10000);

        }
    }
    assert(tsp.n_);

    tsph.solve(100);

    tsp.set(tsph.ans_);
    tsp.solve();

    tsp.output();
    
    /*int start=-1;
    for(int i=0;i<n;i++){
        if(tsp.best_solution_[i]==0) start=i;
    }
 
    for(int i=0;i<n;i++){
        cout<<tsp.best_solution_[(i+start)%n]+1<<endl;
    }
    cout<<1<<endl;*/

    assert(tsp.is_tour(tsp.best_solution_));
    ll dp=0;//tsp.solve_dp();
    //cerr<<g.held_karp()<<endl;
    cerr<<dp<<" "<<tsp.calc_sum_dist(tsp.best_solution_)<<endl;
    //assert(dp==tsp.calc_sum_dist(tsp.best_solution_));
    cerr<<TIME.span()<<"ms"<<endl;
    testCounter.output();
}

void solve_atcoder(){
    int n=15,m;
    cin>>n;
    TSPBB<ll,15> tsp(n);

    vector<double> x(n),y(n);
    for(int i=0;i<n;i++){
        //cin>>dummy;
        cin>>x[i]>>y[i];
        //x[i]=Rand32(1e5);
        //y[i]=Rand32(1e5);
    };
    assert(tsp.n_);
    for(int i=0;i<n;i++){
        for(int j=i+1;j<n;j++){
            tsp.add_edge(i,j,100000LL*sqrt((x[i]-x[j])*(x[i]-x[j])+(y[i]-y[j])*(y[i]-y[j])));
            if(n==2){
                tsp.add_edge(i,j,100000LL*sqrt((x[i]-x[j])*(x[i]-x[j])+(y[i]-y[j])*(y[i]-y[j])));
            }
        }
    }
    assert(tsp.n_);



    
    tsp.solve();

    //tsp.output();
    
    /*int start=-1;
    for(int i=0;i<n;i++){
        if(tsp.best_solution_[i]==0) start=i;
    }
 
    for(int i=0;i<n;i++){
        cout<<tsp.best_solution_[(i+start)%n]+1<<endl;
    }
    cout<<1<<endl;*/

    /*assert(tsp.is_tour(tsp.best_solution_));
    ll dp=0;//tsp.solve_dp();
    cerr<<dp<<" "<<tsp.calc_sum_dist(tsp.best_solution_)<<endl;
    //assert(dp==tsp.calc_sum_dist(tsp.best_solution_));
    cerr<<TIME.span()<<"ms"<<endl;
    testCounter.output();*/
    cout<<tsp.calc_sum_dist(tsp.best_solution_)/100000.0<<endl;

}

int main(const int argc,const char** argv){

    FastIO();
    solve_atcoder();
}
/*
10/14:18:00 n100:85260ms
85260ms
node: 15659
calc_lower_bound: 42555
*/
/*
4章(1)(2)
67846ms
node: 8411
calc_lower_bound: 22019
*/
/*
order_edge_などを外で持つようにした
11156ms
node: 3910
calc_1tree: 69298
calc_lower_bound: 9883
!?
*/