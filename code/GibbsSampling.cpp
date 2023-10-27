/*
AHC025の入力でverifyした
*/
//#define NDEBUG

//#define ONLINE_JUDGE

#ifndef ONLINE_JUDGE
//#define OPTUNA
#endif

#ifdef ONLINE_JUDGE
#define NDEBUG
#pragma GCC target("avx2")
#pragma GCC optimize("O3")
#pragma GCC optimize("unroll-loops")
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
enum Dir{
    Right,
    Down,
    Left,
    Up
};
//→　↓ ← ↑
int dh[4]={0,1,0,-1}, dw[4]={1,0,-1,0};
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
template<class T>
using minimum_queue=priority_queue<T,vector<T>,greater<T>>;

using Graph=vector<vector<int>>;

using i8=int8_t;
using i16=int16_t;
using i32=int32_t;
using i64=int64_t;

using u8=uint8_t;
using u16=uint16_t;
using u32=uint32_t;
using u64=uint64_t; 

void FastIO(){
    ios::sync_with_stdio(false);
    cin.tie(0);
    cout << fixed << setprecision(20);
}
//0-indexed vector cin
template<typename T>
inline istream &operator>>(istream &is,vector<T> &v) {
    for(size_t i=0;i<v.size();i++) is>>v[i];
	return is;
}
 
//0-indexed vector cin
template<typename T>
inline istream &operator>>(istream &is,vector<vector<T>> &v) {
    for(size_t i=0;i<v.size();i++){
        is>>v[i];
    }
    return is;
}


struct Xor32{
    using u32=uint32_t;
 
    u32 x=1234567;
    inline u32 rnd_make(){
        x^=(x<<13);
        x^=(x>>17);
        x^=(x<<5);
        return x;
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
    //http://web.archive.org/web/20200105011004/https://topcoder.g.hatena.ne.jp/tomerun/20171216/
    //[0,b)の中から異なる2つを選ぶ [0]の値<[1]の値
    inline array<int,2> two(int b){
        assert(b>=2);
        int v1=rnd_make()%b;
        int v2=rnd_make()%(b-1);
        if (v2>=v1) return {v1,v2+1};
        else return {v2,v1};
    }
    inline float random01(){
        return float(rnd_make())/mask(32);
    }
    //確率pでtrueを返す
    inline bool gen_bool(float p){
        return p>random01();
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
    float local;
    Timer(){
#ifndef ONLINE_JUDGE
        local=1.0;
#endif
        start();
    }
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
    void pop_back(){
        size_--;
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

//ソート
template<typename T>
inline void vsort(vector<T> &v){
    sort(v.begin(),v.end());
}
//逆順ソート
template<typename T>
inline void rvsort(vector<T> &v){
	sort(v.rbegin(),v.rend());
}
//1ビットの数を返す
inline int popcount(int x){
	return __builtin_popcount(x);
}
//1ビットの数を返す
inline int popcount(ll x){
	return __builtin_popcountll(x);
}
template<typename T>
inline void Compress(vector<T> &C){
    sort(C.begin(),C.end());
    C.erase(unique(C.begin(),C.end()),C.end());
}
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

//vは昇順
bool is_in(const vector<int> &v,int x){
    int n=v.size();
    if(n==0) return false;
    int ng=-1,ok=n-1;
    while(ok-ng!=1){
        int mid=(ok+ng)/2;
        if(v[mid]<x) ng=mid;
        else ok=mid;
    }
    if(v[ok]==x) return true;
    return false;
}

template<typename T >
struct edge {
	int to;
	T cost;
    int id;
	edge()=default;
	edge(int to, T cost,int id) : to(to), cost(cost), id(id) {}
};
template<class T >
struct Edge {
    int from, to,id;
    T cost;
    Edge(int from,int to,T cost,int id):from(from),to(to),cost(cost),id(id){}
    Edge()=default;

    bool operator<(const Edge<T> &e){
        return cost<e.cost;
    }
    bool operator<=(const Edge<T> &e){
        return cost<=e.cost;
    }
};
template<typename T>
using WeightGraph=vector<vector<edge<T>>>;
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
//pair cout
template<typename T, typename U>
inline ostream &operator<<(ostream &os,const pair<T,U> &p) {
	os<<p.first<<" "<<p.second;
	return os;
}
//map cout
template<typename F, typename S>
inline ostream &operator<<(ostream &os,const map<F,S> &M) {
	bool first=false;
    for(auto x:M){
        if(first) os<<endl;
        first=true;
        os<<x;
    }
	return os;
}
//set cout
template<typename T>
inline ostream &operator<<(ostream &os,const set<T> &S) {
	bool first=false;
    for(auto x:S){
        if(first) os<<endl;
        first=true;
        os<<x;
    }
	return os;
}
 
//pair cin
template<typename T, typename U>
inline istream &operator>>(istream &is,pair<T,U> &p) {
	is>>p.first>>p.second;
	return is;
}
template<typename T>
void append(vector<T> &v,const vector<T> &vp){
    for(auto p:vp){
        v.push_back(p);
    }
}
//Fisher–Yatesアルゴリズム
template<typename T>
void shuffle(vector<T> &v){
    int sz=v.size();
    for(int i=sz-1;i>0;i--){
        swap(v[Rand32()%(i+1)],v[i]);
    }
}

///++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
///++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
///++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//https://atcoder.jp/contests/asprocon9/submissions/34659956
#ifndef OPTUNA 
#define REGIST_PARAM(name, type, defaultValue) constexpr type name = defaultValue
#else
#define REGIST_PARAM(name, type, defaultValue) type name = defaultValue
#endif

namespace OP{
    REGIST_PARAM(yama,bool,false);
    REGIST_PARAM(startTemp,double,500000);
    REGIST_PARAM(endTemp,float,0);
    REGIST_PARAM(TIME_END,int,1900);
};

constexpr int max_num_number=100;
int num_number;
int num_divide;
int num_query;

int num_done_query;
#ifndef ONLINE_JUDGE
vector<int> weight_input;
#endif

void input(){
    cin>>num_number>>num_divide>>num_query;
#ifndef ONLINE_JUDGE
    weight_input.resize(num_number);
    cin>>weight_input;
#endif
}

map<pair<vector<int>,vector<int>>,int> memo_query;


int num_estimation=100;
auto weight_estimation=vmake(num_estimation,max_num_number,0);

struct F{
    vector<int> L;
    vector<int> R;
    int result;
    F(vector<int> L,vector<int> R,int result):
        L(L),R(R),result(result){

    }
    bool operator()(const vector<int> &state){
        int sum_left=0,sum_right=0;
        for(int i:L){
            sum_left+=state[i];
        }
        for(int i:R){
            sum_right+=state[i];
        }
        int res=0;
        if(sum_left<sum_right){
            res=1;
        }
        if(sum_left>sum_right){
            res=-1;
        }
        return res==result;
    }
};

//条件がどんどん追加されていくものを想定
//sampleの型はvector<S>
//F: bool operator()(vector<S> state) をもつクラス
//参考: https://twitter.com/wata_orz/status/1716373444273885472
//  2で、適当に生成して条件を満たしたら採用、をする
template<class S,class F>
struct GibbsSampling{
    vector<vector<S>> samples_now_;
    vector<F> conditions_now_;
    function<S()>  generation_; //変数1個を生成する
    int num_samples_;
    int num_dimension_;

    GibbsSampling(){

    }

    GibbsSampling(function<S()> generation,int num_samples,int num_dimension)
        :generation_(generation),num_samples_(num_samples),num_dimension_(num_dimension){
        //初期サンプル
        for(int i=0;i<num_samples_;i++){
            vector<S> state;
            for(int j=0;j<num_dimension_;j++){
                state.push_back(generation_());
            }
            samples_now_.push_back(state);
        }
    }

    //stateはconditionsを満たしていると仮定
    //  この仮定を満たしていなくても動くが……
    void change(vector<S> &state,vector<F> conditions){
        for(auto &s:state){
            auto pre_s=s;
            int cnt=0;
            while(true){
                s=generation_();
                bool ok=true;
                for(auto condition:conditions){
                    if(condition(state)==false){
                        ok=false;
                        break;
                    }
                }
                if(ok) break;
                cnt++;
                // if(cnt==100){ //高速化: ある程度やってダメなら変化させない
                //     s=pre_s;
                //     break;
                // }
            }
        }
    }

    void add_condition(F condition_new){
        //条件に合ってるものを残す
        vector<vector<S>> samples_new_;
        for(auto &state:samples_now_){
            if(condition_new(state)){
                samples_new_.push_back(state);
            }
        }

        //samples_new_が空のとき、
        while(samples_new_.size()==0){
            auto state=samples_now_[Rand32(samples_now_.size())];
            change(state,conditions_now_);
            if(condition_new(state)){
                samples_new_.push_back(state);
            }
        }

        conditions_now_.push_back(condition_new);

        while(samples_new_.size()<num_samples_){
            auto state=samples_new_[Rand32(samples_new_.size())];
            change(state,conditions_now_);
            samples_new_.push_back(state);
        }
        samples_now_=samples_new_;
    }
};
GibbsSampling<int,F> Gib;
//0:= 1:< -1:>
int query(const vector<int> &L,const vector<int> &R){

    int result=-1;
    ll L_sum=0,R_sum=0;
    for(int l:L){
        L_sum+=weight_input[l];
    }
    for(int r:R){
        R_sum+=weight_input[r];
    }
    //cerr<<L_sum<<" "<<R_sum<<endl;
    if(L_sum<R_sum){
        result=1;
    }else if(L_sum>R_sum){
        result=-1;
    }else{
        result=0;
    }

    num_done_query++;
    memo_query[pair(L,R)]=result;
    cerr<<L<<" "<<R<<endl;
    assert(result!=0); //等号があるとつらい

    F condition(L,R,result);
    Gib.add_condition(condition);
    return result;
}


void solve(){
    input();
    auto generation=[&]{
        int x=1e9;;
        while(x>100000*num_number/num_divide){
            x=max(1.0,round(-log(Rand32.random01())*1e5));
        }
        return x;
    };

    Gib=GibbsSampling<int,F>(generation,100,num_number);

    vector<vector<int>> Ls,Rs;
    vector<int> results;

    while(num_query--){
        vector<int> L;
        vector<int> R;
        for(int i=0;i<num_number;i++){
            int rand=Rand32(3);
            if(rand==0){
                L.push_back(i);
            }else if(rand==1){
                R.push_back(i);
            }
        }
        Ls.push_back(L);
        Rs.push_back(R);
        results.push_back(query(L,R));
    }

    //生成したサンプルが条件を満たすか確認
    for(auto state:Gib.samples_now_){
        for(int i=0;i<Ls.size();i++){
            int sum_left=0,sum_right=0;
            for(int j:Ls[i]){
                sum_left+=state[j];
            }
            for(int j:Rs[i]){
                sum_right+=state[j];
            }
            int res=0;
            if(sum_left<sum_right){
                res=1;
            }
            if(sum_left>sum_right){
                res=-1;
            }
            assert(res==results[i]);
        }

    }

#ifndef ONLINE_JUDGE
    testTimer.output();
    testCounter.output();
    cerr<<TIME.span()<<"ms"<<endl;
#endif
}
 
int main(const int argc,const char** argv){
#ifndef OPTUNA
    if(argc!=1){

    }
#endif
    int T=1;
    //cin>>T;
    while(T--) solve();
}