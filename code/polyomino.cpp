/*
ポリオミノ列挙
有向(向きが異なる物を別の物と考える)
n=8,9,10,15で試したところ、総数は一致していた
参考: https://qiita.com/ref3000/items/af18a4532123c22a19a4
verify:
https://atcoder.jp/contests/rco-contest-2017-qual/submissions/45764447
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
    int from, to;
    T cost;
    Edge(int from,int to,T cost):from(from),to(to),cost(cost){}
    Edge()=default;

    bool operator<(const Edge<T> &e){
        return cost<e.cost;
    }
    bool operator<=(const Edge<T> &e){
        return cost<=e.cost;
    }
};
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
template<typename T>
inline int lower_idx(const vector<T> &C,T value){
    return lower_bound(C.begin(),C.end(),value)-C.begin();
}
template<typename T>
inline int upper_idx(const vector<T> &C,T value){
    return upper_bound(C.begin(),C.end(),value)-C.begin();
}

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
        for(int i=0;i<CAPACITY;i++){
            pos_[i]=-1;
        }
        set_.clear();
    }
    void insert(int v){
        assert(pos_[v]==-1);
        if(pos_[v]!=-1) return;
        pos_[v]=set_.size();
        set_.push_back(v);
    }

    void remove(int v){
        assert(pos_[v]!=-1);
        int last=set_[set_.size()-1];
        set_[pos_[v]]=last;
        pos_[last]=pos_[v];
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
        assert(set_.size());
        int x=set_[Rand32(set_.size())];
        assert(pos_[x]!=-1);
        return x;
    }

    int random_extract(){
        int v=set_[Rand32(set_.size())];
        remove(v);
        return v;
    }
};

///++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
///++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
///++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

template<class T,class U>
T linear_function(U x,U start_x,U end_x,T start_value,T end_value){
    if(x>=end_x) return end_value;
    return start_value+(end_value-start_value)*(x-start_x)/(end_x-start_x);
}

//http://gasin.hatenadiary.jp/entry/2019/09/03/162613
struct SimulatedAnnealing{
    float startTemp; //差の最大値(あくまでも参考)
    float endTemp; //差の最小値(あくまでも参考)
    float startTime;
    float endTime;
    bool yama;
    bool minimum;
    SimulatedAnnealing(){}
    SimulatedAnnealing(float startTemp,float endTemp,float startTime,float endTime,bool yama,bool minimum):
        startTemp(startTemp),endTemp(endTemp),startTime(startTime),endTime(endTime),yama(yama),minimum(minimum){
    }
    float calc_temp(){
        return linear_function(float(TIME.span()),startTime,endTime,startTemp,endTemp);
    }
    float calc_prob(float diff){
        if(minimum) diff*=-1;
        if(diff>0) return 1;
        float temp=calc_temp();
        return exp(diff/temp);
    }
    float calc_diff(float prob){
        float diff=log(prob)*calc_temp();
        if(minimum) diff*=-1;
        return diff;
    }
    inline bool operator()(float diff){
        if(minimum) diff*=-1;
        if(diff>0) return true;
        if(yama) return false;
        float prob=calc_prob(diff);
        if(prob>float(Rand32()&mask(30))/mask(30)) return true;
        else return false;
    }
};
SimulatedAnnealing SA;

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
    REGIST_PARAM(endTemp,float,0);
    REGIST_PARAM(yama,bool,false);
    REGIST_PARAM(swap_prob,int,-1);
    REGIST_PARAM ( num_mark , int , 14 );
    REGIST_PARAM ( radius , int , 476 );
    REGIST_PARAM ( first_stage , double , 0.432904880557773 );
    REGIST_PARAM ( startTemp , double , 535025.7615236373 );
    REGIST_PARAM ( secondTemp , double , 75190.50602424436 );
    REGIST_PARAM(TIME_END,int,5900);
};

const int Height=50,Width=50;
const int num_squares_piece=8;


struct Piece{
    vector<pii> v;

    void add(int h,int w){
        v.emplace_back(h,w);
    }
    void output(){
        for(int k=0;k<v.size();k++){
            cout<<v[k].first+1<<" "<<v[k].second+1<<endl;
        }
    }
};
bool OutGrid(int h,int w){
    return h<0 or h>=Height or w<0 or w>=Width;
}

auto grid=vmake(Height,Width,0);

void input(){
    int dummy;
    cin>>dummy>>dummy>>dummy;
    for(int h=0;h<Height;h++) for(int w=0;w<Width;w++){
        char c;
        cin>>c;
        grid[h][w]=c-'0';
    }
}
//https://qiita.com/ref3000/items/af18a4532123c22a19a4
struct enumerate_polynomio{
    vector<int> dh={-1,0,0,1};
    vector<int> dw={0,-1,1,0};
    vector<vector<pii>> polynomios; //(h,w)の配列
    int size;
    int Height;
    int Width;
    vector<vector<int>> grid;

    void solve(int size_){
        size=size_;
        int Height=size+1;
        int Width=size*2+3;
        grid=vmake(Height,Width,-1);
        grid[0][size+1]=0;
        id_place.resize(Height*Width);
        id_place[0]=pii(0,size+1);
        polynomio.emplace_back(0,size+1);
        dfs(0,size+1,0,1);
    }
    vector<pii> polynomio;
    vector<pii> id_place;
    bool InGrid(int h,int w){
        return h>0 or (h==0 and w>=size+1);
    }
    void dfs(int h,int w,int max_id_in,int next_id){
        if(polynomio.size()==size){
            polynomios.push_back(polynomio);
            return;
        }
        //隣接するグリッドに記入
        vector<pii> fill_grid;
        for(int dir=0;dir<4;dir++){
            int nh=h+dh[dir];
            int nw=w+dw[dir];
            if(not InGrid(nh,nw)) continue;
            assert(grid[nh][nw]<next_id);
            if(grid[nh][nw]!=-1) continue;
            id_place[next_id]=pii(nh,nw);
            grid[nh][nw]=next_id++;
            fill_grid.emplace_back(nh,nw);
        }
        //もっと深く
        for(int i=max_id_in+1;i<next_id;i++){
            polynomio.push_back(id_place[i]);
            dfs(id_place[i].first,id_place[i].second,i,next_id);
            polynomio.pop_back();
        }
        //元に戻す
        for(auto p:fill_grid){
            grid[p.first][p.second]=-1;
        }
        return;
    }
};
struct P{
    ll score;
    int id_polynomio;
    int sh,sw;
    int rand;
    bool operator<(const P &p)const{
        return score*10000+rand>p.score*10000+p.rand;
    }
};
enumerate_polynomio PO;
ll best_score=-1;
vector<Piece> best_ans; 
vector<P> paterns;
void solve(){
    ll ans_score=0;
    auto status=vmake(Height,Width,false);
    vector<Piece> ans; 
    for(auto &pat:paterns){
        pat.rand=Rand32(0,10000);
    }
    vsort(paterns);

    for(auto pat:paterns){
        bool ok=true;
        Piece piece;
        for(int i=0;i<8;i++){
            int h=pat.sh+PO.polynomios[pat.id_polynomio][i].first;
            int w=pat.sw+PO.polynomios[pat.id_polynomio][i].second;
            if(status[h][w]){
                ok=false;
                break;
            }
            piece.add(h,w);
        }
        if(not ok) continue;
        ans_score+=pat.score;
        ans.push_back(piece);
        for(auto p:piece.v){
            status[p.first][p.second]=true;
        }
    }
    cerr<<ans_score<<endl;
    if(chmax(best_score,ans_score)){
        best_ans=ans;
    }
}
 
int main(const int argc,const char** argv){
    FastIO();
    input();
    PO.solve(8);
    cerr<<PO.polynomios.size()<<endl;
    for(int sh=-10;sh<Height;sh++) for(int sw=-10;sw<Width;sw++){
        for(int id=0;id<PO.polynomios.size();id++){
            bool ok=true;
            ll score=1;
            for(int i=0;i<8;i++){
                int h=sh+PO.polynomios[id][i].first;
                int w=sw+PO.polynomios[id][i].second;
                if(OutGrid(h,w) or grid[h][w]==0){
                    ok=false;
                    break;
                }
                score*=grid[h][w];
            }
            if(not ok) continue;
            paterns.push_back({score,id,sh,sw,Rand32(0,10000)});
        }
    }
    int cnt=0;
    solve();
    
    cout<<best_ans.size()<<endl;
    for(auto a:best_ans){
        a.output();
    }
}