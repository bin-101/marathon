/*
グリッドグラフの連結チェックの際、周囲3*3領域のみを見るやつ
constexprをつけるとコンパイル時に計算してくれる(多分)
参考:https://speakerdeck.com/shun_pi/ahc023can-jia-ji-zan-ding-ban-shou-shu-kitu-duo-me?slide=18
verify: https://atcoder.jp/contests/ahc024/submissions/46387069
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
template<typename T1,typename T2> constexpr bool chmin(T1 &a,T2 b){if(a<=b)return 0; a=b; return 1;}
template<typename T1,typename T2> constexpr bool chmax(T1 &a,T2 b){if(a>=b)return 0; a=b; return 1;}
template<typename T> constexpr int bitUP(T x,int a){return (x>>a)&1;}
enum Dir{
    Right,
    Down,
    Left,
    Up
};
//→　↓ ← ↑
int dh[4]={0,1,0,-1}, dw[4]={1,0,-1,0};
//上から時計回り
int dh8[8]={0,1,1,1,0,-1,-1,-1}, dw8[8]={1,1,0,-1,-1,-1,0,1};
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
    //http://web.archive.org/web/20200105011004/https://topcoder.g.hatena.ne.jp/tomerun/20171216/
    //[0,b)の中から異なる2つを選ぶ first<second
    inline pair<int,int> two(int b){
        assert(b>=2);
        int v1=rnd_make()%b;
        int v2=rnd_make()%(b-1);
        if (v2>=v1) return {v1,v2+1};
        else return {v2,v1};
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
        //assert(pos_[v]==-1);
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
    void clear(){
        while(size()){
            remove(set_.back());
        }
    }
};

///++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
///++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
///++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

template<class T,class U>
T linear_function(U x,U start_x,U end_x,T start_value,T end_value){
    if(x>=end_x) return end_value;
    if(x<=start_x) return start_value;
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
    float nowTemp;
    int cnt=0;
    int T;
    //SimulatedAnnealing(){}
    //Tごとに温度の再計算をする
    SimulatedAnnealing(float startTemp,float endTemp,float startTime,float endTime,bool yama,bool minimum,int T=1):
        startTemp(startTemp),endTemp(endTemp),startTime(startTime),endTime(endTime),yama(yama),minimum(minimum),T(T){
    }
    float calc_temp(){
        if(cnt%T) nowTemp=linear_function(float(TIME.span()),startTime,endTime,startTemp,endTemp); //線形
        cnt++;
        return nowTemp;
        //https://atcoder.jp/contests/ahc014/submissions/35326979
        /*float progress=(TIME.span()-startTime)/(endTime-startTime);
        if(progress>1.0) progress=1.0;
        return pow(startTemp,1.0-progress)*pow(endTemp,progress);*/
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
        testCounter.count("try_cnt");
        if(minimum) diff*=-1;
        if(diff>0){
            testCounter.count("plus_change");
            return true;
        }
        if(yama) return false;
        float prob;
        if(minimum) prob=calc_prob(diff*-1);
        else prob=calc_prob(diff);
        if(prob>float(Rand32()&mask(30))/mask(30)){
            testCounter.count("minus_change");
            return true;
        }
        else return false;
    }
    //log(rand)*temp rand:[0,1]の乱数
    inline float calc_minus(){
        float rand=float(Rand32()&mask(30))/mask(30);
        rand=log(rand);
        if(minimum) rand*=-1;
        return rand*calc_temp();
    }
};

struct UnionFind{
    vector<int> par;
    vector<int> esz; //辺の数
    int group; //集合の数
    UnionFind(int N) : par(N,-1),esz(N,0),group(N) {}
    
    //rootを探す
    int find(int x){
        if(par[x]<0) return x;
        else return par[x]=find(par[x]);
    }
    //集合の要素数
    int usize(int x) {return -par[find(x)];}
    //集合の辺の数
    int esize(int x) {return esz[find(x)];}
    //xとyが繋がっていたらfalseを返す
    bool unite(int x,int y){
        x=find(x);
        y=find(y);
        esz[x]++;
        if(x==y) return false;
        group--;
        if(usize(x)<usize(y)) swap(x,y);
        par[x]+=par[y];
        esz[x]+=esz[y];
        par[y]=x;
        return true;
    }

    bool same(int x,int y) {return find(x)==find(y);}

};

///++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
///++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
///++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//https://atcoder.jp/contests/asprocon9/submissions/34659956
#ifndef OPTUNA 
#define REGIST_PARAM(name, type, defaultValue) constexpr type name = defaultValue
#else
#define REGIST_PARAM(name, type, defaultValue) type name = defaultValue
#endif

const int Height=50+2,Width=50+2;
const int num_country=101;
auto grid_input=vmake(Height+2,Width+2,0);
vector<bitset<num_country>> adjust_input(num_country);

struct Place{
    int h,w;
    Place(){}
    Place(int h,int w):h(h),w(w){
 
    }
    Place(int idx):h(idx/Width),w(idx%Width){
 
    }
    int dist(const Place &np){
        return abs(h-np.h)+abs(w-np.w);
    }

    Place pre_place(int dir){
        return Place(h-dh[dir],w-dw[dir]);
    }    
    Place next_place(int dir){
        return Place(h+dh[dir],w+dw[dir]);
    }
    void move(int dir){
        h+=dh[dir];
        w+=dw[dir];
    }
    void remove(int dir){
        h-=dh[dir];
        w-=dw[dir];        
    }
    bool ok_move(int dir){
        return h+dh[dir]<Height and w+dw[dir]<Width and h+dh[dir]>=0 and w+dw[dir]>=0;
    }
    bool out_grid(){
        return h>=Height or w>=Width or h<0 or w<0;
    }
    bool operator==(const Place &p){
        return h==p.h and w==p.w;
    }
    bool operator!=(const Place &p){
        return h!=p.h or w!=p.w;
    }
    int idx(){
        return h*Width+w;
    }
    //上下 左右
    bool operator<(Place &p){
        if(p.h!=h) return h<p.h;
        return w<p.w;
    }
    friend istream& operator>>(istream& is, Place& p){
        is>>p.h>>p.w;
        return is;
    }
    friend ostream& operator<<(ostream& os, Place& p){
        os<<p.h<<" "<<p.w;
        return os;
    }
};

namespace OP{
    REGIST_PARAM(yama,bool,false);
    REGIST_PARAM(startTemp,double,500000);
    REGIST_PARAM(endTemp,float,0);
    REGIST_PARAM(TIME_END,int,1900);
};
vector<vector<int>> cnt_adjust=vmake(num_country,num_country,0);
vector<IntSet<Height*Width>> IS(num_country);
vector<bool> ok_88=vmake(1<<8,false);

void input(){
    int dummy;
    cin>>dummy>>dummy;
    for(int h=1;h<=50;h++) for(int w=1;w<=50;w++){
        cin>>grid_input[h][w];
    }
    for(int h=0;h<=51;h++) for(int w=0;w<=51;w++){
        IS[grid_input[h][w]].insert(Place(h,w).idx());
    }

    for(int i=0;i<num_country;i++){
        adjust_input[i].reset();
    }

    for(int l=0;l<=8;l++){
        for(int s=0;s<8;s++){
            int idx=0;
            for(int i=0;i<l;i++){
                int u=(s+i)%8;
                idx|=(1<<u);
            }
            //cerr<<idx<<endl;
            ok_88[idx]=true;
        }
    }

    for(int h=0;h<Height;h++) for(int w=0;w<Width;w++){
        Place p(h,w);
        for(int dir=0;dir<4;dir++){
            auto np=p.next_place(dir);
            if(np.out_grid()){
                continue;
            }
            int c=grid_input[h][w];
            int nc=grid_input[np.h][np.w];
            if(c!=nc){
                adjust_input[c].set(nc);
                adjust_input[nc].set(c);
                cnt_adjust[c][nc]++;
            }
        }
    }
}

bool is_ok(const vector<vector<int>> &grid){
    vector<bitset<num_country>> adjust(num_country);
    for(int i=0;i<num_country;i++){
        adjust[i].reset();
    }
    UnionFind uni(Height*Width);

    for(int h=0;h<Height;h++) for(int w=0;w<Width;w++){
        Place p(h,w);
        for(int dir=0;dir<4;dir++){
            auto np=p.next_place(dir);
            if(np.out_grid()){
                continue;
            }
            int c=grid[h][w];
            int nc=grid[np.h][np.w];
            if(c!=nc){
                adjust[c].set(nc);
                adjust[nc].set(c);
            }else{
                uni.unite(p.idx(),np.idx());
            }
        }
    }

    if(uni.group!=num_country or adjust!=adjust_input) return false;
    return true;
}



void output(vector<vector<int>> grid){
    for(int h=1;h<=50;h++){
        for(int w=1;w<=50;w++){
            if(w!=1) cout<<" ";
            cout<<grid[h][w];
        }
        cout<<endl;
    }
}

struct GridLocalConnectivity3{
    const array<int,8> dh8={-1,-1,-1,0,0,1,1,1};
    const array<int,8> dw8={-1,0,1,-1,1,-1,0,1};

    const int size3=(1<<8);
    
    array<bool,(1<<8)> table;


    const array<int,4> near={-3,1,3,-1};

    constexpr GridLocalConnectivity3():table{}{
        build();
    }
    constexpr void dfs(array<int,9> &state,array<int,9> &group,int x){
        //4近傍を見る
        for(int d:near){
            int nx=x+d;
            if(nx<0 or nx>=9 or state[nx]==false) continue;
            if(x%3==2 and d==1) continue;
            if(x%3==0 and d==-1) continue;
            if(chmin(group[nx],group[x])){
                dfs(state,group,nx);
            }
        }
    }

    constexpr void build(){
        array<int,9> state{};
        array<int,9> group{};

        for(int S=0;S<(1<<8);S++){

            for(int i=0;i<8;i++){
                int p=0;
                if(i>=4) p=1;
                if(bitUP(S,i)) state[i+p]=true;
                else state[i+p]=false;
                group[i+p]=100;
            }
            state[4]=false;
            group[4]=100;

            for(int i=0;i<9;i++){
                if(state[i]==false) continue;
                if(chmin(group[i],i)){
                    dfs(state,group,i);
                }
            }

            table[S]=true;
            for(int d1:near){
                for(int d2:near){
                    int a=4+d1;
                    int b=4+d2;
                    if(state[a]==false or state[b]==false) continue;
                    if(group[a]!=group[b]) table[S]=false;
                }
            }

        }
    }

    constexpr bool is_connected(int x)const{
        return table[x];
    }
};

constexpr GridLocalConnectivity3 gridLocalConnectivity3;

bool transition(int h,int w,int post,vector<vector<int>> &grid){
    int pre=grid[h][w];
    grid[h][w]=post;

    bool ok=true;
    Place p(h,w);
    for(int dir=0;dir<4;dir++){
        auto np=p.next_place(dir);
        int c=grid[np.h][np.w];
        if(c!=post){
            cnt_adjust[post][c]++;
            cnt_adjust[c][post]++;
            if(cnt_adjust[post][c]==1) ok=false;
        }
    }
    for(int dir=0;dir<4;dir++){
        auto np=p.next_place(dir);
        int c=grid[np.h][np.w];
        if(c!=pre){
            cnt_adjust[pre][c]--;
            cnt_adjust[c][pre]--;
            if(cnt_adjust[pre][c]==0) ok=false;
            //assert(cnt_adjust[pre][c]>=0);
        }
    }
    if(ok==false) return false;

    int idx=0;
    int cnt=0;
    for(int dir=0;dir<8;dir++){
        int nh=p.h+gridLocalConnectivity3.dh8[dir];
        int nw=p.w+gridLocalConnectivity3.dw8[dir];
        if(grid[nh][nw]==pre) idx|=(1<<dir);
    }
    //cerr<<gridLocalConnectivity.is_connected5(idx);
    return gridLocalConnectivity3.is_connected(idx);
}

pair<bool,int> try3(vector<vector<int>> &grid,float minus){
    int h=Rand32(1,Height-1);
    int w=Rand32(1,Width-1);
    int dir=Rand32(4);

    Place p(h,w);

    auto np=p.next_place(dir);
    for(int d=0;d<4;d++){
        auto np=p.next_place(d);
        if(grid[np.h][np.w]==0) dir=d;
    }
    if(grid[h][w]==grid[np.h][np.w]) return {false,-1};

    int pre=grid[p.h][p.w];
    int post=grid[np.h][np.w];
    if(pre==0 and minus>-1) return {false,-1};

    int diff=0;
    if(pre==0) diff=-1;
    else if(post==0) diff=1;

    if(transition(h,w,post,grid)){
        return {true,diff};
    }
    transition(h,w,pre,grid);
    return {false,-1};
}

int calc_score(const vector<vector<int>> &grid){
    int score=1;
    if(is_ok(grid)==false){
        return -1000;
    }
    for(int h=1;h<=50;h++){
        for(int w=1;w<=50;w++){
            if(grid[h][w]==0) score++;
        }
    }
    return score;
}

vector<vector<int>> calc_cnt(vector<vector<int>> grid){
    vector<vector<int>> cnt_adjust=vmake(num_country,num_country,0);
    for(int h=0;h<Height;h++) for(int w=0;w<Width;w++){
        Place p(h,w);
        for(int dir=0;dir<4;dir++){
            auto np=p.next_place(dir);
            if(np.out_grid()){
                continue;
            }
            int c=grid[h][w];
            int nc=grid[np.h][np.w];
            if(c!=nc){
                cnt_adjust[c][nc]++;
            }
        }
    }
    return cnt_adjust;
}

void solve(){
    input();
    // for(int S=0;S<(1<<8);S++){
    //     cerr<<bitset<8>(S)<<endl;
    //     cerr<<gridLocalConnectivity.is_connected5(S)<<endl;
    // }

    auto grid=grid_input;

    SimulatedAnnealing SA(1.0,0.01,TIME.span(),OP::TIME_END,true,false,10000);

    auto best_grid=grid;
    int best_score=calc_score(grid);
    int now_score=best_score;

    int cnt=0;
    while(TIME.span()<OP::TIME_END){
        bool result;
        int diff;
        tie(result,diff)=try3(grid,SA.calc_minus());
        
        if(result){
            now_score+=diff;
            //assert(calc_score(grid)==now_score);
            //if(testCounter.cnt["update"]%1000==0) output(grid);
            testCounter.count("update");
        }

        if(best_score<now_score){
            cnt++;
            best_score=now_score;
            best_grid=grid;
            //cerr<<best_score<<endl;
        }
        testCounter.count("try");
    }
    assert(cnt_adjust==calc_cnt(grid));
    grid=best_grid;
    output(grid);

#ifndef ONLINE_JUDGE
    testTimer.output();
    testCounter.output();
    cerr<<TIME.span()<<"ms"<<endl;
    cerr<<calc_score(grid)<<endl;
    //cerr<<"score: "<<simulate(best_grid,true)<<endl;
#endif
}
 
int main(const int argc,const char** argv){
#ifndef OPTUNA
    if(argc!=1){

    }
#endif
    FastIO();
    int T=1;
    //cin>>T;
    while(T--) solve();
}