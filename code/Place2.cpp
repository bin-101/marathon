/*
2次元座標
verify: https://atcoder.jp/contests/atcoder11live/submissions/42791824
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
            random_extract();
        }
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
        double t=TIME.span()/1950.0;
        chmin(t,1);
        return pow(0.01,1.0-t)*pow(0.001,t);
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
        float prob;
        if(minimum) prob=calc_prob(diff*-1);
        else prob=calc_prob(diff);
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
    REGIST_PARAM(swap_prob,int,-1);
    REGIST_PARAM ( num_mark , int , 14 );
    REGIST_PARAM ( radius , int , 476 );
    REGIST_PARAM ( first_stage , double , 0.432904880557773 );
    REGIST_PARAM ( secondTemp , double , 75190.50602424436 );

    REGIST_PARAM(yama,bool,false);
    REGIST_PARAM ( startTemp , double , 500000);
    REGIST_PARAM(TIME_END,int,1900);
    REGIST_PARAM(greedy_candidate,int,20);
};

struct UnionFind{
    vector<int> par;
    UnionFind(int N) : par(N,-1) {}
    
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

};

int Height,Width;
vector<vector<char>> grid_input;
int empty_cnt;


struct Place{
    int h,w;
    Place(){}
    Place(int h,int w):h(h),w(w){
 
    }
    Place(int idx):h(idx/Width),w(idx%Width){
 
    }
    int dist(Place &np){
        return sqrt(double(h-np.h)*(h-np.h) + double(w-np.w)*(w-np.w))+1;
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
    bool operator==(Place &p){
        return h==p.h and w==p.w;
    }
    bool operator!=(Place &p){
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
Place pos_start;

void input(){
    cin>>Height;
    Width=Height;
    cin>>pos_start;

    grid_input.resize(Height);
    for(int h=0;h<Height;h++){
        grid_input[h].resize(Width);
        for(int w=0;w<Width;w++){
            cin>>grid_input[h][w];
            if(grid_input[h][w]=='.') empty_cnt++;
        }
    }
}

vector<array<int,4>> used;
vector<pii> path_pos;
int start_idx;
bool is_put;
int bfs_cnt;
IntSet<50*50> obs_pos;
double real_score=-1;
double simulate(const vector<vector<char>> &grid,bool real=false){
    bfs_cnt++;
    path_pos.clear();
    Place pos_now=pos_start;
    int dir_now=Right;
    int cnt=0;

    while(true){
        if(used[pos_now.idx()][dir_now]==bfs_cnt) break;
        path_pos.emplace_back(pos_now.idx(),dir_now);
        used[pos_now.idx()][dir_now]=bfs_cnt;
        auto pos_next=pos_now.next_place(dir_now);
        //cerr<<pos_now.ok_move(dir_now)<<endl;
        if(pos_now.ok_move(dir_now)==false or grid[pos_next.h][pos_next.w]=='#'){
            dir_now=(dir_now+1)%4;
            continue;
        }
        pos_now.move(dir_now);
        cnt++;
    }
    real_score=round(1e6*cnt/4/empty_cnt);
    if(real) return round(1e6*cnt/4/empty_cnt);
    return cnt+max(0.0,1.0-TIME.span()/1950.0*1.1)*obs_pos.size()*0.3;
}

void kick(){

}

void solve(){
    input();
    used=vmake(Height*Width,array<int,4>());
    for(int i=0;i<Height*Width;i++) for(int d=0;d<4;d++){
        used[i][d]=false;
    }
    auto best_grid=grid_input;
    simulate(best_grid);
    double best_score=real_score;
    //cerr<<best_score<<endl;

    auto grid_now=grid_input;
    auto now_score=simulate(best_grid);

    SimulatedAnnealing SA(5,0,TIME.span(),1940,false,false);

    while(TIME.span()<1950){
        Place put_pos;
        if(Rand32(obs_pos.size()+path_pos.size())<obs_pos.size()){
            put_pos=Place(obs_pos.random());
        }else{
            start_idx=Rand32(path_pos.size());
            put_pos=Place(path_pos[start_idx].first);
            is_put=true;
            if(Rand32(3)) continue;
        }
        int h=put_pos.h;
        int w=put_pos.w;
        if(Place(h,w)==pos_start) continue;
        //int out_range=round(Height*TIME.span()/1950.0*0.9);
        //if(min(h,Height-1-h)>out_range or min(w,Width-1-w)>out_range) continue;
        if(grid_now[h][w]=='#') grid_now[h][w]='.';
        else grid_now[h][w]='#';

        testCounter.count("try");

        double temp_score=simulate(grid_now);
        //cerr<<temp_score-now_score<<endl;
        if(chmax(best_score,real_score)){
            best_grid=grid_now;
        }
        if(SA( double(temp_score-now_score)/(now_score+0.1) )){
            now_score=temp_score;
            testCounter.count("success");
            //cerr<<best_score<<endl;
            //cerr<<h<<" "<<w<<endl;
            if(grid_now[h][w]=='#') obs_pos.insert(put_pos.idx());
            else obs_pos.remove(put_pos.idx());
        }else{
            if(grid_now[h][w]=='#') grid_now[h][w]='.';
            else grid_now[h][w]='#';
        }

    }
    vector<Place> ans;
    for(int h=0;h<Height;h++) for(int w=0;w<Width;w++){
        if(best_grid[h][w]!=grid_input[h][w]){
            ans.emplace_back(h,w);
        }
    }
    cout<<ans.size()<<endl;
    for(auto p:ans){
        cout<<p<<endl;
    }
#ifndef ONLINE_JUDGE
    testTimer.output();
    testCounter.output();
    cerr<<TIME.span()<<"ms"<<endl;
    cerr<<"score: "<<simulate(best_grid,true)<<endl;
    // cerr<<"cost_tree: "<<cost_tree<<endl;
    // cerr<<"cost_power: "<<cost_power<<endl;
    // cerr<<"sum: "<<cost_tree+cost_power<<endl;
    // cerr<<"score: "<<ll( round(1e6*(1+1e8/(cost_tree+cost_power+1e7)))   )<<endl;
#endif
}
 
int main(const int argc,const char** argv){
#ifndef OPTUNA
    if(argc!=1){

    }
#endif
    //FastIO();
    int T=1;
    //cin>>T;
    while(T--) solve();
}
