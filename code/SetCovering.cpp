/*
TODO:

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
    inline double random_double(double a,double b){
        double sa=b-a;
        a+=random01()*sa;
        return a;
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
    //最初から見ていき、一致したものを削除(remove)
    bool erase(T value){
        for(int i=0;i<size_;i++){
            if(array_[i]==value){
                remove(i);
                return true;
            }
        }
        return false;
    }
    void reverse(){
        for(int i=0;i<size_/2;i++){
            swap(array_[i],array_[size_-1-i]);
        }
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

template<class T,class U>
T linear_function(U x,U start_x,U end_x,T start_value,T end_value){
    if(x>=end_x) return end_value;
    if(x<=start_x) return start_value;
    return start_value+(end_value-start_value)*(x-start_x)/(end_x-start_x);
}

//http://gasin.hatenadiary.jp/entry/2019/09/03/162613
struct SimulatedAnnealing{
    float temp_start; //差の最大値(あくまでも参考)
    float temp_end; //差の最小値(あくまでも参考)
    float time_start;
    float time_end;
    bool is_hill;
    bool minimum;
    int interval; //intervalごとに温度の再計算

    float temp_now;
    int cnt_calc_temp;
    /*
    0:線形
    1:pow pow
    2:指数
    */
    int type_temp=0;

    //SimulatedAnnealing(){}
    SimulatedAnnealing(float temp_start,float temp_end,float time_start,float time_end,bool is_hill,bool minimum,int interval=1):
        temp_start(temp_start),temp_end(temp_end),time_start(time_start),time_end(time_end),
        is_hill(is_hill),minimum(minimum),interval(interval),temp_now(temp_start),cnt_calc_temp(0){
    }
    float calc_temp(){
        if(cnt_calc_temp%interval==0){
            float progress=float(TIME.span()-time_start)/(time_end-time_start);
            if(progress>1.0) progress=1.0;
            if(type_temp==0){//線形
                temp_now=temp_start*(1.0-progress)+temp_end*progress;
            }else if(type_temp==1){ //https://atcoder.jp/contests/ahc014/submissions/35326979
                temp_now = pow(temp_start,1.0-progress)*pow(temp_end,progress);
            }else{ //https://ozy4dm.hateblo.jp/entry/2022/12/22/162046#68-%E3%83%97%E3%83%AB%E3%83%BC%E3%83%8B%E3%83%B3%E3%82%B0%E6%97%A9%E6%9C%9F%E7%B5%82%E4%BA%86%E5%8D%98%E7%B4%94%E5%8C%96%E3%81%95%E3%82%8C%E3%81%9F%E8%A8%88%E7%AE%97%E3%82%92%E4%BD%BF%E7%94%A8%E3%81%99%E3%82%8B
                temp_now = temp_start*pow(temp_end/temp_start,progress);
            }
        }
        cnt_calc_temp++;
        return temp_now;
    }
    //diff: スコアの変化量
    //確率を計算
    float calc_prob(float diff){
        if(minimum) diff*=-1;
        if(diff>0) return 1;
        float temp=calc_temp();
        return exp(diff/temp);
    }
    inline bool operator()(float diff){
        testCounter.count("try_cnt");
        if(minimum) diff*=-1;
        if(diff>=0){
            if(diff==0) testCounter.count("zero_change");
            else testCounter.count("plus_change");
            return true;
        }
        if(is_hill) return false;

        float prob = exp(diff/calc_temp());

        if(Rand32.gen_bool(prob)){
            testCounter.count("minus_change");
            return true;
        }
        else return false;
    }
    //最大化の場合,返り値<変化量なら遷移してもよい
    float calc_tolerance(float prob){
        float tolerance=log(prob)*calc_temp();
        if(minimum) tolerance*=-1;
        return tolerance;
    }
    //log(prob)*temp prob:[0,1]の乱数
    float calc_tolerance(){
        float prob=Rand32.random01();
        return calc_tolerance(prob);
    }
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

namespace OP{
    REGIST_PARAM(yama,bool,false);
    REGIST_PARAM(startTemp,double,500000);
    REGIST_PARAM(endTemp,float,0);
    REGIST_PARAM(TIME_END,int,1900);
};
constexpr int Height=50;
constexpr int Width=50;
vector<string> ans;
struct Place{
    int h,w;
    Place(){}
    Place(int h,int w):h(h),w(w){
 
    }
    Place(int idx):h(idx/Width),w(idx%Width){
 
    }
    int dist(const Place &np)const{
        return abs(np.h-h)+abs(np.w-w);
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
    int id(){
        return idx();
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
    void move(Place p){
        h+=p.h;
        w+=p.w;
    }
};

//[0,n)の集合を管理
//値の追加・削除・存在確認: O(1)
//空間計算量: O(n)
//重複は許さない
//https://topcoder-tomerun.hatenablog.jp/entry/2021/06/12/134643
template<int CAPACITY>
struct IndexSet{
    DynamicArray<int,CAPACITY> set_;
    array<int,CAPACITY> pos_;

    IndexSet(){
        for(int i=0;i<CAPACITY;i++){
            pos_[i]=-1;
        }
        set_.clear();
    }
    void insert(int v){
        assert(v>=0 and v<CAPACITY);
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
    void clear(){
        while(size()){
            remove(set_.back());
        }
    }
    vector<int> make_vector(){
        vector<int> v;
        for(int x:set_){
            v.push_back(x);
        }
        return v;        
    }
    friend ostream& operator<<(ostream& os, const IndexSet& S) {
        vector<int> v;
        for(int x:S.set_){
            v.push_back(x);
        }
        sort(v.begin(),v.end());
        bool first=true;
        for (auto& e : v){
            if(not first) os<<" ";
            first=false;
            os<<e;
        }
        return os;
    }
	inline auto begin() -> decltype(set_.begin()) {
		return set_.begin();
	}

	inline auto end() -> decltype(set_.begin()) {
		return set_.begin() + set_.size_;
	}

	inline auto begin() const -> decltype(set_.begin()) {
		return set_.begin();
	}

	inline auto end() const -> decltype(set_.begin()) {
		return set_.begin() + set_.size_;
	}
};

template<class Cost>
struct GridDistance{
    using pti=pair<Cost,int>;
    int Height;
    int Width;
    vector<vector<Cost>> cost_grid;
    GridDistance(int Height,int Width):Height(Height),Width(Width){
        cost_grid=vmake(Height,Width,Cost());
    }
    void set_cost(int h,int w,Cost cost){
        cost_grid[h][w]=cost;
    }
    vector<Cost> dijkstra(Place start){
        vector<Cost> dist(Height*Width,numeric_limits<Cost>::max());
        priority_queue<pti,vector<pti>,greater<pti>> que;

        dist[start.id()]=0;
        que.emplace(dist[start.id()],start.id());

        while(que.size()){
            Place now_place;
            Cost now_cost;
            tie(now_cost,now_place)=que.top(); que.pop();

            if(now_cost>dist[now_place.id()]) continue;

            for(int dir=0;dir<4;dir++){
                auto next_place=now_place.next_place(dir);
                if(next_place.out_grid()) continue;
                Cost next_cost=dist[now_place.id()]+cost_grid[next_place.h][next_place.w];
                if(chmin(dist[next_place.id()],next_cost)){
                    que.emplace(next_cost,next_place.id());
                }
            }
        }
        return dist;
    }

    pair<Cost,vector<char>> query_path(Place start,Place goal){
        auto dist=dijkstra(start);
        vector<Place> path;
        auto now_place=goal;
        path.push_back(now_place);
        while(now_place!=start){
            for(int dir=0;dir<4;dir++){
                auto np=now_place.next_place(dir);
                if(np.out_grid()) continue;
                if(dist[np.id()]+cost_grid[now_place.h][now_place.w]==dist[now_place.id()]){
                    now_place=np;
                    break;
                }
            }
            path.push_back(now_place);
        }
        reverse(path.begin(),path.end());
        vector<char> path_dir;
        for(int i=0;i+1<path.size();i++){
            auto p=path[i];
            for(int dir=0;dir<4;dir++){
                auto np=p.next_place(dir);
                if(np==path[i+1]){
                    path_dir.push_back(dir);
                    break;
                }
            }
        }
        return {dist[goal.id()],path_dir};
    }
};
GridDistance<int> Distance(Height,Width);

Place find_nearest(const Place &p,const vector<Place> &ps,bool easy=true){
    int distance=MAX;
    Place nearest;
    for(auto np:ps){
        int dist;
        if(easy) dist=p.dist(np);
        else dist=Distance.query_path(p,np).first;
        if(chmin(distance,dist)){
            nearest=np;
        }
    }
    return nearest;
}
template<class T>
pair<Place,T> find_nearest(Place &p,vector<pair<Place,T>> &ps,bool easy=true){
    int distance=MAX;
    pair<Place,T> nearest;
    for(auto np:ps){
        int dist;
        if(easy) dist=p.dist(np.first);
        else dist=Distance.query_path(p,np.first).first;
        if(chmin(distance,dist)){
            nearest=np;
        }
    }
    return nearest;
}
void output_move(Place &start,Place goal){
    auto path_dir=Distance.query_path(start,goal).second;
    start=goal;
    for(int dir:path_dir){
        string s="1 ";
        string c="U";
        if(dir==0) c="R";
        if(dir==1) c="D";
        if(dir==2) c="L";
        s+=c;
        ans.push_back(s);
    }
}

struct Bomb{
    int cost;
    vector<Place> ps;
};
constexpr int num_bomb=20;
auto map_input=vmake(Height,Width,'.');
vector<Bomb> bombs(num_bomb);

void input(){
    int dummy;
    cin>>dummy>>dummy;
    for(int h=0;h<Height;h++) for(int w=0;w<Width;w++){
        cin>>map_input[h][w];
    }
    for(int b=0;b<num_bomb;b++){
        cin>>bombs[b].cost;
        int size;
        cin>>size;
        bombs[b].ps.resize(size);
        cin>>bombs[b].ps;
    }
}

/*
集合被覆問題のライブラリ

メインのアルゴリズム: 焼きなまし法(近傍は[2]を参考にした)
初期解: 「新しくカバーできる箇所の個数/コスト」が最大のものを選ぶ貪欲で構築
近傍: 解の一部を削除する。カバーできていない要素をランダムに選んで、それを含んでいる集合の中で「」が最大のものを選ぶ貪欲で再構築。解の中に削除できるものがあれば削除

kanwa: 集合を少なくする([1]の3章を参考にして書いた)
narrow_downの引数degはパラメータ調整が必要

flip2(削除・挿入)とflip3(削除・削除・挿入)も書いた(https://yukicoder.me/submissions/929410)

verify: 
参考
[1]https://www.kurims.kyoto-u.ac.jp/~kyodo/kokyuroku/contents/pdf/1114-22.pdf
[2]https://onlinelibrary.wiley.com/doi/abs/10.1002/1520-6750(199510)42:7%3C1129::AID-NAV3220420711%3E3.0.CO;2-M
*/
template<class Name,class Cost,int max_num_element,int max_num_set>
struct SetCovering{
    struct Set{
        Name name;
        Cost cost;
        vector<int> elements;
        bitset<max_num_element> bits;
        Set(Name name,Cost cost,vector<int> elements):
            name(name),cost(cost),elements(elements){
                bits.reset();
                for(int element:elements){
                    bits.flip(element);
                }
            }
    };
    int num_element;
    vector<Set> sets;

    vector<int> best_solution;
    Cost best_cost;

    vector<vector<int>> sets_contain;
    vector<int> candidate_sets;

    //現在注目している解
    IndexSet<max_num_set> now_solution;
    IndexSet<max_num_element> uncovered_element;
    vector<int> cnt_covered;
    vector<int> num_uncovered_element;
    Cost now_cost;

    SetCovering(int num_element):num_element(num_element),sets_contain(num_element),
        cnt_covered(num_element),
        num_uncovered_element(max_num_set),now_cost(0){
        for(int i=0;i<num_element;i++){
            uncovered_element.insert(i);
        }
    }
    void add_set(Name name,Cost cost,vector<int> elements){
        Set set({name,cost,elements});
        sets.push_back(set);
        for(int element:set.elements){
            sets_contain[element].push_back(sets.size()-1);
        }
        candidate_sets.push_back(sets.size()-1);
        is_candidate.push_back(true);
    }
    void calc_sets_contain(){
        sets_contain.clear();
        sets_contain.resize(num_element);
        for(int id:candidate_sets){
            for(int element:sets[id].elements){
                sets_contain[element].push_back(id);
            }
        }
    }

    //set_idsの中から一番コスパ(num_uncovered/cost)が良いものを貪欲に選んで返す
    int greedy_one(const vector<int> &set_ids,bool contains_erased=true){
        int id_best=-1;
        double costper_best=-1;
        for(int id:set_ids){
            if(num_uncovered_element[id]==0) continue;
            if(contains_erased==false and sets_erase.contains(id)) continue;
            if(chmax(costper_best,double(num_uncovered_element[id])/sets[id].cost)){
                id_best=id;
            }
        }
        return id_best;
    }
    
    void add_solution(int id_set){
        for(int element:sets[id_set].elements){
            cnt_covered[element]++;
            if(cnt_covered[element]>1) continue;
            uncovered_element.remove(element);
            for(int id_set:sets_contain[element]){
                num_uncovered_element[id_set]--;
            }
        }
        now_cost+=sets[id_set].cost;
        now_solution.insert(id_set);
        if(sets_erase.contains(id_set)){
            sets_erase.remove(id_set);
        }else{
            sets_add.insert(id_set);
        }
    }
    void erase_solution(int id_set){
        for(int element:sets[id_set].elements){
            cnt_covered[element]--;
            if(cnt_covered[element]) continue;
            uncovered_element.insert(element);
            for(int id_set:sets_contain[element]){
                num_uncovered_element[id_set]++;
            }
        }
        now_cost-=sets[id_set].cost;
        now_solution.remove(id_set);
        if(sets_add.contains(id_set)){
            sets_add.remove(id_set);
        }else{
            sets_erase.insert(id_set);
        }
    }
    //0: 消した 1:追加した
    IndexSet<max_num_set> sets_add;
    IndexSet<max_num_set> sets_erase;

    void neighborhood(){
        //解から消す
        int num_erase=Rand32(1,now_solution.size()*0.1);
        for(int i=0;i<num_erase;i++){
            int id=now_solution.random();
            erase_solution(id);
        }
        bool not_first=false;
        //貪欲で実行可能解にする
        while(uncovered_element.size()){
            int element=uncovered_element.random();
            int set_id=greedy_one(sets_contain[element],not_first);
            //int set_id=greedy_one(candidate_sets);
            assert(set_id!=-1);
            add_solution(set_id);
            not_first=true;
        }
        while(true){
            int set_id=flip1();
            if(set_id==-1) break;
            erase_solution(set_id);
        }
    }
    //O(解のサイズの次数の合計)
    int flip1(){
        for(int id_set:now_solution){
            bool ok=true;
            for(int element:sets[id_set].elements){
                if(cnt_covered[element]==1){
                    ok=false;
                    break;
                }
            }
            if(ok){
                return id_set;
            }
        }
        return -1;
    }
    //焼きなまし
    void solve(int time_limit,double startTemp,bool is_hill){

        for(int s:candidate_sets){
            num_uncovered_element[s]=sets[s].elements.size();
        }
        now_cost=0;
        now_solution.clear();

        //貪欲法で初期解を作る
        while(uncovered_element.size()){
            int element=uncovered_element.random();
            int set_id=greedy_one(candidate_sets);
            add_solution(set_id);
        }

        cerr<<"cost(greedy): "<<now_cost<<endl;

        best_cost=now_cost;
        best_solution=now_solution.make_vector();

        //candidate_setsを小さくする
        //kanwa();

        cerr<<"size(candidate_sets): "<<candidate_sets.size()<<endl;

        int start_time=TIME.span();
    //焼きなまし
        //float temp_start,float temp_end,float time_start,float time_end,bool is_hill,bool minimum,int interval=1
        SimulatedAnnealing SA(startTemp,0,start_time,start_time+time_limit,is_hill,true,1);

        int cnt_update=0;
        int cnt_try=0;
        int T=10;



        while(start_time+TIME.span()<time_limit or T--){
            sets_add.clear();
            sets_erase.clear();
            Cost pre_cost=now_cost;
            //cerr<<now_cost<<endl;
            neighborhood();
            cnt_try++;
            double tol=SA.calc_tolerance();
            if(SA(now_cost-pre_cost)){
                cnt_update++;
                if(chmin(best_cost,now_cost)){
                    best_solution=now_solution.make_vector();
                }
            }else{
                assert(sets_add.size() or sets_erase.size());
                for(int id_set:sets_add){
                    erase_solution(id_set);
                }
                for(int id_set:sets_erase){
                    add_solution(id_set);
                }
                assert(pre_cost==now_cost);
            }
        }
        cerr<<cnt_update<<"/"<<cnt_try<<endl;
        cerr<<"best_cost: "<<best_cost<<endl;
    }

    vector<bool> is_candidate;

    vector<int> s;
    vector<double> relative_cost;

    //candidate_sets
    void set_relative_cost(const vector<double> &v){
        relative_cost.resize(sets.size());
        for(int id_set:candidate_sets){
            double cost=sets[id_set].cost;
            for(int element:sets[id_set].elements){
                cost-=v[element];
            }
            relative_cost[id_set]=cost;
        }
    }
    //各要素につき、上位deg個を残す
    //被ったものも数に数える（つまり、deg*num_elementより少なくなる）
    void narrow_down(int deg,const vector<double> &v){
        vector<int> pre_candidates=candidate_sets;
        set_relative_cost(v);
        candidate_sets.clear();
        is_candidate.assign(sets.size(),false);

        sort(pre_candidates.begin(),pre_candidates.end(),
            [&](int i,int j){
                return relative_cost[i]<relative_cost[j];
            });
        for(int i=0;i<min<int>(pre_candidates.size(),deg*num_element);i++){
            candidate_sets.push_back(pre_candidates[i]);
            is_candidate[i]=true;
        }

        for(int element=0;element<num_element;element++){
            vector<int> ids;
            for(int id:sets_contain[element]){
                ids.push_back(id); 
            }
            sort(ids.begin(),ids.end(),
                [&](int i,int j){
                    return relative_cost[i]<relative_cost[j];
                });
            for(int i=0;i<min<int>(deg,ids.size());i++){
                int id=ids[i];
                if(is_candidate[id]) continue;
                candidate_sets.push_back(id);
                is_candidate[id]=true;
            }
        }
        calc_sets_contain();
    }

    double calc_L(const vector<double> &v){
        set_relative_cost(v);
        s.assign(num_element,1);
        double sum=0;
        for(double x:v){
            sum+=x;
        }
        for(int set_id:candidate_sets){
            if(relative_cost[set_id]<0){
                sum+=relative_cost[set_id];
                for(int element:sets[set_id].elements){
                    s[element]--;
                }
            }
        }
        return sum;
    }

    /*
    最初: 1個の要素につき上位10個
    最終: 1個の要素につき上位5個
    */
    void kanwa(){
        assert(best_cost!=0);
        //vの初期化
        vector<double> v(num_element,numeric_limits<double>::max());
        for(auto &set:sets){
            for(int element:set.elements){
                chmin(v[element],double(set.cost)/set.elements.size());
            }
        }
        //1個の要素につき上位10個
        narrow_down(20,v);

        //vの更新
        int T=100;
        double lamda=4;
        int beta=15;
        double rho=1.2;
        
        double max_L=calc_L(v);
        vector<int> max_s=s;
        vector<double> max_v=v; //Lを最大にするv

        while(T--){
            double L=calc_L(v);
            if(chmax(max_L,L)){
                max_v=v;
            }else{
                L=max_L;
                v=max_v;
                s=max_s;
                lamda/=rho;
            }
            double s2_sum=0;
            for(int x:s){
                s2_sum+=x*x;
            }
            for(int element=0;element<num_element;element++){
                v[element]=max(0.0,v[element]+lamda*(best_cost-L)/s2_sum*s[element]);
                //v[element]=max<double>(0.0,v[element]+0.1*s[element]);
            }
            cerr<<"L: "<<L<<endl;
            assert(L<=best_cost);
        }
        cerr<<"max_L: "<<max_L<<endl;

        
        //1個の要素につき上位5個
        narrow_down(10,max_v);
    }

};

struct HashMap{
    vector<u64> table;
    HashMap(int size):table(size){
        for(auto &x:table){
            x=Rand64();
        }
    }
    u64 operator[](int id){
        return table[id];
    }
};
HashMap hashmap(1000);
Place center_shop;
template<class Action,class Score,class Hash>
struct State{
    static vector<Place> place_shop;
    static vector<Place> place_action;
    static vector<vector<int>> bakuha_shop;
    vector<pii> history_actions;
    vector<bool> is_bakuha_shop;
    vector<bool> is_done_action;
    int score;
    Place now_place;

    State(int action_size,int shop_size):is_bakuha_shop(shop_size),is_done_action(action_size),score(0),now_place(0,0){
        
    }

    Score calc_score(){
        return score;
    }
    Hash calc_hash(){
        u64 hash=0;
        for(int i=0;i<is_done_action.size();i++){
            if(is_done_action[i]==false) continue;
            hash^=hashmap[i];
        }
        hash^=hashmap[100+now_place.id()];
        return hash;
    }

    void apply(Action action){
        int min_dist=MAX;
        int nearest_shop=-1;
        for(int i=0;i<is_bakuha_shop.size();i++){
            if(is_bakuha_shop[i]) continue;
            int dist=now_place.dist(place_shop[i])+place_shop[i].dist(place_action[action])*2;
            if(chmin(min_dist,dist)){
                nearest_shop=i;
            }
        }

        score+=min_dist;
        history_actions.emplace_back(action,nearest_shop);
        is_done_action[action]=true;

        //shopを爆破
        for(int shop:bakuha_shop[action]){
            is_bakuha_shop[shop]=true;
        }
        //移動
        now_place=place_action[action];
        if(history_actions.size()==is_done_action.size()){
            score+=now_place.dist(center_shop);
        }
    }

    vector<Action> find_next_actions(){
        vector<int> candidate;
        int size_action=is_done_action.size();
        for(int i=0;i<size_action;i++){
            if(is_done_action[i]) continue;
            candidate.push_back(i);
        }
        sort(candidate.begin(),candidate.end(),
            [&](int i,int j){
                return now_place.dist(place_action[i])<now_place.dist(place_action[j]);
            }
        );
        if(candidate.size()>4){
            candidate.resize(4);
        }
        return candidate;
    }
    bool operator<(State &s){
        return calc_score()<s.calc_score();
    }
};
template<typename T1, typename T2,typename T3>
vector<Place> State<T1, T2,T3>::place_shop;

template<typename T1, typename T2,typename T3>
vector<Place> State<T1, T2,T3>::place_action;

template<typename T1, typename T2,typename T3>
vector<vector<int>> State<T1, T2,T3>::bakuha_shop;

template<class Action,class Score,class State,class Hash>
vector<pii> beam_search(State init_state,int width_beam,int max_depth){
    vector<State> states={init_state};
    vector<State> next_states;

    for(int depth=0;depth<max_depth;depth++){
        next_states.clear();
#ifdef ONLINE_JUDGE
        if(TIME.span()>2850){
            width_beam=1;
        }
#endif

        for(int i=0;i<min<int>(width_beam,states.size());i++){
            vector<Action> next_actions=states[i].find_next_actions();
            for(Action action:next_actions){
                State state=states[i];
                state.apply(action);
                next_states.push_back(state);
            }
        }
        sort(next_states.begin(),next_states.end());
        states.clear();
        unordered_set<u64> set;
        for(int i=0;i<next_states.size() and states.size()<width_beam;i++){
            auto &state=next_states[i];
            u64 hash=state.calc_hash();
            if(set.count(hash)){
                continue;
            }
            set.insert(hash);
            states.push_back(state);
        }
    }

    sort(states.begin(),states.end());
    cerr<<"score: "<<states[0].calc_score()<<endl;
    return states[0].history_actions;
}

/*
店は最後に爆破する
*/
void solve(){
    input();
//set coveringのインスタンスを作る
    auto id_map=vmake(Height,Width,-1);
    vector<Place> place_shops;
    int num_hakai=0;

    //一番コスパが高い爆弾の特定
    int cheapest_bomb=-1;
    double cheapest_cost=-1;
    for(int i=0;i<num_bomb;i++){
        if(chmax(cheapest_cost,bombs[i].ps.size()/bombs[i].cost)){
            cheapest_bomb=i;
        }
    }
{
    int min_center_dist=10000;
    int id=0;
    for(int h=0;h<Height;h++) for(int w=0;w<Width;w++){
        if(map_input[h][w]=='.'){
            Distance.set_cost(h,w,1);
        }else{
            Distance.set_cost(h,w,2);
        }
        if(map_input[h][w]=='@'){
            if(chmin(min_center_dist,Place(25,25).dist(Place(h,w)))){
                center_shop=Place(h,w);
            }
            place_shops.emplace_back(h,w);
        }
        if(map_input[h][w]!='#') continue;
    }
    for(int h=0;h<Height;h++) for(int w=0;w<Width;w++){
        if(map_input[h][w]=='.') continue;
        if(center_shop==Place(h,w)) continue;
        bool ex=true;
        for(auto dp:bombs[cheapest_bomb].ps){
            auto np=center_shop;
            np.move(dp);
            if(np==Place(h,w)) ex=false;
        }
        if(ex){
            id_map[h][w]=id++;
        }
    }
    num_hakai=id;
}
    using pPi=pair<Place,int>;
    SetCovering<pPi,int,2500,50000> SC(num_hakai);
    for(int ph=0;ph<Height;ph++){
        for(int pw=0;pw<Width;pw++){
            Place p(ph,pw);
            auto nearest_shop=find_nearest(p,place_shops,true);
            int dist=p.dist(nearest_shop);
            for(int b=0;b<num_bomb;b++){
                vector<int> ids;
                bool ok=true;
                for(auto dp:bombs[b].ps){
                    auto np=p;
                    np.move(dp);
                    if(np==center_shop){
                        ok=false;
                    }
                    if(np.out_grid() or id_map[np.h][np.w]==-1){
                        continue;
                    }
                    ids.push_back(id_map[np.h][np.w]);
                }
                if(not ok) continue;
                SC.add_set(pPi(p,b),bombs[b].cost+4*dist*2,ids);
            }
        }
    }
    SC.solve(2000,10,false);
    using pPi=pair<Place,int>;
    vector<pPi> names;
    for(int id:SC.best_solution){
        names.push_back(SC.sets[id].name);
    }
    Place place_now(0,0);

//ビームサーチ
// template<>
// vector<Place> State<int, int>::place_shop;
// template<>
// vector<Place> State<int, int>::place_action;
// template<>
// vector<int> State<int, int>::bakuha_shop;
    using State=State<int,int,u64>;
    State state(names.size(),place_shops.size());

    state.place_shop=place_shops;
    for(auto name:names){
        state.place_action.push_back(name.first);
    }
    state.bakuha_shop.resize(names.size());
    for(int a=0;a<names.size();a++){
        for(int i=0;i<place_shops.size();i++){
            auto shop=place_shops[i];
            bool hakai=false;
            for(auto dp:bombs[names[a].second].ps){
                auto np=names[a].first;
                np.move(dp);
                if(shop==np) hakai=true;
            }
            if(hakai){
                state.bakuha_shop[a].push_back(i);
            }
        }
    }
    auto actions_beam=beam_search<int,int,State,u64>(state,3000,names.size());


    for(int i=0;i<actions_beam.size();i++){
        //次のターゲット
        pPi name_nearest=names[actions_beam[i].first];
        //一番近くの店に行って必要な爆弾を買う
        Place nearest_shop=state.place_shop[actions_beam[i].second];
        output_move(place_now,nearest_shop);
        ans.push_back("2 "+to_string(name_nearest.second+1));

        //ターゲットのところに行って爆弾を落とす
        output_move(place_now,name_nearest.first);
        ans.push_back("3 "+to_string(name_nearest.second+1));

        //ターゲットを消す
        // for(int i=0;i<names.size();i++){
        //     if(name_nearest.first==names[i].first and name_nearest.second==names[i].second){
        //         names.erase(names.begin()+i);
        //         break;
        //     }
        // }
        vector<Place> next_place_shops;
        for(auto shop:place_shops){
            bool hakai=false;
            for(auto dp:bombs[name_nearest.second].ps){
                auto np=place_now;
                np.move(dp);
                if(shop==np) hakai=true;
            }
            if(not hakai) next_place_shops.push_back(shop);
        }
        for(auto dp:bombs[name_nearest.second].ps){
            auto np=place_now;
            np.move(dp);
            if(np.out_grid()) continue;
            if(Distance.cost_grid[np.h][np.w]==2){
                Distance.set_cost(np.h,np.w,1);
            }
        }

        place_shops=next_place_shops;
    }
    cerr<<-1<<endl;


    while(place_shops.size()){
        //一番近い店に行って一番安い爆弾を使う
        Place nearest_shop=find_nearest(place_now,place_shops);
        output_move(place_now,nearest_shop);
        ans.push_back("2 "+to_string(cheapest_bomb+1));
        ans.push_back("3 "+to_string(cheapest_bomb+1));

        vector<Place> next_place_shops;
        for(auto shop:place_shops){
            bool hakai=false;
            for(auto dp:bombs[cheapest_bomb].ps){
                auto np=nearest_shop;
                np.move(dp);
                if(shop==np) hakai=true;
            }
            if(not hakai) next_place_shops.push_back(shop);
        }

        place_shops=next_place_shops;
    }


    cout<<ans.size()<<endl;
    for(auto s:ans){
        cout<<s<<"\n";
    }
    
#ifndef ONLINE_JUDGE
    testTimer.output();
    testCounter.output();
    cerr<<TIME.span()<<"ms"<<endl;
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