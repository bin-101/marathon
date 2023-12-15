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
    //ソートされた状態を保つように挿入
    void insert(const T &e){
        int ng=-1,ok=size_;
        while(ok-ng!=1){
            int mid=(ok+ng)/2;
            if(array_[mid]>e) ok=mid;
            else ng=mid;
        }
        for(int i=size_;i>ok;i--){
            array_[i]=array_[i-1];
        }
        array_[ok]=e;
        size_++;
    }
    //eをこえる一番左の添字
    int find_binary_search(const T &e)const{
        int ng=-1,ok=size_;
        while(ok-ng!=1){
            int mid=(ok+ng)/2;
            if(array_[mid]>=e) ok=mid;
            else ng=mid;
        }
        return ok;
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
    const inline T& back()const{
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
    bool operator==(const DynamicArray &v){
        if(size_!=v.size_) return false;
        for(int i=0;i<size_;i++){
            if(array_[i]!=v[i]){
                return false;
            }
        }
        return true;
    }
    bool operator==(const vector<T> &v){
        if(size_!=v.size()) return false;
        for(int i=0;i<size_;i++){
            if(array_[i]!=v[i]){
                return false;
            }
        }
        return true;
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

template<class T,class U>
T linear_function(U x,U start_x,U end_x,T start_value,T end_value){
    if(x>=end_x) return end_value;
    if(x<=start_x) return start_value;
    return start_value+(end_value-start_value)*(x-start_x)/(end_x-start_x);
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

int Height,Width;
vector<vector<vector<bool>>> ok_move_input;
struct Place{
    int h,w;
    Place(){}
    Place(int h,int w):h(h),w(w){
 
    }
    Place(int idx):h(idx/Width),w(idx%Width){
 
    }
    int dist(const Place &np){
        return sqrt(double(h-np.h)*(h-np.h) + double(w-np.w)*(w-np.w))+1;
    }

    Place pre_place(int dir){
        return Place(h-dh[dir],w-dw[dir]);
    }    
    Place next_place(int dir){
        if(ok_move_input[h][w][dir]==false){
            return Place(-1,-1);
        }
        return Place(h+dh[dir],w+dw[dir]);
    }
    void move(int dir){
        if(ok_move_input[h][w][dir]==false){
            h=-1;
            w=-1;
            return;
        }
        h+=dh[dir];
        w+=dw[dir];
    }
    void remove(int dir){
        h-=dh[dir];
        w-=dw[dir];        
    }
    bool ok_move(int dir){
        return ok_move_input[h][w][dir];
    }
    bool out_grid()const{
        return h>=Height or w>=Width or h<0 or w<0;
    }
    bool operator==(const Place &p){
        return h==p.h and w==p.w;
    }
    bool operator!=(const Place &p){
        return h!=p.h or w!=p.w;
    }
    int id(){
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

vector<vector<ll>> dust_input;
vector<Place> solution;
vector<vector<DynamicArray<int,1000>>> solution_visit_day;

vector<vector<DynamicArray<int,1000>>> make_solution_visit_day(const vector<Place> &solution){
    auto solution_visit_day=vmake(Height,Width,DynamicArray<int,1000>(0));
    for(int i=0;i<solution.size();i++){
        if(solution[i].out_grid()) continue;
        solution_visit_day[solution[i].h][solution[i].w].push_back(i);
    }
    return solution_visit_day;
}

template<class Cost>
struct GridDistance{
    using pti=pair<Cost,int>;
    int Height;
    int Width;
    vector<vector<Cost>> cost_grid;
    vector<vector<Cost>> dist_matrix;
    GridDistance(int Height,int Width):Height(Height),Width(Width){
        cost_grid=vmake(Height,Width,Cost(1));
    }
    void set_cost(int h,int w,Cost cost){
        cost_grid[h][w]=cost;
    }
    void build(){
        dist_matrix=vmake(Height*Width,Height*Width,numeric_limits<Cost>::max());
        for(int h=0;h<Height;h++) for(int w=0;w<Width;w++){
            dist_matrix[Place(h,w).id()]=dijkstra(Place(h,w));
        }
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

    vector<Place> query_path(Place start,Place goal){
        auto &dist=dist_matrix[start.id()];
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
        return path;
    }
    Cost dist(Place start,Place goal){
        return dist_matrix[start.id()][goal.id()];
    }
};
GridDistance<int> Distance(10,10);

void input(){
    cin>>Height;
    Width=Height;
    ok_move_input=vmake(Height,Width,4,false);

    auto below=vmake(Height-1,Width,'a');
    cin>>below;

    for(int h=0;h<Height-1;h++) for(int w=0;w<Width;w++){
        if(below[h][w]=='1') continue;
        ok_move_input[h][w][Down]=true;
        ok_move_input[h+1][w][Up]=true;
    }

    auto right=vmake(Height,Width-1,'a');
    cin>>right;

    for(int h=0;h<Height;h++) for(int w=0;w<Width-1;w++){
        if(right[h][w]=='1') continue;
        ok_move_input[h][w][Right]=true;
        ok_move_input[h][w+1][Left]=true;
    }

    dust_input=vmake(Height,Width,1LL);
    cin>>dust_input;
}

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
ll sum_sequense(ll num){
    return num*(num-1)/2;
}
ll calc_score(const DynamicArray<int,1000> &days,ll dust,ll solution_len){
    if(days.size()==0) return MAX;
    ll sum=0;
    for(int i=0;i<days.size();i++){
        ll sa=-1;
        if(i+1==days.size()){
            sa=solution_len+days[0]-days[i];
        }else{
            sa=days[i+1]-days[i];
        }
        sum+=sum_sequense(sa)*dust;
    }
    return sum;
}
//solutionのスコアを計算
ll calc_score(const vector<Place> &solution){
    testTimer.start("calc_score");
    auto visit=vmake(Height,Width,vector<int>());

    int day=-1;
    for(int i=0;i<solution.size();i++){
        if(solution[i].out_grid()) continue;
        visit[solution[i].h][solution[i].w].push_back(i);
    }
    /*for(Place p:solution){
        day++;
        if(p.out_grid()) continue;
        visit[p.h][p.w].push_back(day);
    }*/

    ll sum=0;
    ll len=solution.size();
    //cerr<<sum_sequense(len)<<endl;
    for(int h=0;h<Height;h++) for(int w=0;w<Width;w++){
        if(visit[h][w].size()==0){
            testTimer.end("calc_score");
            sum+=MAX;
            continue;
        }
        visit[h][w].push_back(len+visit[h][w][0]);
        for(int i=0;i+1<visit[h][w].size();i++){
            ll sa=visit[h][w][i+1]-visit[h][w][i];
            sum+=sum_sequense(sa)*dust_input[h][w];
        }
        visit[h][w].pop_back();
    }
#ifndef NDEBUG
    ll sum2=0;
    auto visit_day=make_solution_visit_day(solution);
    for(int h=0;h<Height;h++) for(int w=0;w<Width;w++){
        assert(visit_day[h][w]==visit[h][w]);
        sum2+=calc_score(visit_day[h][w],dust_input[h][w],solution.size());
    }
    assert(sum==sum2);
#endif
    return sum;
}

void make_initial_solution(){
    solution.push_back(Place(0,0));
    auto visited=vmake(Height,Width,false);
    visited[0][0]=true;
    Place now_place(0,0);
    while(true){
        Place next_place(-1,-1);
        int near_dist=MAX;
        for(int h=0;h<Height;h++) for(int w=0;w<Width;w++){
            if(visited[h][w]) continue;
            if(chmin(near_dist,Distance.dist(now_place,Place(h,w)))){
                next_place=Place(h,w);
            }
        }
        if(next_place.h==-1) break;
        assert(next_place!=Place(-1,-1));
        auto route=Distance.query_path(now_place,next_place);
        for(int i=1;i<route.size();i++){
            solution.push_back(route[i]);
        }
        visited[next_place.h][next_place.w]=true;
        now_place=next_place;
    }
    if(now_place!=Place(0,0)){
        auto route=Distance.query_path(now_place,Place(0,0));
        for(int i=1;i<route.size();i++){
            solution.push_back(route[i]);
        }
    }
    auto solution2=solution;
    for(int i=1;i<solution2.size();i++){
        solution.push_back(solution2[i]);
    }
    //solution_visit_day
    solution_visit_day=vmake(Height,Width,DynamicArray<int,1000>(0));
    for(int i=0;i<solution.size();i++){
        solution_visit_day[solution[i].h][solution[i].w].push_back(i);
    }
}
void output(){
    vector<char> c={'R','D','L','U'};
    vector<char> path_dir;
    for(int i=0;i+1<solution.size();i++){
        auto p=solution[i];
        for(int dir=0;dir<4;dir++){
            auto np=p.next_place(dir);
            if(np==solution[i+1]){
                path_dir.push_back(c[dir]);
                break;
            }
        }
    }
    cout<<path_dir<<endl;
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
    int temp_type;
    

    //SimulatedAnnealing(){}
    SimulatedAnnealing(float temp_start,float temp_end,float time_start,float time_end,bool is_hill,bool minimum,int temp_type=2,int interval=1):
        temp_start(temp_start),temp_end(temp_end),time_start(time_start),time_end(time_end),
        is_hill(is_hill),minimum(minimum),temp_type(temp_type),interval(interval),temp_now(temp_start),cnt_calc_temp(0){
    }
    float calc_temp(){
        if(cnt_calc_temp%interval==0){
            float progress=float(TIME.span()-time_start)/(time_end-time_start);
            if(progress>1.0) progress=1.0;
            if(temp_type==0){//線形
                temp_now=temp_start*(1.0-progress)+temp_end*progress;
            }else if(temp_type==1){ //https://atcoder.jp/contests/ahc014/submissions/35326979
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

//float temp_start,float temp_end,float time_start,float time_end,bool is_hill,bool minimum,int temp_type=0,int interval=1
SimulatedAnnealing SA(10000,0,0,1800,false,true);

//サイズが0になったら+MAX
ll calc_different_erase(const DynamicArray<int,1000> &days,int erase_day,int solution_len,ll dust){
    assert(days.size());
    if(days.size()==1){
        return ll(MAX)-sum_sequense(solution_len)*dust;
    }
    int erase_id=days.find_binary_search(erase_day);
    int pre_day=-1,post_day=-1;
    if(erase_id==0){
        pre_day=int(days.back())-solution_len;
    }else{
        pre_day=days[erase_id-1];
    }
    if(erase_id+1==days.size()){
        post_day=days[0]+solution_len;
    }else{
        post_day=days[erase_id+1];
    }
    ll ret=sum_sequense(post_day-pre_day)-sum_sequense(erase_day-pre_day)-sum_sequense(post_day-erase_day);
    return ret*dust;
}

//サイズが0になったら+MAX
ll calc_different_add(const DynamicArray<int,1000> &days,const vector<int> &add_days,int solution_len,ll dust,int start_day,int num_interval){
    ll ret=0;
    if(days.size()==0){
        if(add_days.size()){
            for(int i=0;i+1<add_days.size();i++){
                ret+=sum_sequense(add_days[i+1]-add_days[i]);
            }
            ret+=sum_sequense(add_days[0]+solution_len-add_days.back()+num_interval);
            ret=-MAX+ret*dust;
        }else{
            ret=0;
        }
    }else{
        int pre_day=-1,post_day=-1;

        int pre_id=days.find_binary_search(start_day);
        if(pre_id==0){
            pre_day=days.back()-solution_len;
        }else{
            pre_day=days[pre_id-1];
        }

        int post_id=days.find_binary_search(start_day);
        if(post_id==days.size()){
            post_day=solution_len+days[0];
        }else{
            post_day=days[post_id];
        }

        ret=-sum_sequense(post_day-pre_day);

        if(add_days.size()==0){
            ret+=sum_sequense(post_day-pre_day+num_interval);
        }else{
            //最初
            ret+=sum_sequense(add_days[0]-pre_day);
            ret+=sum_sequense(post_day-add_days.back()+num_interval);
            for(int i=0;i+1<add_days.size();i++){
                ret+=sum_sequense(add_days[i+1]-add_days[i]);
            }
        }
        ret*=dust;
    }

#ifndef NDEBUG
    ll pre_score=calc_score(days,dust,solution_len);
    DynamicArray<int,1000> change_days;
    for(int day:days){
        if(day>=start_day) break;
        change_days.push_back(day);
    }
    for(int day:add_days){
        change_days.push_back(day);
    }
    for(int day:days){
        if(day<start_day) continue;
        change_days.push_back(day+num_interval);
    }
    ll post_score=calc_score(change_days,dust,solution_len+num_interval);
    assert(post_score-pre_score==ret);
#endif
    return ret;
}

vector<vector<vector<int>>> add_day=vmake(40,40,vector<int>());
void neighbor(vector<Place> &now_solution,ll &now_cost){
#ifndef NDEBUG
{
    auto pre_solution=now_solution;
    assert(calc_score(now_solution)==now_cost);
    auto visit_day=make_solution_visit_day(now_solution);
    for(int h=0;h<Height;h++) for(int w=0;w<Width;w++){
        assert(visit_day[h][w]==solution_visit_day[h][w]);
    }
}
#endif
    int start_id=Rand32(0,now_solution.size()-10);
    int goal_id=Rand32(start_id+2,start_id+10);

    //startからgoalまでの道を作る
    //近くのよるマスを決めて、そこに最短経路で行って最短経路でgoalに行く
    Place start_place=now_solution[start_id];
    Place goal_place=now_solution[goal_id];
    Place mid_place;
    vector<Place> candidate;
    for(int h=0;h<Height;h++) for(int w=0;w<Width;w++){
        if(Distance.dist(start_place,Place(h,w))<=5){
            candidate.emplace_back(h,w);
        }
    }
    mid_place=candidate[Rand32(candidate.size())];
    auto route1=Distance.query_path(start_place,mid_place);
    auto route2=Distance.query_path(mid_place,goal_place);
    vector<Place> route;
    for(int i=1;i<route1.size();i++){
        route.push_back(route1[i]);
    }
    for(int i=1;i+1<route2.size();i++){
        route.push_back(route2[i]);
    }

    //ここで差分計算を頑張る
    //trueなら追加
    struct ChangeDay{
        bool is_add;
        Place p;
        int day;
    };
    ll change_sum=0;
    //まず消す
    vector<ChangeDay> change_day;
    for(int i=start_id+1;i<goal_id;i++){
        Place p=now_solution[i];
        change_day.push_back({false,p,i});
        //cerr<<"erase: "<<calc_different_erase(solution_visit_day[p.h][p.w],i,now_solution.size(),dust_input[p.h][p.w])<<endl;
        ll change_value=calc_different_erase(solution_visit_day[p.h][p.w],i,now_solution.size(),dust_input[p.h][p.w]);
        change_sum+=change_value;
#ifndef NDEBUG
        ll pre_score=calc_score(solution_visit_day[p.h][p.w],dust_input[p.h][p.w],now_solution.size());
#endif

        solution_visit_day[p.h][p.w].erase(i);

#ifndef NDEBUG
        ll post_score=calc_score(solution_visit_day[p.h][p.w],dust_input[p.h][p.w],now_solution.size());
        assert(post_score-pre_score==change_value);
#endif
    }
#ifndef NDEBUG
{
    auto erase_solution=now_solution;
    for(int i=start_id+1;i<goal_id;i++){
        erase_solution[i]=Place(-1,-1);
    }
    auto visit_day=make_solution_visit_day(erase_solution);
    for(int h=0;h<Height;h++) for(int w=0;w<Width;w++){
        assert(visit_day[h][w]==solution_visit_day[h][w]);
    }
    //cerr<<calc_score(now_solution)+change_sum<<endl;
    //cerr<<calc_score(erase_solution)<<endl;
    assert(abs(calc_score(now_solution)+change_sum-calc_score(erase_solution))==0);
}
#endif

    ll num_interval=int(route.size())-int(change_day.size());
    //add_dayを作成
    for(int i=0;i<route.size();i++){
        add_day[route[i].h][route[i].w].push_back(start_id+i+1);
    }
    //追加するのを差分計算　同時に間が開くのも考慮する
    for(int h=0;h<Height;h++) for(int w=0;w<Width;w++){
        ll change_value=calc_different_add(solution_visit_day[h][w],add_day[h][w],solution.size(),dust_input[h][w],start_id+1,num_interval);
        change_sum+=change_value;
    }

#ifndef NDEBUG
{
    //next_solution作成
    vector<Place> next_solution;
    for(int i=0;i<=start_id;i++){
        next_solution.push_back(now_solution[i]);
    }
    for(auto p:route){
        next_solution.push_back(p);
    }
    for(int i=goal_id;i<now_solution.size();i++){
        next_solution.push_back(now_solution[i]);
    }
    //cerr<<calc_score(next_solution)<<endl;
    //cerr<<calc_score(now_solution)<<endl;
    //cerr<<calc_score(now_solution)+change_sum-calc_score(next_solution)<<endl;
    assert(calc_score(now_solution)+change_sum==calc_score(next_solution));
    //cerr<<"yay!"<<endl;
}
#endif
    ll real_now_cost=now_cost/now_solution.size();
    ll real_next_cost=(now_cost+change_sum)/(ll(now_solution.size())+num_interval);

    //cerr<<real_next_cost-real_now_cost<<endl;

    if(not SA(real_next_cost-real_now_cost)){
        reverse(change_day.begin(),change_day.end());
        //元に戻す
        //solution_visit_dayを変化させる
        for(auto c:change_day){
            if(c.is_add){
                solution_visit_day[c.p.h][c.p.w].erase(c.day);
            }else{
                solution_visit_day[c.p.h][c.p.w].insert(c.day);
            }
        }
        for(int h=0;h<Height;h++) for(int w=0;w<Width;w++){
            add_day[h][w].clear();
        }
        return;
    }
    //now_cost
    now_cost+=change_sum;
    
    //solution_visit_day
    for(int h=0;h<Height;h++) for(int w=0;w<Width;w++){
        //まず足す
        for(auto &day:solution_visit_day[h][w]){
            if(day>start_id){
                day+=int(route.size())-int(change_day.size());
            }
        }
        //挿入
        for(int day:add_day[h][w]){
            solution_visit_day[h][w].insert(day);
        }
        add_day[h][w].clear();
    }
    //next_solution作成
    vector<Place> next_solution;
    for(int i=0;i<=start_id;i++){
        next_solution.push_back(now_solution[i]);
    }
    for(auto p:route){
        next_solution.push_back(p);
    }
    for(int i=goal_id;i<now_solution.size();i++){
        next_solution.push_back(now_solution[i]);
    }
    now_solution=next_solution;
}

void solve(){
    //適当に解を作る
    //一部分を壊して作りなおす

    input();

    Distance=GridDistance<int>(Height,Width);
    Distance.build();

    make_initial_solution();
/*
{
    vector<int> pre_solution;
    cerr<<calc_score(pre_solution)<<endl;
    ll change_sum=0;
    auto add_day=vmake(Height,Width,vector<int>());
    for(int i=0;i<soltuion.size();i++){
        add_day[solution[i].h][solution[i].w].push_back(i);
    }
    for(int h=0;h<Height;h++) for(int w=0;w<Width;w++){
        calc_different_add(DynamicArray<int,1000>(),add_day[h][w],0,)
    }
}
*/
    ll now_cost=calc_score(solution);

    int T=10;
    while(TIME.span()<1800){
        neighbor(solution,now_cost);
    }

    output();
#ifndef ONLINE_JUDGE
    testTimer.output();
    testCounter.output();
    cerr<<TIME.span()<<"ms"<<endl;
    cerr<<"score: "<<ll(calc_score(solution)/solution.size())<<endl;
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