/*
verify: https://atcoder.jp/contests/ahc020/submissions/42299239
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
        float prob=calc_prob(diff*-1);
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

struct Place{
    int h,w;
    Place(){}
    Place(int h,int w):h(h),w(w){
 
    }
    bool operator==(Place &p){
        return h==p.h and w==p.w;
    }
    int dist(Place &np){
        return sqrt(double(h-np.h)*(h-np.h) + double(w-np.w)*(w-np.w))+1;
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

constexpr int num_cast=100;
int num_edge;
int num_home;

vector<Place> place_cast(num_cast);
vector<Place> place_home;

WeightGraph<ll> graph;

vector<Edge<ll>> edges_input;

vector<vector<int>> near_cast;

template<class T>
pair<T,vector<int>> Kruskal(vector<Edge<T>> edges,int V,vector<bool> used){
    sort(edges.begin(),edges.end(),[](const Edge<T> &a,const Edge<T> &b){
        return (a.cost<b.cost);
    });
    UnionFind uni(V);
    T ans=0;
    vector<int> tree(num_edge);
    int cnt=0;
    vector<int> cnt_deg(V);
    vector<vector<int>> cast_edges(V);
    for(auto &e:edges){
        if(uni.unite(e.from,e.to)){
            tree[e.id]=1;
            ans+=e.cost;
            cnt++;
            cnt_deg[e.from]++;
            cnt_deg[e.to]++;
            cast_edges[e.from].push_back(e.id);
            cast_edges[e.to].push_back(e.id);
        } 
    }
    int TT=100;
    while(TT--){
        for(int v=0;v<V;v++){
            if(cnt_deg[v]==1 and used[v]==false){
                //cerr<<-1<<endl;
                int e_id=cast_edges[v][0];

                int to=edges_input[e_id].from;
                if(to==v) to=edges_input[e_id].to;

                tree[e_id]=0;

                cnt_deg[to]--;
                cnt_deg[v]=0;
                auto it=find(cast_edges[to].begin(),cast_edges[to].end(),e_id);
                cast_edges[to].erase(it);
                ans-=edges_input[e_id].cost;
            }
        }
    }

    //if(cnt!=V-1) ans=-1;
    return make_pair(ans,tree);
}

void input(){
    int dummy;
    cin>>dummy>>num_edge>>num_home;

    for(int i=0;i<num_cast;i++){
        cin>>place_cast[i];
    }
    graph.resize(num_home);
    for(int i=0;i<num_edge;i++){
        int u,v,w;
        cin>>u>>v>>w;
        u--; v--; w--;
        graph[u].emplace_back(v,w,i);
        graph[v].emplace_back(u,w,i);
        edges_input.emplace_back(u,v,w,i);
    }
    place_home.resize(num_home);
    for(int i=0;i<num_home;i++){
        cin>>place_home[i];
    }


    near_cast.resize(num_home);
    for(int h=0;h<num_home;h++){
        for(int i=0;i<num_cast;i++){
            near_cast[h].push_back(i);
        }
        sort(near_cast[h].begin(),near_cast[h].end(),
            [&](int c1,int c2){
                return place_home[h].dist(place_cast[c1]) < place_home[h].dist(place_cast[c2]);
            });
    }
}
void output(vector<int> edge_id,vector<ll> power){
    cout<<power<<endl;
    cout<<edge_id<<endl;
}

pair<int,ll> greedy_assign(int home_id,vector<ll> &power,int ex_id){
    ll minimum_cost=INF;
    int id_cast=-1;
    for(int i=0;i<OP::greedy_candidate;i++){
        int c=near_cast[home_id][i];
        if(ex_id==c) continue;
        ll need_power=place_cast[c].dist(place_home[home_id]);
        if(need_power>5000) break;
        if(chmin(minimum_cost,need_power*need_power-power[c]*power[c])){
            id_cast=c;
        }
    }
    return {id_cast,place_cast[id_cast].dist(place_home[home_id])};
}

ll diff_square(ll pre,ll post){
    return post*post-pre*pre;
}

//https://core.ac.uk/download/pdf/82609861.pdf
template<class T>
struct SteinerTree{
    struct edge {
        int to;
        T cost;
        int id;
        edge()=default;
        edge(int to, T cost,int id) : to(to), cost(cost), id(id) {}
    };
    struct Edge {
        int from, to,id;
        T cost;
        Edge(int from,int to,T cost,int id):from(from),to(to),cost(cost),id(id){}
        Edge()=default;

        bool operator<(const Edge &e){
            return cost<e.cost;
        }
        bool operator<=(const Edge &e){
            return cost<=e.cost;
        }
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

    int num_vertex_;
    int num_edge_;
    WeightGraph<T> graph_;
    vector<Edge> edges_;
    bool is_sorted=true;

    SteinerTree(int num_vertex):num_vertex_(num_vertex),graph_(num_vertex){
        
    }

    void add_edge(int u,int v,T cost,int id=-1){
        if(id==-1) id=num_edge_;
        is_sorted=false;
        num_edge_++;
        edges_.emplace_back(u,v,cost,id);
        graph_[u].emplace_back(v,cost,id);
        graph_[v].emplace_back(u,cost,id);
    }
    void sort_edges(){
        sort(edges_.begin(),edges_.end());
        is_sorted=true;
    }

    //terminals同士を結ぶ辺しか考えない
    //連結しなかったら-1を返す
    pair<T,vector<int>> kruskal(vector<bool> is_terminal){
        if(is_sorted==false){
            is_sorted=true;
            sort(edges_.begin(),edges_.end());
        }

        UnionFind uni(num_vertex_);
        int size_terminal=0;
        for(bool t:is_terminal){
            if(t) size_terminal++;
        }
        T sum_cost=0;
        vector<int> edges_used;

        for(auto e:edges_){
            if(is_terminal[e.from]==false or is_terminal[e.to]==false) continue;
            if(uni.unite(e.from,e.to)){
                size_terminal--;
                sum_cost+=e.cost;
                edges_used.push_back(e.id);
            }
        }
        if(size_terminal>1) return {-1,edges_used};
        return {sum_cost,edges_used};
    }

    //解のコストの合計、使った辺のid
    //O(|terminals|*(N+MlogN) + N)
    pair<T,vector<int>> minimum_path_heuristic(vector<int> terminals){
        if(terminals.size()==0){
            return {0,{}};
        }
        vector<bool> is_contain(num_vertex_,false);
        vector<bool> is_terminal(num_vertex_,false);
        for(int terminal:terminals){
            is_terminal[terminal]=true;
        }

        is_contain[terminals[0]]=true;
        vector<int> tree={terminals[0]};
        vector<int> edges_used;
        T sum_cost=0;

        int loops=terminals.size()-1;
        
        while(loops--){
            priority_queue<pair<T,int>,vector<pair<T,int>>,greater<pair<T,int>>> que;
            vector<T> dist(num_vertex_,numeric_limits<T>::max());
            for(int v:tree){
                dist[v]=0;
                que.emplace(0,v);
            }
            int next_terminal=-1;
            while(que.size()){
                int now_vertex;
                T now_cost;
                tie(now_cost,now_vertex)=que.top(); que.pop();

                if(now_cost>dist[now_vertex]) continue;
                if(is_terminal[now_vertex] and is_contain[now_vertex]==false){
                    next_terminal=now_vertex;
                    break;
                }

                for(auto e:graph_[now_vertex]){
                    if(chmin(dist[e.to],dist[now_vertex]+e.cost)){
                        que.emplace(dist[e.to],e.to);
                    }
                }

            }
            assert(next_terminal!=-1);

            int now_vertex=next_terminal;
            while(is_contain[now_vertex]==false){
                tree.push_back(now_vertex);
                is_contain[now_vertex]=true;
                for(auto e:graph_[now_vertex]){
                    if(dist[e.to]==numeric_limits<T>::max()) continue;
                    if(dist[e.to]+e.cost==dist[now_vertex]){
                        edges_used.push_back(e.id);
                        now_vertex=e.to;
                        break;
                    }
                }            
            }
            sum_cost+=dist[next_terminal];
        }
        return {sum_cost,edges_used};
    }

    pair<T,vector<int>> minimum_path_heuristic(vector<int> terminals,int loops){
        chmin(loops,terminals.size());
        
        T min_cost=numeric_limits<T>::max();
        vector<int> min_edges;

        for(int i=0;i<loops;i++){
            swap(terminals[i],terminals[0]);
            T temp_cost;
            vector<int> temp_edges;
            tie(temp_cost,temp_edges)=minimum_path_heuristic(terminals);

            if(chmin(min_cost,temp_cost)){
                min_edges=temp_edges;
            }
        }
        return {min_cost,min_edges};
    }

};


void solve(){
    input();
    ll cost_tree;
    
    vector<ll> power(num_cast,1);
    ll cost_power=num_cast;
    vector<int> home_cast(num_home);
    vector<vector<int>> cast_home(num_cast);
 
    for(int h=0;h<num_home;h++){
        ll minimum_cost=INF;
        int id_cast=-1;
        ll need_power;
        tie(id_cast,need_power)=greedy_assign(h,power,-1);
        cost_power+=max(need_power*need_power-power[id_cast]*power[id_cast],0LL);
        chmax(power[id_cast],need_power);
        home_cast[h]=id_cast;
        cast_home[id_cast].push_back(h);
    }
 
    SimulatedAnnealing SA(OP::startTemp,0,TIME.span(),OP::TIME_END,OP::yama,true);
    SteinerTree<ll> ST(num_cast);
    for(auto e:edges_input){
        ST.add_edge(e.from,e.to,e.cost,e.id);
    }
    vector<bool> is_use(num_cast,true);
    cost_tree=ST.kruskal(is_use).first;
    int cnt=0;
    while(TIME.span()<OP::TIME_END){
        cnt++;
        //if(cnt==10) break;
        int id_cast=Rand32(num_cast);
        int pre_cost=power[id_cast];
        if(cast_home[id_cast].size()==0){
            if(id_cast==0) continue;
            power[id_cast]=1-power[id_cast];
            //cerr<<power[id_cast]<<endl;
        }
 
        sort(cast_home[id_cast].begin(),cast_home[id_cast].end(),
            [&](int a,int b){
                return place_cast[id_cast].dist(place_home[a]) < place_cast[id_cast].dist(place_home[b]); 
            });
        
        int start_id=0;
        vector<pil> ps;
        vector<pii> hcs;
        ll change_cost=0;
        bool ok=true;


        if(cast_home[id_cast].size()){
            start_id=Rand32(cast_home[id_cast].size());
            if(start_id==0){
                change_cost=diff_square(power[id_cast],0);
                if(id_cast==0) power[id_cast]=1;
                else power[id_cast]=Rand32(2);
            }else{
                change_cost=diff_square(power[id_cast],place_cast[id_cast].dist(place_home[cast_home[id_cast][start_id-1]]));
            }
    
            for(int h=start_id;h<cast_home[id_cast].size();h++){
                int c;
                ll need_power;
                tie(c,need_power)=greedy_assign(cast_home[id_cast][h],power,id_cast);
                hcs.emplace_back(cast_home[id_cast][h],c);
                if(c==-1){
                    ok=false;
                    break;
                }
                
                if(power[c]<need_power){
                    ps.emplace_back(c,power[c]);
                    change_cost+=diff_square(power[c],need_power);
                    power[c]=need_power;
                    //cerr<<c<<" "<<need_power<<endl;
                }
            }
        }

        for(int i=0;i<num_cast;i++){
            is_use[i]=(power[i]>0);
        }
        ll temp_tree=ST.kruskal(is_use).first;
        if(temp_tree==-1) ok=false;
        ll change_tree=temp_tree-cost_tree;
 
        //cerr<<change_cost<<endl;
        if(SA(change_cost+change_tree) and ok /*and testCounter.cnt["success"]<3*/){
            testCounter.count("success");
            cost_power+=change_cost;
            cost_tree+=change_tree;
            //cerr<<cost_tree<<endl;
            //powerはid_cast以外は変更済み
            if(start_id) power[id_cast]=place_cast[id_cast].dist(place_home[cast_home[id_cast][start_id-1]]);
            //cast_homeとhome_cast
            for(auto hc:hcs){
                cast_home[hc.second].push_back(hc.first);
                home_cast[hc.first]=hc.second;
            }
            cast_home[id_cast].resize(start_id);
        }else{
            testCounter.count("false");
            //powerを元に戻す
            reverse(ps.begin(),ps.end());
            for(auto p:ps){
                power[p.first]=p.second;
            }
            power[id_cast]=pre_cost;
        }
    }

    for(int i=0;i<num_cast;i++){
        is_use[i]=(power[i]>0);
    }
    auto edge_used=ST.kruskal(is_use).second;

    vector<int> id_edge_used(num_edge);
    for(auto e_id:edge_used){
        id_edge_used[e_id]=1;
    }
    output(id_edge_used,power);
#ifndef ONLINE_JUDGE
    testTimer.output();
    testCounter.output();
    cerr<<TIME.span()<<"ms"<<endl;
    cerr<<"cost_tree: "<<cost_tree<<endl;
    cerr<<"cost_power: "<<cost_power<<endl;
    cerr<<"sum: "<<cost_tree+cost_power<<endl;
    cerr<<"score: "<<ll( round(1e6*(1+1e8/(cost_tree+cost_power+1e7)))   )<<endl;
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
