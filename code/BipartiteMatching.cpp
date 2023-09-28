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


template<typename T >
struct edge {
	int to;
	T cost;
	edge()=default;
	edge(int to, T cost) : to(to), cost(cost){}
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

/*
//とりあえず、簡単化のために0~N/2-1を左と仮定
参考: https://arxiv.org/abs/2208.11325
verify: 
TODO:
辺の重さの変更・削除に対応
辺を追加・変更をためといて効率よく差分更新をさせたい(頂点の順番は貪欲でよさそう)
参考の3.2節を読む
右の頂点周りの変更も1回のダイクストラで処理可能にする
solve(i)でiがマッチしている場合とiがマッチしていない場合で処理を変える
*/
template<typename T>
struct BipartiteMatching{
    int N_;
    WeightGraph<T> graph_;
    vector<T> dist_;
    vector<int> pre_;
    vector<T> potential_;
    vector<int> matching_;
    vector<T> cost_rev_;
    const T inf=numeric_limits< T >::max();

    BipartiteMatching(int N):N_(N),graph_(N),dist_(N),pre_(N_,-1),potential_(N,0),matching_(N,-1),
        cost_rev_(N,0){

    }

    void add_edge(int u,int v,T weight){
        graph_[u].emplace_back(v,weight);
    }

    T f(int from,edge<T> e){
        //cerr<<e.cost+potential_[from]-potential_[e.to]<<endl;
        assert(e.cost+potential_[from]-potential_[e.to]>=0);
        return e.cost+potential_[from]-potential_[e.to];
    }
    T f(int from,int to,T cost){
        assert(cost+potential_[from]-potential_[to]>=0);
        return cost+potential_[from]-potential_[to];
    }

    T solve(int s){
        potential_[s]=inf;
        //potentialの更新
        for(auto e:graph_[s]){
            chmin(potential_[s],e.cost-potential_[e.to]);
        }
        potential_[s]*=-1;

        //distの初期化
        for(int i=0;i<N_;i++){
            dist_[i]=inf;
        }
        //shortest pathの計算
        minimum_queue<pair<T,int>> que;
        que.emplace(0,s);
        dist_[s]=0;
        while(que.size()){
            int v;
            T cost;
            tie(cost,v)=que.top(); que.pop();

            if(cost>dist_[v]) continue;

            for(auto e:graph_[v]){
                if(e.to==matching_[v]) continue;
                if(chmin(dist_[e.to],cost+f(v,e))){
                    pre_[e.to]=v;
                    if(matching_[e.to]==-1) continue;
                    int next_left=matching_[e.to];
                    if(chmin(dist_[next_left],dist_[e.to]+f(e.to,next_left,cost_rev_[e.to]))){
                        //assert(dist_[next_left]>=dist_[v]);
                        pre_[next_left]=e.to;
                        que.emplace(dist_[next_left],next_left);
                    }
                }
            }
        }
        for(int i=0;i<N_;i++){
            if(dist_[i]==inf) continue;
            dist_[i]+=potential_[i]-potential_[s];
        }

        int t=-1;
        T min_dist=inf;
        for(int i=N_/2;i<N_;i++){
            if(matching_[i]!=-1) continue;
            if(chmin(min_dist,dist_[i])){
                t=i;
            }
        }
        //cerr<<t<<endl;
        //shortest pathを復元しながらmatching_とcost_rev_を更新
        while(true){
            //右の頂点
            int left=pre_[t];
            matching_[t]=pre_[t];
            cost_rev_[t]=(dist_[t]-dist_[left])*-1;
            
            //左の頂点
            matching_[left]=t;
            t=pre_[left];
            if(left==s) break;
        }


        //potentialの更新　左だけでいい
        for(int i=0;i<N_;i++){
            if(dist_[i]==inf) continue;
            potential_[i]=dist_[i];
        }

        return min_dist;
    }
};

void solve(){
    int N;
    cin>>N;
    auto W=vmake(N,N,0);
    auto E=vmake(N,N,0);
    auto C=vmake(N,N,'a');

    cin>>W>>E>>C;

    BipartiteMatching<ll> matching(2*N);

    ll res=0;

    for(int h=0;h<N;h++){
        for(int w=0;w<N;w++){
            int sum=0;
            for(int k=0;k<N;k++){
                if(k==w and C[h][k]=='.') sum+=W[h][k];
                if(k!=w and C[h][k]=='o') sum+=E[h][k];
            }
            matching.add_edge(h,w+N,sum);
        }
        res+=matching.solve(h);
    }
    vector<pair<pii,string>> ans;
    for(int h=0;h<N;h++) for(int w=0;w<N;w++){
        if(C[h][w]=='.' and matching.matching_[h]==w+N) ans.emplace_back(pii(h+1,w+1),"write");
        if(C[h][w]=='o' and matching.matching_[h]!=w+N) ans.emplace_back(pii(h+1,w+1),"erase");
    }

    cout<<res<<endl;
    cout<<ans.size()<<endl;
    for(auto a:ans){
        cout<<a<<endl;
    }
}
 
int main(){
    FastIO();
    solve();
}

