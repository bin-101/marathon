/*
Kernighan–Lin algorithm: https://en.wikipedia.org/wiki/Kernighan%E2%80%93Lin_algorithm
集合ごとに個数が固定されている状況を仮定
	そうでないならば、swapのところにmoveを追加するとよさそう
辺コストはすべて正を仮定
あまり強くない
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

using Graph=vector<vector<int>>;

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
template<class T>
struct GraphPartitioning{
    int N_;
    WeightGraph<T> graph_; //頂点の重み
	vector<vector<T>> cost_matrix_;

    GraphPartitioning(int N):N_(N),graph_(N){
		cost_matrix_=vmake(N_,N_,T(0));
    }

    void add_edge(int u,int v,T cost){
        graph_[u].emplace_back(v,cost);
        graph_[v].emplace_back(u,cost);
		cost_matrix_[u][v]+=cost;
		cost_matrix_[v][u]+=cost;
    }
	T naive_caliculate(const vector<int> &partition){
		T sum=0;
		for(int i=0;i<N_;i++){
			for(int j=i+1;j<N_;j++){
				if(partition[i]==partition[j]) continue;
				sum+=cost_matrix_[i][j];
			}
		}
		return sum;
	}

	pair<T,vector<int>> solve(int max_depth){
		//初期解を作る
		vector<int> partition(N_);
		for(int i=0;i<N_;i++){
			partition[i]=i%2;
		}

		T now_value=naive_caliculate(partition);

		int cnt_loop=0;
		while(true){
			cnt_loop++;
			vector<T> D(N_);
			//Dを計算
			for(int v=0;v<N_;v++){
				for(auto e:graph_[v]){
					if(partition[e.to]==partition[v]){
						D[v]-=e.cost;
					}else{
						D[v]+=e.cost;
					}
				}
			}
			vector<T> diff_history;
			vector<array<int,2>> pair_history;
			array<vector<int>,2> order_D;
			for(int i=0;i<N_;i++){
				order_D[partition[i]].push_back(i);
			}

			for(int depth=0;depth<max_depth;depth++){
				//Dをソート
				for(int i=0;i<2;i++){
					sort(order_D[i].begin(),order_D[i].end(),
						[&](int x,int y){
							return D[x]>D[y];
						});
				}
				//max_pairを求める
				array<int,2> max_pair;
				T max_diff=numeric_limits<T>::min();
				for(int i=0;i<order_D[0].size();i++){
					int a=order_D[0][i];
					for(int j=0;j<order_D[1].size();j++){
						int b=order_D[1][j];
						if(D[a]+D[b]<=max_diff) break;
						if(chmax(max_diff,D[a]+D[b]-2*cost_matrix_[a][b])){
							max_pair[0]=a;
							max_pair[1]=b;
						}
					}
				}
				diff_history.push_back(max_diff);
				pair_history.push_back(max_pair);
				//order_Dからmax_pairを削除
				for(int i=0;i<2;i++){
					order_D[i].erase(find(order_D[i].begin(),order_D[i].end(),max_pair[i]));
				}
				//Dを更新
				for(int p=0;p<2;p++){
					for(int v:order_D[p]){
						for(int i=0;i<2;i++){
							if(i==p){
								D[v]+=2*cost_matrix_[v][max_pair[i]];
							}else{
								D[v]-=2*cost_matrix_[v][max_pair[i]];
							}
						}
					}
				}

			}
			int max_id=-1;
			T max_diff=0;
			T sum_diff=0;
			for(int i=0;i<max_depth;i++){
				sum_diff+=diff_history[i];
				if(chmax(max_diff,sum_diff)){
					max_id=i;
				}
			}
			if(max_id==-1) break;
			now_value-=max_diff;
			for(int i=0;i<=max_id;i++){
				swap(partition[pair_history[i][0]],partition[pair_history[i][1]]);
			}
			assert(now_value==naive_caliculate(partition));
		}
		cerr<<cnt_loop<<endl;
		return {now_value,partition};
	}

};

void solve(){
	Xor32 Rand32;

	int N=1000;
	GraphPartitioning<ll> solver(N);
	for(int i=0;i<N;i++){
		for(int j=i+1;j<N;j++){
			ll cost=Rand32(10,100000);
			solver.add_edge(i,j,cost);
		}
	}
	for(int i=1;i<=N/2;i++){
		cerr<<i<<": "<<solver.solve(i).first<<endl;
	}

}
 
int main(){
    FastIO();
    solve();
}

