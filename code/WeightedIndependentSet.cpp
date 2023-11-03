/*
重みがすべて同じ場合、O(1.466^n n)
__builtin_ctzllを使っているから、頂点が65以上の場合は無理
頂点数が32以下: C=u32 みたいな感じ
参考: https://judge.yosupo.jp/submission/5455
verify: https://judge.yosupo.jp/submission/169899
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

template<class T,class C>
struct WeightedIndependentSet{
    int N_;
    vector<T> weight_; //頂点の重み
    vector<C> graph_;

    //dfs中に更新されるもの
    C best_solution_=0;
    T best_value_=0;

    WeightedIndependentSet(int N,vector<T> weight):N_(N),graph_(N),weight_(weight){

    }

    void add_edge(int u,int v){
        graph_[u]|=(C(1)<<v);
        graph_[v]|=(C(1)<<u);
    }

	inline bool can_add(int v,C now_solution){
		return (now_solution&graph_[v])==0;
	}
	inline C bit_set(C x,int idx){
		return x|=C(1)<<idx;
	}
	inline C bit_flip(C x,int idx){
		return x^=C(1)<<idx;
	}
	//vが、undecidedの頂点との辺が存在しないならtrue
	inline bool can_always_use(int v,C vertices_undecided){
		C adjacent=(vertices_undecided&graph_[v]);
		if(adjacent==0) return true;
		else if(popcount(adjacent)>1){
			return false;
		}
		int v_adj=__builtin_ctzll(adjacent);
		return weight_[v]>=weight_[v_adj];
	}
	inline C update_undecided(C vertices_undecided,int v_added){
		//v_addedを加えることができた、ということはvertices_undecidedにあったと考える
		vertices_undecided=bit_flip(vertices_undecided,v_added);
		return vertices_undecided&~graph_[v_added];
	}

    void dfs(C now_solution,T now_value,C vertices_undecided){
		//必ず使っていいものを見つけた場合、使う
		bool updated=false;
		do{
			updated=false;
			for(C S=vertices_undecided;S;S&=S-1){
				int v=__builtin_ctzll(S); //Sを下から見る
				if(can_always_use(v,vertices_undecided) and weight_[v]>0){
					//vを使う
					now_solution=bit_set(now_solution,v);
					vertices_undecided=update_undecided(vertices_undecided,v);
					now_value+=weight_[v];
					updated=true;
					break;
				}
		}
		}while(updated);

		//解が改善されたなら更新
		if(now_value>best_value_){
			best_value_=now_value;
			best_solution_=now_solution;
		}

		//上界を計算
		//ついでに選ぶ頂点を決める
		//重みなし: 次数が低い頂点
		//重みあり: 一番重い頂点　がよさそう？
		T value_upper=now_value;
		int v_select=-1;
		int min_degree=MAX;
		for(C S=vertices_undecided;S;S&=S-1){
			int v=__builtin_ctzll(S); //Sを下から見る
			if(can_add(v,now_solution) and weight_[v]>0){
				value_upper+=weight_[v];
			}
			if(chmin(min_degree,popcount(graph_[v]))){
				v_select=v;
			}
		}
		if(value_upper<=best_value_){
			return;
		}

		//頂点vを使う
		if(weight_[v_select]>0) dfs(bit_set(now_solution,v_select),now_value+weight_[v_select],update_undecided(vertices_undecided,v_select));
		//頂点vを使わない
		dfs(now_solution,now_value,bit_flip(vertices_undecided,v_select));
    }

    vector<int> solve(){
		dfs(0,0,mask(N_));
		vector<int> ans;
		for(int i=0;i<N_;i++){
			if(bitUP(best_solution_,i)){
				ans.push_back(i);
			}
		}
		return ans;
    }
};

void solve(){
	int N,M;
	cin>>N>>M;
	vector<int> weight(N,1);
	WeightedIndependentSet<int,ll> S(N,weight);

	for(int i=0;i<M;i++){
		int u,v;
		cin>>u>>v;
		S.add_edge(u,v);
	}
	auto ans=S.solve();
	cout<<ans.size()<<endl;
	cout<<ans<<endl;
}
 
int main(){
    FastIO();
    solve();
}

