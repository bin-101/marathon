/*
参考: https://qiita.com/hamko/items/cceb1a92da14e2755527
verify: https://atcoder.jp/contests/abc032/submissions/47185043
todo:
https://zenn.dev/kounoike/articles/20210327-hard-knapsack も読む
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

template<class T,class W,class D>
struct Knapsack{
	struct Item{
		T value;
		W weight;
		int idx;
		D par_value;
	};
	vector<Item> items_;

    Knapsack(){

    }

	void add_item(T value,W weight,int idx=-1){
		items_.push_back({value,weight,idx,D(value)/weight});
	}

	int N_;
	vector<W> weight_sum_;
	vector<T> value_sum_;

	T best_value_=0;
	vector<bool> best_solution_;

	vector<bool> now_solution_;
	W now_capacity_=0;
	T now_value_=0;

	//[a,b]の合計
	template<class X>
	X caliculate_sum(vector<X> sum,int a,int b){
		if(a>b) return 0;
		if(a==0) return sum[b];
		return sum[b]-sum[a-1];
	}

	void add_solution(int idx){
		now_solution_[idx]=true;
		now_capacity_-=items_[idx].weight;
		now_value_+=items_[idx].value;
	}
	void erase_solution(int idx){
		now_solution_[idx]=false;
		now_capacity_+=items_[idx].weight;
		now_value_-=items_[idx].value;
	}

	void dfs(int idx){
		//使えない場合
		if(idx<N_ and now_capacity_<items_[idx].weight){
			dfs(idx+1);
			return;
		}

		//緩和問題の最適解を計算
		//O(log(N-idx))
		int ok=idx-1,ng=N_;
		while(ng-ok!=1){
			int mid=(ok+ng)/2;
			if(caliculate_sum(weight_sum_,idx,mid)<=now_capacity_){
				ok=mid;
			}else{
				ng=mid;
			}
		}
		if(ok==N_-1){
			//すべて解に含めることができる
			for(int i=idx;i<N_;i++){
				add_solution(i);
			}
			if(chmax(best_value_,now_value_)){
				best_solution_=now_solution_;
			}
			for(int i=idx;i<N_;i++){
				erase_solution(i);
			}
			return;
		}
		T upper_value_=now_value_+caliculate_sum(value_sum_,idx,ok);
		upper_value_+=(now_capacity_-caliculate_sum(weight_sum_,idx,ok))*items_[ok+1].par_value;

		if(upper_value_<=best_value_){
			return;
		}

		//解に入れる
		add_solution(idx);
		dfs(idx+1);
		erase_solution(idx);
		dfs(idx+1);
	}

	T solve(W capacity){
		now_capacity_=capacity;
		sort(items_.begin(),items_.end(),
			[](Item &a,Item &b){
				return a.par_value>b.par_value;
			});
		N_=items_.size();
		now_solution_.resize(N_);
		weight_sum_.resize(N_);
		value_sum_.resize(N_);
		for(int i=0;i<N_;i++){
			if(i){
				weight_sum_[i]=weight_sum_[i-1];
				value_sum_[i]=value_sum_[i-1];
			}
			weight_sum_[i]+=items_[i].weight;
			value_sum_[i]+=items_[i].value;
		}
		dfs(0);
		return best_value_;
	}
};

void solve(){
	int N,W;
	cin>>N>>W;
	Knapsack<ll,ll,double> K;
	for(int i=0;i<N;i++){
		ll v,w;
		cin>>v>>w;
		K.add_item(v,w,-1);
	}
	cout<<K.solve(W)<<endl;
}
 
int main(){
    FastIO();
    solve();
}

