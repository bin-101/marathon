/*
参考: https://topcoder-tomerun.hatenablog.jp/entry/2022/11/06/145156
初期化がO(1)のbool配列
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

struct FastBoolVector{
    int clear_count_;
    vector<int> array_;

    FastBoolVector(int size):clear_count_(0),array_(size,-1){

    }
    void reset(int idx){
        array_[idx]=-1;
    }
    void set(int idx){
        array_[idx]=clear_count_;
    }
    bool test(int idx){
        return array_[idx]==clear_count_;
    }

    void clear(){
        clear_count_++;
    }
    bool operator[](int idx){
        return test(idx);
    }
};
int main(){
    FastBoolVector S(100);
    cout<<S[0]<<endl; //0
    S.set(0);
    cout<<S[0]<<endl; //1
    S.reset(0);
    cout<<S[0]<<endl; //0
    S.set(0);
    S.clear();
    cout<<S[0]<<endl; //0
    S.set(0);
    cout<<S[0]<<endl; //1
}
