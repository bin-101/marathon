/*
参考: https://topcoder-tomerun.hatenablog.jp/entry/2022/11/06/145156
初期化がO(1)の配列
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

template<class T>
struct FastInitVector{
    T initial_value_;
    int clear_count_;
    vector<T> array_;
    vector<int> count_;

    FastInitVector(int size,int initial_value):initial_value_(initial_value),clear_count_(0),
        array_(size,initial_value),count_(size,0){

    }
    void reset(int idx){
        set(idx,initial_value_);
    }
    void set(int idx,T value){
        array_[idx]=value;
        count_[idx]=clear_count_;
    }
    T value(int idx){
        if(count_[idx]==clear_count_) return array_[idx];
        return initial_value_;
    }
    void clear(){
        clear_count_++;
    }
    T operator[](int idx){
        return value(idx);
    }
};
int main(){
    FastInitVector<int> S(100,-1);
    cout<<S[0]<<endl; //-1
    S.set(0,500);
    cout<<S[0]<<endl; //500
    S.reset(0);
    cout<<S[0]<<endl; //-1
    S.set(0,900);
    S.clear();
    cout<<S[0]<<endl; //-1
    S.set(0,100);
    S.set(0,200);
    cout<<S[0]<<endl; //200
}