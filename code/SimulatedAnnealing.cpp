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
TestCounter testCounter;

///++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
///++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
///++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

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
    SimulatedAnnealing(float temp_start,float temp_end,float time_start,float time_end,bool is_hill,bool minimum,int temp_type=0,int interval=1):
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


void solve(){
    
#ifndef ONLINE_JUDGE
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
    int T=1;
    //cin>>T;
    while(T--) solve();
}
