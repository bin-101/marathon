/*
参考: https://cyberagent.ai/blog/research/1036/
多点スタートをいい感じに枝刈りするやつ
まだverifyしていない
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
Timer TIME;

//状態の型、評価値の型
template<class S,class T>
struct SuccessiveHalving{
    vector<S> states_;
    vector<T> scores_;
    double time_start_;
    double time_end_;
    double time_each_stage_;
    int interval_;

    int stage_now_=0;
    double time_now_;
    int cnt_call_=0;

    SuccessiveHalving(int interval=1):interval_(interval){

    }

    void add_state(S state,T score){
        states_.push_back(state);
        scores_.push_back(score);
    }
    void build(double time_end){
        assert(states_.size()>=1);
        time_start_=TIME.span();
        time_now_=time_start_;
        time_end_=time_end;

        //ステージの数を求める
        int num_stage=1;
        int x=1;
        while(x<states_.size()){
            x*=2;
            num_stage++;
        }

        time_each_stage_=(time_end_-time_start_)/num_stage;
    }
    //次に近傍を計算する解を返す
    //すでに終了している場合、評価値-1を返す
    inline pair<S&,T&> next_state(){
        if(cnt_call_%interval_==0){
            time_now_=TIME.span();
        }
        if(time_now_>=time_end_){
            T x=-INF; //ありえない値
            cerr<<x<<endl;
            return {states_[0],x};
        }
        if(time_now_>=time_start_+time_each_stage_*(stage_now_+1)){
            //次のステージに移行
            vector<int> order;
            for(int i=0;i<states_.size();i++) order.push_back(i);
            sort(order.begin(),order.end(),
                [&](int i,int j){
                    return scores_[i]<scores_[j]; //最小化
                });

            //エラー出力
            cerr<<"stage: "<<stage_now_<<endl;
            for(int i=0;i<1;i++){
                cerr<<" "<<i<<": "<<states_[order[i]]<<endl;
                cerr<<scores_[order[i]]<<endl;
            }

            vector<S> next_states;
            vector<T> next_scores;
            for(int i=0;i<(states_.size()+1)/2;i++){
                int id=order[i];
                next_states.push_back(states_[id]);
                next_scores.push_back(scores_[id]);
            }
            states_=next_states;
            scores_=next_scores;
            stage_now_++;
        }
        int idx=cnt_call_%states_.size();
        cnt_call_++;
        return {states_[idx],scores_[idx]};
    }
};
double f(const vector<double> &v){
    /*double sum2=0;
    double sum_cos=0;

    for(double x:v){
        sum2+=x*x;
        sum_cos+=cos(2*M_PI*x);
    }
    double n=v.size();

    return 20-20*exp(-0.2*sqrt(sum2/n))+M_E-exp(sum_cos/n);*/

    //Schwefel function
    double ret=0;
    for(auto x:v){
        ret-=x*sin(sqrt(abs(x)));
    }
    return ret;
}

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

int main(){
    SuccessiveHalving<vector<double>,double> SH;
    for(int i=0;i<100000;i++){
        vector<double> v;
        for(int j=0;j<10;j++){
            double x=rand()%400;
            if(rand()%2) x*=-1;
            v.push_back(x);
        }
        SH.add_state(v,f(v));
    }

    SH.build(1000);

    while(true){
        auto [v,score]=SH.next_state();
        if(score<-INF/10) break;
        int id=rand()%10;
        double diff=1;
        if(rand()%2) diff*=-1;
        v[id]+=diff;
        if(abs(v[id])>500) continue;

        double next_score=f(v);
        if(f(v)<score){
            score=next_score;
        }else{
            v[id]-=diff;
        }
    }
    cerr<<SH.cnt_call_<<endl;
}