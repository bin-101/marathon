/*
参考: https://cyberagent.ai/blog/research/1036/
多点スタートをいい感じに枝刈りするやつ
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

    int stage_now_;
    int cnt_call_=0;

    SuccessiveHalving(){

    }

    void add_state(S state,T score){
        states_.push_back(state);
        scores_.push_back(score);
    }
    void build(double time_end){
        assert(states_.size()>=1);
        time_start_=TIME.span();
        time_end_=time_end;

        //ステージの数を求める
        int num_stage=1;
        int x=1;
        while(x>=states_.size()){
            x*=2;
            num_stage++;
        }

        time_each_stage_=(time_end_-time_start_)/num_stage;
    }
    //次に近傍を計算する解を返す
    //すでに終了している場合、評価値-1を返す
    inline pair<S,T> &next_state(){
        double now_time_=TIME.span();
        if(now_time_>=time_end_){
            return {states_,-1};
        }
        if(time_start_+time_each_stage_*(stage_now_+1)>=now_time_){
            //次のステージに移行
            vector<int> order;
            for(int i=0;i<states_.size();i++) order.push_back(i);
            sort(order.begin(),order.end(),
                [&](int i,int j){
                    return scores_[i]>scores_[j];
                });

            //エラー出力
            cerr<<"stage: "<<stage_now_<<endl;
            for(int i=0;i<order.size();i++){
                cerr<<" "<<i<<": "<<scores_[order[i]]<<endl;
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


int main(){

}
