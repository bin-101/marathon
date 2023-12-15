/*
有向グラフの閉路検出
verify: https://judge.yosupo.jp/submission/173487
*/
#include <bits/stdc++.h>
using namespace std;

template<class T,class U>constexpr bool chmin(T&a,const U b){if(a<=b)return false;a=b;return true;}
template<class T,class U>constexpr bool chmax(T&a,const U b){if(a>=b)return false;a=b;return true;}

typedef long long ll;
constexpr int MAX=1<<30;
constexpr ll INF=1LL<<62;
constexpr int MOD=1e9+7;

struct Cycle{
    struct edge{
        int from;
        int to;
        int id;
    };
    int num_vertex_;
    vector<vector<edge>> graph_;
    Cycle(int n):num_vertex_(n),graph_(n){

    }
    void add_edge(int from,int to,int id){
        graph_[from].push_back({from,to,id});
    }

    //https://judge.yosupo.jp/submission/109411
    vector<edge> find_cycle(){
        
        vector<edge> cycle;

        vector<bool> visited(num_vertex_,false);
        vector<edge> stk(num_vertex_);
        using iterator=vector<edge>::iterator;
        iterator ptr_stk=stk.begin();
        vector<iterator> idx(num_vertex_,stk.end());
        
        auto dfs=[&](auto dfs,int now)->bool{
            idx[now]=ptr_stk;
            for(const auto &e:graph_[now]){
                int nxt=e.to;
                if(not visited[nxt]){
                    visited[nxt]=true;
                    *ptr_stk++=e;
                    if(dfs(dfs,nxt)){
                        return true;
                    }
                    ptr_stk--;
                }else if(idx[nxt]!=stk.end()){ //訪問後かつまだ探索が終わっていない
                    *ptr_stk++=e;
                    cycle.resize(ptr_stk-idx[nxt]);
                    move(idx[nxt],ptr_stk,cycle.begin());
                    return true;
                }
            }
            idx[now]=stk.end();
            return false;
        };

        for(int i=0;i<num_vertex_;i++){
            if(visited[i]) continue;
            if(dfs(dfs,i)) return cycle;
        }
        return {};
    }
};

int main(){
    int N,M;
    cin>>N>>M;
    Cycle cycle(N);
    for(int i=0;i<M;i++){
        int from,to;
        cin>>from>>to;
        cycle.add_edge(from,to,i);
    }
    auto ans=cycle.find_cycle();
    if(ans.size()==0) cout<<-1<<endl;
    else{
        cout<<ans.size()<<endl;
        for(auto e:ans){
            cout<<e.id<<endl;
        }
    }
}