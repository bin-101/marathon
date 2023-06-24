/*
verify:https://judge.yosupo.jp/submission/146422
ワーシャルフロイドはまだ経路復元できない
*/
#include <bits/stdc++.h>
using namespace std;

template<class T,class U>constexpr bool chmin(T&a,const U b){if(a<=b)return false;a=b;return true;}
template<class T,class U>constexpr bool chmax(T&a,const U b){if(a>=b)return false;a=b;return true;}

typedef long long ll;
constexpr int MAX=1<<30;
constexpr ll INF=1LL<<62;
constexpr int MOD=1e9+7;



template<typename T >
struct edge {
    int to;
    T cost;
    edge()=default;
    edge(int to, T cost) : to(to), cost(cost) {}
};

//有向グラフを想定
template<typename T>
struct Dijkstra{
    struct edge {
        int to;
        T cost;
        int id;
        edge()=default;
        edge(int to, T cost,int id) : to(to), cost(cost), id(id) {}
    };
    using pti=pair<T,int>;
    using pii=pair<int,int>;

    int num_vertex_;
    int num_edge_;
    vector<vector<edge>> graph_;
    vector<vector<T>> dist_;
    //頂点,辺id
    vector<vector<pii>> pre_id_;
    bool is_directed;
 
    Dijkstra(int num_vertex,bool is_directed):num_vertex_(num_vertex),graph_(num_vertex),
        dist_(num_vertex),pre_id_(num_vertex),is_directed(is_directed){
        
    }
 
    void add_edge(int u,int v,T cost,int id=-1){
        graph_[u].emplace_back(v,cost,id);
        if(is_directed==false) graph_[v].emplace_back(u,cost,id);
    }

    //始点
    void dijkstra(int vertex_start){
        vector<T> dist(num_vertex_,numeric_limits<T>::max());
        vector<pii> pre_id(num_vertex_,pii(-1,-1));
        priority_queue<pti,vector<pti>,greater<pti>> que;

        dist[vertex_start]=0;
        que.emplace(dist[vertex_start],vertex_start);

        while(que.size()){
            int now_vertex;
            T now_cost;
            tie(now_cost,now_vertex)=que.top(); que.pop();

            if(now_cost>dist[now_vertex]) continue;

            for(auto e:graph_[now_vertex]){
                if(chmin(dist[e.to],dist[now_vertex]+e.cost)){
                    que.emplace(dist[e.to],e.to);
                    pre_id[e.to]={now_vertex,e.id};
                }
            }
        }

        dist_[vertex_start]=dist;
        pre_id_[vertex_start]=pre_id;
    }

    //https://37zigen.com/floyd-warshall-algorithm/
    //経路復元よく分からない
    void warshall_floyd(){
        for(int i=0;i<num_vertex_;i++){
            dist_[i].assign(num_vertex_,numeric_limits<T>::max()/4);
            dist_[i][i]=0;
            pre_id_.resize(num_vertex_);
        }
        for(int k=0;k<num_vertex_;k++){
            for(int i=0;i<num_vertex_;i++){
                for(int j=0;j<num_vertex_;j++){
                    chmin(dist_[i][j],dist_[i][k]+dist_[k][j]);    
                    
                }
            }
        }
    }

    T query_cost(int s,int t){
        return dist_[s][t];
    }
    ////頂点,辺id sからtのパス 最後は(t,-1)
    //pathが存在しない場合、空を返す
    pair<T,vector<pii>> query_path(int s,int t){
        vector<pii> path;
        int vertex_now=t;

        if(pre_id_[s][t].first==-1) return {-1,vector<pii>()};

        path.emplace_back(t,-1);

        while(vertex_now!=s){
            path.emplace_back(pre_id_[s][vertex_now]);
            vertex_now=pre_id_[s][vertex_now].first;
        }

        reverse(path.begin(),path.end());
        return {dist_[s][t],path};
    }

};

int main(){
    int N,M,s,t;
    cin>>N>>M>>s>>t;

    Dijkstra<ll> D(N,true);

    while(M--){
        int a,b,c;
        cin>>a>>b>>c;
        D.add_edge(a,b,c);
    }
    D.dijkstra(s);

    vector<pair<int,int>> path;
    ll cost;
    tie(cost,path)=D.query_path(s,t);
    if(path.size()==0){
        cout<<-1<<endl;
        return 0;
    }
    vector<pair<int,int>> ans;
    for(int i=0;i<path.size()-1;i++){
        ans.emplace_back(path[i].first,path[i+1].first);
    }
    cout<<cost<<" "<<ans.size()<<"\n";
    for(auto x:ans){
        cout<<x.first<<" "<<x.second<<"\n";
    }
}
