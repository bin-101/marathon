/*
verify: https://atcoder.jp/contests/rco-contest-2018-qual/submissions/45728639
*/
#define NDEBUG

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


struct Xor32{
    using u32=uint32_t;
 
    u32 x=1234567;
    inline u32 rnd_make(){
        x^=(x<<13);
        x=x^(x>>17);
        return x=x^(x<<5);
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
    //[0,b)の中から異なる2つを選ぶ first<second
    inline pair<int,int> two(int b){
        assert(b>=2);
        int v1=rnd_make()%b;
        int v2=rnd_make()%(b-1);
        if (v2>=v1) return {v1,v2+1};
        else return {v2,v1};
    }
};

struct Xor64{
    u64 x=1234567;
    inline u64 rnd_make(){
    	x ^= x << 13;
    	x ^= x >> 7;
    	x ^= x << 17;
    	return x;
    }
    inline u64 operator()(){
        return rnd_make();
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
struct TestTimer{
    chrono::high_resolution_clock::time_point st;
    unordered_map<string,ll> sum_time;
    unordered_map<string,chrono::high_resolution_clock::time_point> start_time;

    TestTimer(){}
    void start(const string &s){
#ifndef ONLINE_JUDGE
        start_time[s]=chrono::high_resolution_clock::now();
#endif
    }
    void end(const string &s){
#ifndef ONLINE_JUDGE
        auto now=chrono::high_resolution_clock::now();
        sum_time[s]+=chrono::duration_cast<chrono::nanoseconds>(now-start_time[s]).count();
#endif
    }
    void output()const{
#ifndef ONLINE_JUDGE
        for(auto m:sum_time){
            cerr<<m.first<<": "<<m.second/1e6<<"ms"<<endl;
        }
#endif
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
Xor64 Rand64;
TestTimer testTimer;
TestCounter testCounter;

//https://atcoder.jp/contests/asprocon9/submissions/34659956
template<class T,int CAPACITY>
class DynamicArray{
public:
    array<T,CAPACITY> array_={};
    int size_=0;

    DynamicArray(){}

    DynamicArray(int n){
        resize(n);
    }

    void push_back(const T &e){
        assert(size_<CAPACITY);
        array_[size_++]=e;
    }
    void pop_back(){
        size_--;
    }

    inline T& operator[](int index){
        return array_[index];
    }
	inline const T& operator[](int index) const {
		return array_[index];
	}
    inline int size()const{
        return size_;
    }
    inline T& back(){
        return array_[size_-1];
    }
	inline auto begin() -> decltype(array_.begin()) {
		return array_.begin();
	}

	inline auto end() -> decltype(array_.begin()) {

		return array_.begin() + size_;
	}

	inline auto begin() const -> decltype(array_.begin()) {
		return array_.begin();
	}

	inline auto end() const -> decltype(array_.begin()) {

		return array_.begin() + size_;
	}
    inline void resize(int new_size){
        size_=new_size;
    }
    void operator=(const DynamicArray &e){
        for(int i=0;i<e.size_;i++){
            array_[i]=e[i];
        }
        size_=e.size_;
    }
    void clear(){
        size_=0;
    }
    //O(1)
    //末尾と入れ替える。順序が保持されないが高速
    void swap_remove(int idx){
        array_[idx]=array_[size_-1];
        size_--;
    }
    //O(size)
    //順序を気にしない場合、swap_removeの方がいい
    void remove(int idx){
        for(int i=idx;i<size_-1;i++){
            array_[i]=array_[i+1];
        }
        size_--;
    }
    void fill(T x){
        for(int i=0;i<size_;i++){
            array_[i]=x;
        }
    }
};

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

//vは昇順
bool is_in(const vector<int> &v,int x){
    int n=v.size();
    if(n==0) return false;
    int ng=-1,ok=n-1;
    while(ok-ng!=1){
        int mid=(ok+ng)/2;
        if(v[mid]<x) ng=mid;
        else ok=mid;
    }
    if(v[ok]==x) return true;
    return false;
}

template<typename T >
struct edge {
	int to;
	T cost;
    int id;
	edge()=default;
	edge(int to, T cost,int id) : to(to), cost(cost), id(id) {}
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
 
//pair cin
template<typename T, typename U>
inline istream &operator>>(istream &is,pair<T,U> &p) {
	is>>p.first>>p.second;
	return is;
}
template<typename T>
void append(vector<T> &v,const vector<T> &vp){
    for(auto p:vp){
        v.push_back(p);
    }
}

//[0,n)の集合を管理
//値の追加・削除・存在確認: O(1)
//空間計算量: O(n)
//重複は許さない
//https://topcoder-tomerun.hatenablog.jp/entry/2021/06/12/134643
template<int CAPACITY>
struct IntSet{
    DynamicArray<int,CAPACITY> set_;
    array<int,CAPACITY> pos_;

    IntSet(){
        for(int i=0;i<CAPACITY;i++){
            pos_[i]=-1;
        }
        set_.clear();
    }
    void insert(int v){
        //assert(pos_[v]==-1);
        if(pos_[v]!=-1) return;
        pos_[v]=set_.size();
        set_.push_back(v);
    }

    void remove(int v){
        assert(pos_[v]!=-1);
        int last=set_[set_.size()-1];
        set_[pos_[v]]=last;
        pos_[last]=pos_[v];
        set_.pop_back();
        pos_[v]=-1;
    }

    bool contains(int v)const{
        return pos_[v]!=-1;
    }

    int size()const{
        return set_.size();
    }

    int random()const{
        assert(set_.size());
        int x=set_[Rand32(set_.size())];
        assert(pos_[x]!=-1);
        return x;
    }

    int random_extract(){
        int v=set_[Rand32(set_.size())];
        remove(v);
        return v;
    }
    void clear(){
        while(size()){
            remove(set_.back());
        }
    }
};

///++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
///++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
///++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

template<class T,class U>
T linear_function(U x,U start_x,U end_x,T start_value,T end_value){
    if(x>=end_x) return end_value;
    if(x<=start_x) return start_value;
    return start_value+(end_value-start_value)*(x-start_x)/(end_x-start_x);
}

//http://gasin.hatenadiary.jp/entry/2019/09/03/162613
struct SimulatedAnnealing{
    float startTemp; //差の最大値(あくまでも参考)
    float endTemp; //差の最小値(あくまでも参考)
    float startTime;
    float endTime;
    bool yama;
    bool minimum;
    //SimulatedAnnealing(){}
    SimulatedAnnealing(float startTemp,float endTemp,float startTime,float endTime,bool yama,bool minimum):
        startTemp(startTemp),endTemp(endTemp),startTime(startTime),endTime(endTime),yama(yama),minimum(minimum){
    }
    float calc_temp(){
        return linear_function(float(TIME.span()),startTime,endTime,startTemp,endTemp); //線形
        //https://atcoder.jp/contests/ahc014/submissions/35326979
        /*float progress=(TIME.span()-startTime)/(endTime-startTime);
        if(progress>1.0) progress=1.0;
        return pow(startTemp,1.0-progress)*pow(endTemp,progress);*/
    }
    float calc_prob(float diff){
        if(minimum) diff*=-1;
        if(diff>0) return 1;
        float temp=calc_temp();
        return exp(diff/temp);
    }
    float calc_diff(float prob){
        float diff=log(prob)*calc_temp();
        if(minimum) diff*=-1;
        return diff;
    }
    inline bool operator()(float diff){
        testCounter.count("try_cnt");
        if(minimum) diff*=-1;
        if(diff>0){
            testCounter.count("plus_change");
            return true;
        }
        if(yama) return false;
        float prob;
        if(minimum) prob=calc_prob(diff*-1);
        else prob=calc_prob(diff);
        if(prob>float(Rand32()&mask(30))/mask(30)){
            testCounter.count("minus_change");
            return true;
        }
        else return false;
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

const int num_map=100;
const int num_use_map=8;
const int height=50;
const int width=50;
const int max_length_command=2500;

auto hash_map=vmake(height,width,u64());

struct Map{
    char map_[height][width];
    int now_h,now_w;
    int score=0;
    bool traped=false;

    DynamicArray<pii,3000> pre_now;
    DynamicArray<char,3000> pre_state;

    bool is_wall(int h,int w){
        return map_[h][w]=='#';
    }
    bool is_coin(int h,int w){
        return map_[h][w]=='o';
    }
    bool is_trap(int h,int w){
        return map_[h][w]=='x';
    }

    //実際に行動する
    void command(int dir){
        pre_now.push_back(pii(now_h,now_w));

        if(traped){
            pre_state.push_back('.');
            return;
        }
        int next_h=now_h+dh[dir];
        int next_w=now_w+dw[dir];
        if(is_wall(next_h,next_w)){
            pre_state.push_back('.');
            return;
        }
        now_h=next_h;
        now_w=next_w;
        pre_state.push_back(map_[next_h][next_w]);

        if(is_coin(next_h,next_w)){
            score++;
            map_[next_h][next_w]='.';
        }
        if(is_trap(next_h,next_w)){
            score-=5000;
            traped=true;
        }
        return;
    }
    void re_command(){
        map_[now_h][now_w]=pre_state.back();

        if(pre_state.back()=='x' and pii(now_h,now_w)!=pre_now.back()){
            score+=5000;
            traped=false;
        }else if(pre_state.back()=='o'){
            score--;
        }

        tie(now_h,now_w)=pre_now.back();

        pre_state.pop_back();
        pre_now.pop_back();
    }

    pair<int,u64> try_command(int dir){
        if(traped){
            return {score,hash_map[now_h][now_w]};
        }
        int next_h=now_h+dh[dir];
        int next_w=now_w+dw[dir];
        u64 hash=hash_map[next_h][next_w];
        if(is_wall(next_h,next_w)){
            return {score,hash_map[now_h][now_w]};
        }

        if(is_coin(next_h,next_w)){
            return {score+1,hash};
        }
        if(is_trap(next_h,next_w)){
            return {score-5000,hash};
        }
        return {score,hash};
    }

    void input(){
        for(int h=0;h<height;h++){
            for(int w=0;w<width;w++){
                cin>>map_[h][w];
                if(map_[h][w]=='@'){
                    now_h=h;
                    now_w=w;
                    map_[h][w]='.';
                }
            }
        }
        return;
    }
};


vector<Map> init_map(num_map);

void input(){
    int dummy;
    for(int i=0;i<5;i++){
        cin>>dummy;
    }
    for(int i=0;i<num_map;i++){
        init_map[i].input();
    }
}
using Score=int;
using Action=int;
using Hash=u64;

struct State{
    vector<Map> map_;
    int score=0;

    State(vector<Map> map):map_(map),score(0){

    }
    State(){

    }
    Score calc_score(){
        int sum=0;
        for(auto map:map_){
            sum+=map.score;
        }
        return sum;
    }

    bool operator<(State &state){
        return score<state.score;
    }
    // スコアの差分計算
    // 状態は更新しない
    //返り値: (score,hash)
    //op:操作 score,hash: 元の値??
    pair<Action,Hash> try_apply(int dir,u64 hash_){
        // cerr << "old: " << score << " " << hash << endl;

        int score=0;
        u64 hash=0;
        for(int i=0;i<num_use_map;i++){
            int s;
            u64 h;
            tie(s,h)=map_[i].try_command(dir);
            score+=s;
            hash^=h;
        }

        return {score,hash};
    }
    // 状態を更新する
    // 元の状態に戻すための情報を返す
    //op:操作
    Action apply(Action dir){
        
        for(auto &m:map_){
            m.command(dir);
        }

        return -1;
    }
    // applyから返された情報をもとに状態を元に戻す
    void back(Action op){
        for(auto &m:map_){
            m.re_command();
        }
    }
    //次の操作を列挙
    vector<int> find_next_actions(){
        return {0,1,2,3};
    } 
};

template<class Action,class Score,class Hash,uint num_branch>
struct BeamSearch{
    struct Node{
        Action action;
        uint parent;
        uint num_child;
        Score score;
        Hash hash;
        DynamicArray<pair<Action,uint>,num_branch> children;
    };
    struct Candidate{
        Action act;
        uint parent;
        Score score;
        u64 hash;
        uint priority;
    };

    struct Tree{
        uint node_size = 1;

        State state;
        uint cur_pos = 0; //candidate[cur_pos]が現在の状態
        uint candidate_size = 0; //candidateが現在なんこあるか
        uint rank = 0; //現在の位置.nターン後の状態だったらn

        vector<Node> node;
        vector<Candidate> candidate;
        set<int> S;

        Tree(State state):state(state){

        }

        // 注意: depthは深くなっていくごとに-1されていく
        void dfs(bool single,int depth,uint &priority){
            if(depth==0){
                //depthが0になると展開する
                Score score = node[cur_pos].score;
                Hash hash = node[cur_pos].hash;

                // 検算
                //cerr<<score<<" "<<state.calc_score()<<endl;
                assert(score==state.calc_score());

                // 次の操作を列挙
                vector<Action> next_actions = state.find_next_actions();

                //candidateに突っ込む
                for(Action next_action:next_actions){
                    Score next_score;
                    Hash next_hash;
                    tie(next_score,next_hash)=state.try_apply(next_action,hash);
                    candidate[candidate_size] = Candidate{next_action,cur_pos,next_score,next_hash,priority};
                    candidate_size++;
                    priority--;
                }
            }else{
                uint backup_pos = cur_pos;
                auto& cur_children = node[cur_pos].children;
                // 無駄なノードの削除
                //cerr<<-1<<endl;
                if(depth!=1){
                    //cur_childrenの中のnode_weightが0なものを削除
                    int num_erase=0;
                    for(uint i=0;i<cur_children.size();i++){
                        pair<Action,uint> &c=cur_children[i];
                        if(node[c.second].num_child==0){
                            num_erase++;
                        }else{
                            cur_children[i-num_erase]=cur_children[i];
                        }
                    }
                    cur_children.resize(cur_children.size()-num_erase);
                    //cur_children.erase(remove_if(cur_children.begin(),cur_children.end(),[&](pair<Action,uint>& x){return node[x.second].num_child == 0;}), cur_children.end());
                }
                //cerr<<-2<<endl;
                bool next_single=(single&cur_children.size()==1);
                if(depth==1){
                    priority=100;
                }

                //潜るからrankを足す
                rank++;
                for(const auto& x:cur_children){
                    auto &op=x.first;
                    auto &pos=x.second;
                    Action backup=state.apply(op);
                    cur_pos = pos;
                    dfs(next_single,depth-1,priority);
                    if(!next_single){
                        state.back(backup);
                    }
                }
                if(!next_single){
                    //親に戻る
                    cur_pos = backup_pos;
                    --rank;
                }
            }
        }
        void add_node(Candidate candidate){
            node[node_size].action = candidate.act;
            node[node_size].parent = candidate.parent;
            node[node_size].score = candidate.score;
            //cerr<<-3<<endl;
            node[candidate.parent].children.push_back({candidate.act, node_size});
            //cerr<<-4<<endl;
            node_size++;
            node[candidate.parent].num_child++;
        }
        //L~Rの無駄なnodeを削除
        void erase_node(uint L,uint R){
            // 無駄なノードの重み変更
            for(uint i = L;i<R;i++){
                if(node[i].num_child==0){
                    uint cur = i;
                    while(node[cur].num_child==0){
                        uint par = node[cur].parent;
                        node[par].num_child--;
                        cur = par;
                    }
                }
            }        
        }
        vector<Action> restore_action(uint node_id){
            vector<Action> ret;
            uint now_node=node_id;
            // 操作の復元
            while(now_node!=0){
                Action act=node[now_node].action;
                uint par=node[now_node].parent;
                ret.push_back(act);
                now_node=par;
            }
            reverse(ret.begin(),ret.end());
            return ret;
        }

        //上位beam_width個を残す
        void select_candidate(uint beam_width){
            sort(candidate.begin(),candidate.begin()+candidate_size,[&](Candidate& a,Candidate& b){
                    if(a.score==b.score) return a.priority>b.priority;
                    return a.score>b.score;
            });
            // if(candidate_size>beam_width){
            //     nth_element(candidate.begin(),candidate.begin()+beam_width,candidate.begin() + candidate_size,[&](Candidate& a,Candidate& b){
            //             if(a.score==b.score) return a.priority>b.priority;
            //             return a.score>b.score;
            //     });
            // }



            unordered_set<u64> hash_set;
            int cnt=0;
            //nodeに追加
            for(int i = 0;(i<candidate_size and cnt<beam_width);i++){
                if(hash_set.count(candidate[i].hash)==0){
                    cnt++;
                    hash_set.insert(candidate[i].hash);
                    add_node(candidate[i]);
                }
            }
            //cerr<<cnt<<endl;
            //cerr<<-6<<endl;
        }
    };
    State init_state; //初期解
    uint beam_width;

    BeamSearch(){

    }

    vector<Action> beam(uint beam_width_,uint max_turn,State init_state_){
        beam_width=beam_width_;
        init_state=init_state_;


        Tree tree(init_state);

        const uint SIZE=beam_width_*max_turn+10;
        tree.node.resize(SIZE);
        tree.candidate.resize(SIZE);

        uint L = 0;
        uint R = 1;
        for(uint turn=0;turn<max_turn;++turn){
            tree.candidate_size = 0;

            //single,p,depth
            //depthは0
            uint priority=0;
            tree.dfs(true,turn-tree.rank,priority);

            if(turn+1!=max_turn){
#ifdef ONLINE_JUDGE
                if(TIME.span()>=3700){
                    beam_width=1;
                }
#endif
                tree.select_candidate(beam_width);
                tree.erase_node(L,R);
                L = R;
                R = tree.node_size;
            }
            // cerr << turn << " " << node_size << " " << tree.rank << endl;
        }

        // 最良の状態を選択
        int arg_max=-1;
        Score max_score=0;
        for(uint i=0;i<min(tree.candidate_size,uint(beam_width));++i){
            if(tree.candidate[i].score>=max_score){
                max_score=tree.candidate[i].score;
                arg_max=i;
            }
        }

        cerr<<"score: "<<tree.candidate[arg_max].score<<endl;
        // cout<<"rank: "<<num_turn-tree.rank<<endl;



        tree.add_node(tree.candidate[arg_max]);
        auto actions = tree.restore_action(tree.node_size-1);

        return actions;
    }
    //参考: https://qiita.com/rhoo/items/f2be256cde5ad2e62dde
    //参考: https://atcoder.jp/contests/ahc021/submissions/42988795
};

void solve(){
    input();

    for(int h=0;h<height;h++) for(int w=0;w<width;w++){
        hash_map[h][w]=Rand64();
    }

    vector<int> num_coin(num_map,0);
    for(int i=0;i<num_map;i++){
        for(int h=0;h<height;h++){
            for(int w=0;w<width;w++){
                char c=init_map[i].map_[h][w];
                if(c=='o') num_coin[i]++; 
            }
        }
    }

    //使うマップを選ぶ
    vector<int> order;
    for(int i=0;i<num_map;i++) order.push_back(i);
    sort(order.begin(),order.end(),
        [&](int i,int j){
            return num_coin[i]>num_coin[j];
        });
    vector<Map> use_map;
    order.resize(num_use_map);
    cout<<order<<endl;
    for(int i:order){
        use_map.push_back(init_map[i]);
    }

    //ビームを打つ
    const int beam_width=1600;
    State init_state(use_map);

    BeamSearch<int,int,u64,4> beam;
    auto ret=beam.beam(beam_width,max_length_command,init_state);
    for(int dir:ret){
        if(dir==0) cout<<"R";
        else if(dir==1) cout<<"D";
        else if(dir==2) cout<<"L";
        else cout<<"U";
    }
    cout<<endl;
    cerr<<TIME.span()<<endl;
#ifndef ONLINE_JUDGE
    testTimer.output();
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
    FastIO();
    int T=1;
    //cin>>T;
    while(T--) solve();
}