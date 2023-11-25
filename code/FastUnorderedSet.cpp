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
    //最初から見ていき、一致したものを削除(remove)
    bool erase(T value){
        for(int i=0;i<size_;i++){
            if(array_[i]==value){
                remove(i);
                return true;
            }
        }
        return false;
    }
    void reverse(){
        for(int i=0;i<size_/2;i++){
            swap(array_[i],array_[size_-1-i]);
        }
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


//重複は許可しない
//ほとんどが期待計算量: O(1)
//範囲for文に対応しているが、削除された要素が入っている可能性がある
//1e9,1e5で125MBぐらい
template<int SIZE,int CAPACITY>
struct FastUnorderedSet{
    bitset<SIZE> set_;
    DynamicArray<int,CAPACITY> elements_; //削除されたものも入っている可能性がある
    int erased_cnt=0; //削除されたが、まだelements_に残っている要素の数

    FastUnorderedSet(){

    }
    void insert(int v){
        assert(v>=0 and v<SIZE);
        assert(set_.test(v)==false);
        set_.flip(v);
        elements_.push_back(v);
    }

    void remove(int v){
        assert(set_.test(v));
        set_.flip(v);
        erased_cnt++;
    }

    bool contains(int v)const{
        return set_.test(v);
    }
    //O(要素の数)
    int size()const{
        return elements_.size()-erased_cnt;
    }

    int random(){
        while(true){
            assert(elements_.size());
            int idx=Rand32(elements_.size());
            if(set_.test(elements_[idx])){
                return elements_[idx];
            }else{
                elements_.swap_remove(idx);
                erased_cnt--;
            }
        }
    }

    int random_extract(){
        while(true){
            assert(elements_.size());
            int idx=Rand32(elements_.size());
            int element=elements_[idx];
            elements_.swap_remove(idx);
            if(set_.test(element)){
                return element;
            }else{
                erased_cnt--;
            }
        }
    }
    void clear(){
        while(elements_.size()){
            if(set_.test(elements_.back())){
                set_.flip(elements_.back());
            }
            elements_.pop_back();
        }
        assert(erased_cnt==0);
    }
    //elements_の中の削除された要素を消す
    void build(){
        if(erased_cnt==0) return;
        for(int i=0;i<elements_.size() and erased_cnt;i++){
            int element=elements_[i];
            if(set_.test(element)) continue;
            elements_.swap_remove(i);
            i--;
        }
    }
    vector<int> make_vector(){
        build();
        vector<int> v;
        for(int x:elements_){
            v.push_back(x);
        }
        return v;        
    }
    friend ostream& operator<<(ostream& os,FastUnorderedSet& S) {
        vector<int> v=S.make_vector();
        sort(v.begin(),v.end());
        bool first=true;
        for (auto& e : v){
            if(not first) os<<" ";
            first=false;
            os<<e;
        }
        return os;
    }
	inline auto begin() -> decltype(elements_.begin()) {
		return elements_.begin();
	}

	inline auto end() -> decltype(elements_.begin()) {
		return elements_.begin() + elements_.size_;
	}

	inline auto begin() const -> decltype(elements_.begin()) {
		return elements_.begin();
	}

	inline auto end() const -> decltype(elements_.begin()) {
		return elements_.begin() + elements_.size_;
	}
};


void solve(){
    FastUnorderedSet<1000000001,100000> Set;
    int t=1;
    for(int i=0;i<9;i++){
        t*=10;
        Set.insert(t);
    }
    Set.remove(10);
    cerr<<Set<<endl;
    while(Set.size()){
        cerr<<Set.random_extract()<<endl;
    }
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
