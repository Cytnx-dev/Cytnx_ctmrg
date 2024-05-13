#include <cytnx.hpp>
#include <malloc.h>
#include <time.h> // time 函數所需之標頭檔
using namespace std;
int main(){

    clock_t start, end;
    double cpu_time_used;

    cytnx::cytnx_uint64 d = 2;
    cytnx::cytnx_uint64 D = 2;
    cytnx::cytnx_uint64 chi = 64;
	cytnx::UniTensor T = cytnx::UniTensor(cytnx::zeros({chi,D,D,chi})).set_labels({"0","1","2","3"});
	cytnx::UniTensor Pt2 = cytnx::UniTensor(cytnx::zeros({chi,D,D,chi})).set_labels({"0","8","9","4"});
	cytnx::UniTensor P1 = cytnx::UniTensor(cytnx::zeros({chi,D,D,chi})).set_labels({"3","10","11","7"});
	cytnx::UniTensor A = cytnx::UniTensor(cytnx::zeros({d,D,D,D,D})).set_labels({"12","1","8","5","10"});
	cytnx::UniTensor Aconj = cytnx::UniTensor(cytnx::zeros({d,D,D,D,D})).set_labels({"12","2","9","6","11"});
    


    // cytnx::Contract(T,cytnx::Contract(P1,cytnx::Contract(Pt2,cytnx::Contract(A,Aconj,false,false),false,false),false,false),false,false);
    cytnx::UniTensor t1 = cytnx::Contract(A,Aconj,false,false);
    cytnx::UniTensor t2 = cytnx::Contract(Pt2,t1,false,false);
    
    // 計算開始時間
    start = clock();
    cytnx::UniTensor t3 = cytnx::Contract(P1,t2,false,false);
    // 計算結束時間
    end = clock();
    
    // 計算實際花費時間
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

    printf("Time = %f\n", cpu_time_used);
    return 0;
}