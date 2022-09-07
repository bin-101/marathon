import subprocess
import sys
import re
import os
import joblib
import matplotlib.pyplot as plt

input="input/?.txt"
output="output/?.txt"

post=-1



#コマンドを作る
def make_cmd(num,para=[]):
    input_file=input.replace('?',str(num).zfill(4))
    output_file=output.replace('?',str(num).zfill(4))
    cmd="./judge "+'"./a.out '
    for p in para:
        cmd+=" "+str(p)
    cmd+='"'
    cmd+=" <"+input_file
    cmd+=" >"+output_file
    print(cmd)
    return cmd

#実行する
def func(num,para=[]):
    cmd=make_cmd(num,para)
    res=subprocess.run(cmd,shell=True,encoding='utf-8',stderr=subprocess.PIPE)
    res=str(res)
    print(res)
    number = re.findall(r"\d+", res)
    #print("number",number)
    return int(number[post])

#コンパイルをする
def compile(program):
    arg=["g++",program+".cpp","-std=c++17","-Ofast","-DSCORE"]
    print(arg)
    subprocess.call(arg)

def MakeGraph(score,num):
    fig=plt.figure()
    plt.hist(score)
    fig.savefig("graph.png")

def execute(num,para=[],graph=False):
    core=os.cpu_count() - 2
    print("unko")
    print("core",core)
    print(make_cmd(0,para))
    score=joblib.Parallel(n_jobs=core)(joblib.delayed(func)(i,para) for i in range(num))
    sum=0
    for i in range(len(score)):
        sum+=score[i]
    # if graph:
    #     print("sum: {:,}".format(s))
    #     print("ave: {:,}".format(s/num))
    #     MakeGraph(score,num)
    return sum

#args[1].cpp: プログラム名   args[2]: 試すテストケースの個数 args[3]: 後ろから何番目のデータを使うか
if __name__ == '__main__':
    args=sys.argv
    compile(args[1]) #コンパイル
    if len(args)>=4:
        post=-int(args[3])
    score=execute(int(args[2]),[])
    print("sum: {:,}".format(score))
