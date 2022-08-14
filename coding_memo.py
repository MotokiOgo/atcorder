# チートシート
## 1.入力受け取り
import sys
input = sys.stdin.readline
N = int(input())
X, Y = map(int, input().split())

# 1行×X列
arr = list(map(int, input().split()))
# N行×1列
arr = [int(input()) for _ in range(N)]
# N行×X列
arr = [list(map(int, input().split())) for _ in range(N)]

## 2.配列操作
### 逆順にする
arr = arr[::-1]
### -1倍して配列受け取り
a = list(map(lambda x: int(x)*(-1), input().split()))
### 多次元配列の二個目とかのキーでソートしたいとき
arr.sort(key=lambda x:x[1])
arr = sorted(arr, reverse=True, key=lambda x: x[1])
### 多次元配列初期化
arr = [[] for _ in range(N)]
arr = [[0] * 13 for _ in range(N)]
arr = [[0 for _ in range(X)] for _ in range(Y)]

## 3.その他
### MOD
MOD = 10**9 + 7
### 再帰上限解放
import sys
sys.setrecursionlimit(10**7)
### 三項演算子
# (変数) = (条件がTrueのときの値) if (条件) else (条件がFalseのときの値)

## 4.辞書操作
### 辞書初期化
dic = {}
### 値があれば+1無ければ追加
for i in range(N):
    tmp = input()
    if tmp in dic:
        dic[tmp] += 1
    else:
        dic[tmp] = 1
### 辞書のキーを全て出力
for key in dic.keys():
    print(key)
### 辞書の要素を全て出力
for value in dic.values():
    print(value)
### 辞書の要素の最大値取得
max_v = max(dic.values())

## 5.print
### 改行なしprint
print("HOGE", end="")
### 変数出力print
print(X,Y)
print("{} {}".format(N, X+Y))

## 6.count系
### 要素数数え上げ
arr.count(0)
### 最頻値算出
from collections import Counter
tmp_list = [1, 1, 1, 1, 0, 1, 1]
counter = Counter(tmp_list)
print(counter.most_common()[0][0])

## 7.math
### ルート
import math
math.sqrt(N)
### 階乗
import math
math.factorial(5)
### (2**n)%MODを高速に計算
pow(2,N,MOD)
### 順列、組み合わせ
import math
def P(n, r):
    return math.factorial(n)//math.factorial(n-r)
def C(n, r):
    return P(n, r)//math.factorial(r)

### nCk % MODを高速に計算
MOD = 10**9 + 7
def comb(n, k, MOD):
    if n < k or n < 0 or k < 0:
        return 0
    if k == 0:
        return 1
    iinv = [1] * (k + 1)
    ans = n
    for i in range(2, k + 1):
        iinv[i] = MOD - iinv[MOD % i] * (MOD // i) % MOD
        ans *= (n + 1 - i) * iinv[i] % MOD
        ans %= MOD
    return ans

### 切り上げ処理
import math
N, K = map(int, input().split())
print(math.ceil((N-1)/(K-1)))

### 素因数分解
import math
def trial_division(n):
    a = [1]
    for i in range(2,int(math.sqrt(n)) + 1):
        while n % i == 0:
            n //= i
            a.append(i)
    a.append(n)
    return a

### 約数列挙
import math
def make_divisors(n):
    divisors = []
    for i in range(1, int(n**0.5)+1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n//i)
    return divisors

### 最大公約数
import fractions
A, B = map(int, input().split())
max_n = fractions.gcd(A,B)

### 最小公倍数
def lcm(x, y):
    return (x * y) // fractions.gcd(x, y)

### 順列生成
import itertools
a = [1,2,3,4]
for i in itertools.permutations(a):
    print(i)

### 浮動小数点の精度を50桁にする(デフォルトでは28桁)
from decimal import *
getcontext().prec = 50

### 素数判定
import math
def is_prime(n):
    if n == 1: return False
    for k in range(2, int(math.sqrt(n)) + 1):
        if n % k == 0:
            return False
    return True

### エラストテネスの篩
def primes(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if not is_prime[i]:
            continue
        for j in range(i * 2, n + 1, i):
            is_prime[j] = False
    # パターン１
    return is_prime
    # パターン２
    # return [i for i in range(n + 1) if is_prime[i]]


## 8.探索系
### bit探索テンプレ
n = 4
for i in range(2 ** n):
    tmp = [0]*n
    for k in range(n):
        if 1 & i>>k:
            tmp[k] = 1
    print(tmp)
n = 4
for i in range(2 ** n):
    tmp = ''
    for k in range(n):
        if 1 & i>>k:
            tmp = '1' + tmp
        else:
            tmp = '0' + tmp
    print(tmp)

### 二分探索テンプレ
left = 0
right = 10**9
while abs(left-right) > 1:
    tmp = (left+right)//2
    if arr[tmp] < i:
        left = tmp
    else:
        right = tmp

### ワーシャルフロイド法
def warshall_floyd(d):
    #d[i][j]: iからjへの最短距離
    for k in range(n):
        for i in range(n):
            for j in range(n):
                d[i][j] = min(d[i][j],d[i][k] + d[k][j])
    return d
n,w = map(int,input().split()) #n:頂点数　w:辺の数
d = [[float("inf")]*n for i in range(n)]
#d[u][v] : 辺uvのコスト(存在しないときはinf)
for i in range(w):
    x,y,z = map(int,input().split())
    d[x][y] = z
    d[y][x] = z
for i in range(n):
    d[i][i] = 0 #自身のところに行くコストは０
print(warshall_floyd(d))

## 9.キュー
### 優先度付きキュー
import heapq
heapq.heapify(a) #リストaのheap化
heapq.heappush(a,X) #heap化されたリストaに要素xを追加
heapq.heappop(a) #heap化されたリストaから最小値を削除＆その最小値を出力

## 10.アルファベット
dic = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
al = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

##グラフ初期化
import itertools
graph = [[] for _ in range(N)]
for _ in range(M):
    a, b = map(int, input().split())
    graph[a-1].append(b-1)
    graph[b-1].append(a-1)