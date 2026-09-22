# 大语言模型与Transformers
## 统计语言模型
其核心思想是，一个句子出现的概率，等于该句子中每个词出
现的条件概率的连乘。
$P(S)=P(w_1,w_2,...,w_n)=P(w_1)\cdot P(w_2|w_1)\cdot ...\cdot P(w_n|w_{n-1},...w_1)$
$P(w_n|w_{n-1},...w_1)$太难算了
要计算词序列$w_{n-1},...w_1$中的词都出现的情况下$w_n$出现的概率，而且词序列$w_{n-1},...w_1$都不一定在训练数据中出现
为了解决这个问题，提出了**马尔可夫假设(Markov Assumption)**，就是说$w_n$出现的概率只和前$n-1$个词有关，基于这个假设提出的统计语言模型就叫**N-gram**，$N$就代表上下文窗口大小
$N=2$时就叫**Bigram**，这种情况下$P(S)=P(w_1,w_2,...,w_n)=P(w_1)\cdot P(w_2|w_1)\cdot ...\cdot P(w_n|w_{n-1})$
虽然解决了计算的问题，但是还有两个问题：
- 其中有个词序列没在语料库里出现过就会导致概率为0，语料库里没出现不代表现实中没有
- 泛化能力差，无法理解语义相似性
## 神经网络语言模型
与**N-gram**把词视作一个个单独的符号不同，神经网络将词视为向量，将输入的每个词转化成一个向量，也叫词嵌入。语义相似的词在高维向量空间的位置也会更靠近
### 前馈神经网络语言模型
随机初始化每个词的词嵌入，利用神经网络的学习能力来学习一个函数（就是一些隐藏层，学习权重参数）。输入前$n-1$个词嵌入，输出这每个词在第$n$个位置出现的概率，同时还会反向传播来调整每个词的词嵌入使其具有丰富的语义
有了词嵌入，我们就可以来计算它们的相似度，比如通过$cos\theta$
虽然这个解决了语义相似性的问题，但是还有个问题就是上下文窗口是固定的
### RNN与LSTM
上下文窗口是固定的，导致预测时只能看到前$n-1$个词，再往前的信息就丢失了，于是想为它添加记忆功能
RNN引入了隐藏状态向量，对于第n个词，RNN根据第n-1词的词嵌入和上一个时间步的隐藏向量，得到更新的隐藏向量，这个更新的隐藏向量可以转化成第n个词中每个词出现的概率
理论上隐藏状态向量可以包含无限长的序列信息，但是实际上序列过长会导致反向传播梯度爆炸
为了解决这个问题，LSTM出现了，它引入了细胞状态和门控机制
### Transformer
前面提到的RNN和LSTM是按照时间步一步一步来的，无法执行大规模的并行计算，为了提高计算效率，Transformer出现了，它抛弃了循环结构，引入了注意力机制
其输入是序列，输出也是序列。它一开始是用来机器翻译的，比如输入一段中文，输出翻译的英文。到现在用到那些大模型输入一段话，输出也是一段话。
#### 01|词嵌入
为一段话的每个词生成词嵌入，比如“我”$\rightarrow \boldsymbol{a}\in \mathbb{R}^{1\times 50}$
#### 02|位置编码
为每个词嵌入生成位置编码$\boldsymbol{p}\in \mathbb{R}^{1\times 50}$，并相加$\boldsymbol{a}+\boldsymbol{p} = \boldsymbol{a_p}$
![[Pasted image 20260715163810.png]]
#### 03|self-Attension
$\boldsymbol{a_p}$与三个可学习的权重矩阵${\boldsymbol{W}^{Q},\boldsymbol{W}^{K},\boldsymbol{W}^{V}}$相乘得到三个向量$\boldsymbol{Q}_a,\boldsymbol{K}_a,\boldsymbol{V}_a$
分别代表：
- $\boldsymbol{Q}_a$我想找什么，查询
- $\boldsymbol{K}_a$我是谁，键
- $\boldsymbol{V}_a$我的值是什么，值
 $\boldsymbol{Q}_a$要与$(\boldsymbol{K}_a^{T},\boldsymbol{K}_b^{T},....)$相乘，再经过放缩和$\mathsf{softmax}$得到值权重$(w_{a,a},w_{a,b},...)$这是一条行向量，这条行向量与值矩阵相乘$$(w_{a,a},w_{a,b},...) \cdot \begin{pmatrix}

\boldsymbol{V}_a \\

\boldsymbol{V}_b \\
\vdots

\end{pmatrix}$$，相当于让每个词的值向量加权求和，得到一个行向量，这个就是$\boldsymbol{a_p}$代表的词看到了其他词后对自己语义的更新，就相当于更新了自己的词嵌入，使其更符合当前语义了，这种加权求和的方式和GCN中的邻居聚合$\boldsymbol{A}\boldsymbol{X}$挺像的
![[Pasted image 20260716101049.png]]
而多头注意力机制就是从多个角度去更新词嵌入，上面的只是与一组权重矩阵${\boldsymbol{W}^{Q},\boldsymbol{W}^{K},\boldsymbol{W}^{V}}$作运算来更新，而多头是并行与多组权重矩阵作运算，得到不同角度的词嵌入，将其组合得到新的词嵌入
，用GCN的说法来说就是，一个特征矩阵与不同的邻接矩阵聚合（也就是不同的图）得到不同的更新特征矩阵，再将其汇总起来得到新的特征矩阵
**通过这种方式每个词都更新了自己的词嵌入**
#### 04|残差连接与层归一化

self-Attension让每个词得到了新的词嵌入，那之前的词嵌入也不能丢，需要把之前的词嵌入和新的词嵌入加起来再进行归一化，这和GCN中给邻接矩阵自连接，以及GraphSAGE中CONCAT自己的特征和邻居聚合的特征是一样的
#### 05|FFN
作用和self-Attension一样也是对词嵌入进行更新，不过侧重点不同，FFN是让词嵌入更复杂，是纵向的，而self-Attension是横向的，收集更广的信息。
![[Pasted image 20260716105451.png]]
与self-Attension一样后面也要接个残差连接和归一化，不能丢失原来的信息
下面只追踪 \(\delta=5\) 的贡献，并把每一个16-slot向量都写出来。为了简化记号，令：

\[ \alpha_r=a_5[r],\qquad r=0,\ldots,7. \]

因此：

\[ a_5=[\alpha_0,\alpha_1,\alpha_2,\alpha_3, \alpha_4,\alpha_5,\alpha_6,\alpha_7]. \]

这里：

\[ \alpha_r=A[r,(r+5)\bmod8]. \]

我们先追踪输出：

\[ O_0=[h_0\mid h_1]. \]

---

# 1. 输入密文的具体内容

四个输入 GBE 向量：

\[ \begin{aligned} g_0={}&[z_{00},z_{11},z_{22},z_{33}, z_{40},z_{51},z_{62},z_{73}],\\ g_1={}&[z_{01},z_{12},z_{23},z_{30}, z_{41},z_{52},z_{63},z_{70}],\\ g_2={}&[z_{02},z_{13},z_{20},z_{31}, z_{42},z_{53},z_{60},z_{71}],\\ g_3={}&[z_{03},z_{10},z_{21},z_{32}, z_{43},z_{50},z_{61},z_{72}]. \end{aligned} \]

两个输入密文为：

\[ C_0=[g_0\mid g_1], \qquad C_1=[g_2\mid g_3]. \]

使用 \(g_k[r]\) 表示，具体是：

\[ \begin{aligned} C_0=[ &g_0[0],g_0[1],g_0[2],g_0[3], g_0[4],g_0[5],g_0[6],g_0[7]\\ \mid\;& g_1[0],g_1[1],g_1[2],g_1[3], g_1[4],g_1[5],g_1[6],g_1[7] ], \end{aligned} \]\[ \begin{aligned} C_1=[ &g_2[0],g_2[1],g_2[2],g_2[3], g_2[4],g_2[5],g_2[6],g_2[7]\\ \mid\;& g_3[0],g_3[1],g_3[2],g_3[3], g_3[4],g_3[5],g_3[6],g_3[7] ]. \end{aligned} \]

---

# 2. \(\delta=5\) 要得到什么

分解：

\[ \delta=5=q+j=4+1. \]

对于：

\[ O_0=[h_0\mid h_1], \]

第一段 \(h_0\) 需要：

\[ (k-\delta)\bmod4=(0-5)\bmod4=3, \]

所以来源是：

\[ R_5(g_3). \]

第二段 \(h_1\) 需要：

\[ (1-5)\bmod4=0, \]

所以来源是：

\[ R_5(g_0). \]

因此，最终目标是：

\[ \boxed{ [a_5\odot R_5(g_3)\mid a_5\odot R_5(g_0)] }. \]

具体来说：

\[ R_5(g_3) = [g_3[5],g_3[6],g_3[7],g_3[0], g_3[1],g_3[2],g_3[3],g_3[4]], \]\[ R_5(g_0) = [g_0[5],g_0[6],g_0[7],g_0[0], g_0[1],g_0[2],g_0[3],g_0[4]]. \]

BSGS 不直接计算 \(R_5\)，而是拆成：

\[ R_5=R_4\circ R_1. \]

其中：

- \(R_1\)：baby shift；
- \(R_4\)：giant shift。

---

# 3. Baby normal rotation：\(\operatorname{Rot}_9\)

由于一个密文有两个8-slot segment，逻辑上的每段左移1，不能直接只用全局 \(\operatorname{Rot}_1\)。

baby normal 行 \(r=0,\ldots,6\) 使用：

\[ t_{\mathrm N}=1-8\equiv9\pmod{16}. \]

## 旋转 \(C_0\)

\[ \begin{aligned} \operatorname{Rot}_9(C_0)=[ &g_1[1],g_1[2],g_1[3],g_1[4], g_1[5],g_1[6],g_1[7],g_0[0]\\ \mid\;& g_0[1],g_0[2],g_0[3],g_0[4], g_0[5],g_0[6],g_0[7],g_1[0] ]. \end{aligned} \]

其中：

- 第一段前7槽是 \(g_1[1],\ldots,g_1[7]\)；
- 第二段前7槽是 \(g_0[1],\ldots,g_0[7]\)。

对于 \(O_0\)，我们要使用第二段的 \(g_0\)。

## 旋转 \(C_1\)

\[ \begin{aligned} \operatorname{Rot}_9(C_1)=[ &g_3[1],g_3[2],g_3[3],g_3[4], g_3[5],g_3[6],g_3[7],g_2[0]\\ \mid\;& g_2[1],g_2[2],g_2[3],g_2[4], g_2[5],g_2[6],g_2[7],g_3[0] ]. \end{aligned} \]

对于 \(O_0\)，我们要使用第一段的 \(g_3\)。

---

# 4. Baby wrap rotation：\(\operatorname{Rot}_1\)

第7行需要：

\[ (r+1)\bmod8=(7+1)\bmod8=0. \]

所以要单独取每段的第0个元素。

## 旋转 \(C_0\)

\[ \begin{aligned} \operatorname{Rot}_1(C_0)=[ &g_0[1],g_0[2],g_0[3],g_0[4], g_0[5],g_0[6],g_0[7],g_1[0]\\ \mid\;& g_1[1],g_1[2],g_1[3],g_1[4], g_1[5],g_1[6],g_1[7],g_0[0] ]. \end{aligned} \]

第二段最后一槽正好是：

\[ g_0[0]. \]

## 旋转 \(C_1\)

\[ \begin{aligned} \operatorname{Rot}_1(C_1)=[ &g_2[1],g_2[2],g_2[3],g_2[4], g_2[5],g_2[6],g_2[7],g_3[0]\\ \mid\;& g_3[1],g_3[2],g_3[3],g_3[4], g_3[5],g_3[6],g_3[7],g_2[0] ]. \end{aligned} \]

第一段最后一槽正好是：

\[ g_3[0]. \]

---

# 5. 为什么又分成三个行区间

baby shift 的边界在：

\[ 8-j=7. \]

所以：

- baby normal：\([0,7)\)
- baby wrap：\([7,8)\)

giant step 是 \(q=4\)，最终还要把 bucket 左旋4。因此 giant 边界在预旋转行：

\[ \rho=4. \]

最终输出行与 bucket 行的关系是：

\[ \rho=(r+4)\bmod8. \]

于是三个区间是：

|bucket 行 \(\rho\)|对应最终输出行 \(r\)|baby|giant|
|---|---|---|---|
|\([0,4)\)|\([4,8)\)|normal|wrap|
|\([4,7)\)|\([0,3)\)|normal|normal|
|\([7,8)\)|\(r=3\)|wrap|normal|

把它们记为：

- 区间 A：\([0,4)\)
- 区间 B：\([4,7)\)
- 区间 C：\([7,8)\)

---

# 6. 为什么 mask 权重顺序发生了变化

最终 giant rotation 是4步。

bucket 第 \(\rho\) 行最终会到达：

\[ r=(\rho-4)\bmod8. \]

所以 bucket 第 \(\rho\) 行的权重必须是：

\[ a_5[(\rho-4)\bmod8]. \]

因此：

|bucket 行 \(\rho\)|应放权重|
|---|---|
|0|\(a_5[4]=\alpha_4\)|
|1|\(a_5[5]=\alpha_5\)|
|2|\(a_5[6]=\alpha_6\)|
|3|\(a_5[7]=\alpha_7\)|
|4|\(a_5[0]=\alpha_0\)|
|5|\(a_5[1]=\alpha_1\)|
|6|\(a_5[2]=\alpha_2\)|
|7|\(a_5[3]=\alpha_3\)|

所以预旋转状态下的权重顺序是：

\[ [\alpha_4,\alpha_5,\alpha_6,\alpha_7, \alpha_0,\alpha_1,\alpha_2,\alpha_3]. \]

---

# 7. 六个具体 mask

每个行区间还要分别处理 segment 0 和 segment 1，因此一共6个 mask。

## 区间 A：\([0,4)\)，giant wrap

segment 0：

\[ M_{A,0}= [ \alpha_4,\alpha_5,\alpha_6,\alpha_7, 0,0,0,0 \mid 0,0,0,0,0,0,0,0 ]. \]

segment 1：

\[ M_{A,1}= [ 0,0,0,0,0,0,0,0 \mid \alpha_4,\alpha_5,\alpha_6,\alpha_7, 0,0,0,0 ]. \]

## 区间 B：\([4,7)\)，giant normal

segment 0：

\[ M_{B,0}= [ 0,0,0,0, \alpha_0,\alpha_1,\alpha_2,0 \mid 0,0,0,0,0,0,0,0 ]. \]

segment 1：

\[ M_{B,1}= [ 0,0,0,0,0,0,0,0 \mid 0,0,0,0, \alpha_0,\alpha_1,\alpha_2,0 ]. \]

## 区间 C：\([7,8)\)，baby wrap、giant normal

segment 0：

\[ M_{C,0}= [ 0,0,0,0,0,0,0,\alpha_3 \mid 0,0,0,0,0,0,0,0 ]. \]

segment 1：

\[ M_{C,1}= [ 0,0,0,0,0,0,0,0 \mid 0,0,0,0,0,0,0,\alpha_3 ]. \]

---

# 8. 哪个输入密文乘哪个 mask

对于 \(O_0=[h_0|h_1]\)：

- segment 0 需要 \(g_3\)，所以使用 \(C_1\)；
- segment 1 需要 \(g_0\)，所以使用 \(C_0\)。

因此流向 \(O_0\) 的六次乘法是：

\[ P_{A,0}=\operatorname{Rot}_9(C_1)\odot M_{A,0}, \]\[ P_{A,1}=\operatorname{Rot}_9(C_0)\odot M_{A,1}, \]\[ P_{B,0}=\operatorname{Rot}_9(C_1)\odot M_{B,0}, \]\[ P_{B,1}=\operatorname{Rot}_9(C_0)\odot M_{B,1}, \]\[ P_{C,0}=\operatorname{Rot}_1(C_1)\odot M_{C,0}, \]\[ P_{C,1}=\operatorname{Rot}_1(C_0)\odot M_{C,1}. \]

---

# 9. 六个乘法结果的具体内容

## \(P_{A,0}\)

\[ \begin{aligned} P_{A,0}=[ &\alpha_4g_3[1], \alpha_5g_3[2], \alpha_6g_3[3], \alpha_7g_3[4], 0,0,0,0\\ \mid\;& 0,0,0,0,0,0,0,0 ]. \end{aligned} \]

## \(P_{A,1}\)

\[ \begin{aligned} P_{A,1}=[ &0,0,0,0,0,0,0,0\\ \mid\;& \alpha_4g_0[1], \alpha_5g_0[2], \alpha_6g_0[3], \alpha_7g_0[4], 0,0,0,0 ]. \end{aligned} \]

## \(P_{B,0}\)

\[ \begin{aligned} P_{B,0}=[ &0,0,0,0, \alpha_0g_3[5], \alpha_1g_3[6], \alpha_2g_3[7], 0\\ \mid\;& 0,0,0,0,0,0,0,0 ]. \end{aligned} \]

## \(P_{B,1}\)

\[ \begin{aligned} P_{B,1}=[ &0,0,0,0,0,0,0,0\\ \mid\;& 0,0,0,0, \alpha_0g_0[5], \alpha_1g_0[6], \alpha_2g_0[7], 0 ]. \end{aligned} \]

## \(P_{C,0}\)

\[ \begin{aligned} P_{C,0}=[ &0,0,0,0,0,0,0, \alpha_3g_3[0]\\ \mid\;& 0,0,0,0,0,0,0,0 ]. \end{aligned} \]

## \(P_{C,1}\)

\[ \begin{aligned} P_{C,1}=[ &0,0,0,0,0,0,0,0\\ \mid\;& 0,0,0,0,0,0,0, \alpha_3g_0[0] ]. \end{aligned} \]

---

# 10. 累加到两个 giant bucket

区间 A 属于 giant wrap：

\[ B_{4,\delta=5}^{\mathrm W}[0] = P_{A,0}+P_{A,1}. \]

具体是：

\[ \begin{aligned} B_{4,\delta=5}^{\mathrm W}[0]=[ &\alpha_4g_3[1], \alpha_5g_3[2], \alpha_6g_3[3], \alpha_7g_3[4], 0,0,0,0\\ \mid\;& \alpha_4g_0[1], \alpha_5g_0[2], \alpha_6g_0[3], \alpha_7g_0[4], 0,0,0,0 ]. \end{aligned} \]

区间 B、C 属于 giant normal：

\[ B_{4,\delta=5}^{\mathrm N}[0] = P_{B,0}+P_{B,1}+P_{C,0}+P_{C,1}. \]

具体是：

\[ \begin{aligned} B_{4,\delta=5}^{\mathrm N}[0]=[ &0,0,0,0, \alpha_0g_3[5], \alpha_1g_3[6], \alpha_2g_3[7], \alpha_3g_3[0]\\ \mid\;& 0,0,0,0, \alpha_0g_0[5], \alpha_1g_0[6], \alpha_2g_0[7], \alpha_3g_0[0] ]. \end{aligned} \]

实际算法还会把 \(\delta=4,6,7\) 的贡献累加进相同的 \(B_4^{\mathrm N/W}\)。这里为了看清楚，只保留 \(\delta=5\) 的部分。

---

# 11. 对 normal bucket 做 giant rotation

计算：

\[ \operatorname{Rot}_4 \left(B_{4,\delta=5}^{\mathrm N}[0]\right). \]

左旋4步后：

\[ \begin{aligned} =[ &\alpha_0g_3[5], \alpha_1g_3[6], \alpha_2g_3[7], \alpha_3g_3[0], 0,0,0,0\\ \mid\;& \alpha_0g_0[5], \alpha_1g_0[6], \alpha_2g_0[7], \alpha_3g_0[0], 0,0,0,0 ]. \end{aligned} \]

它填充最终输出每段的前4行：

\[ r=0,1,2,3. \]

---

# 12. 对 wrap bucket 做 giant rotation

计算：

\[ \operatorname{Rot}_{12} \left(B_{4,\delta=5}^{\mathrm W}[0]\right). \]

因为16个槽中：

\[ 12\equiv-4\pmod{16}, \]

所以这是整体向右旋转4步。

得到：

\[ \begin{aligned} =[ &0,0,0,0, \alpha_4g_3[1], \alpha_5g_3[2], \alpha_6g_3[3], \alpha_7g_3[4]\\ \mid\;& 0,0,0,0, \alpha_4g_0[1], \alpha_5g_0[2], \alpha_6g_0[3], \alpha_7g_0[4] ]. \end{aligned} \]

它填充最终输出每段的后4行：

\[ r=4,5,6,7. \]

---

# 13. normal 与 wrap 相加

把两个 giant rotation 结果相加：

\[ \begin{aligned} T_{5\rightarrow O_0}=[ &\alpha_0g_3[5], \alpha_1g_3[6], \alpha_2g_3[7], \alpha_3g_3[0],\\ &\alpha_4g_3[1], \alpha_5g_3[2], \alpha_6g_3[3], \alpha_7g_3[4]\\ \mid\;& \alpha_0g_0[5], \alpha_1g_0[6], \alpha_2g_0[7], \alpha_3g_0[0],\\ &\alpha_4g_0[1], \alpha_5g_0[2], \alpha_6g_0[3], \alpha_7g_0[4] ]. \end{aligned} \]

也就是：

\[ \boxed{ T_{5\rightarrow O_0} = [a_5\odot R_5(g_3)\mid a_5\odot R_5(g_0)] }. \]

---

# 14. 全部换成原始 \(z_{rc}\)

因为：

\[ g_3= [z_{03},z_{10},z_{21},z_{32}, z_{43},z_{50},z_{61},z_{72}], \]

所以：

\[ R_5(g_3) = [z_{50},z_{61},z_{72},z_{03}, z_{10},z_{21},z_{32},z_{43}]. \]

又因为：

\[ g_0= [z_{00},z_{11},z_{22},z_{33}, z_{40},z_{51},z_{62},z_{73}], \]

所以：

\[ R_5(g_0) = [z_{51},z_{62},z_{73},z_{00}, z_{11},z_{22},z_{33},z_{40}]. \]

因此 \(\delta=5\) 对 \(O_0\) 的最终贡献是：

\[ \boxed{ \begin{aligned} [ &\alpha_0z_{50}, \alpha_1z_{61}, \alpha_2z_{72}, \alpha_3z_{03}, \alpha_4z_{10}, \alpha_5z_{21}, \alpha_6z_{32}, \alpha_7z_{43}\\ \mid\;& \alpha_0z_{51}, \alpha_1z_{62}, \alpha_2z_{73}, \alpha_3z_{00}, \alpha_4z_{11}, \alpha_5z_{22}, \alpha_6z_{33}, \alpha_7z_{40} ]. \end{aligned} } \]

例如：

- 第0槽：\(A_{05}z_{50}\)，贡献给 \(H_{00}\)
- 第3槽：\(A_{30}z_{03}\)，贡献给 \(H_{33}\)
- 第4槽：\(A_{41}z_{10}\)，贡献给 \(H_{40}\)
- 第8槽：\(A_{05}z_{51}\)，贡献给 \(H_{01}\)

都与普通矩阵乘法一致。

---

# 15. 同时产生的 \(O_1\) 贡献

相同的6个 mask 还会作用于另一个输入密文，构造：

\[ O_1=[h_2\mid h_3]. \]

它最终得到：

\[ [a_5\odot R_5(g_1)\mid a_5\odot R_5(g_2)]. \]

其中：

\[ R_5(g_1) = [z_{52},z_{63},z_{70},z_{01}, z_{12},z_{23},z_{30},z_{41}], \]\[ R_5(g_2) = [z_{53},z_{60},z_{71},z_{02}, z_{13},z_{20},z_{31},z_{42}]. \]

所以：

\[ \begin{aligned} T_{5\rightarrow O_1}=[ &\alpha_0z_{52}, \alpha_1z_{63}, \alpha_2z_{70}, \alpha_3z_{01}, \alpha_4z_{12}, \alpha_5z_{23}, \alpha_6z_{30}, \alpha_7z_{41}\\ \mid\;& \alpha_0z_{53}, \alpha_1z_{60}, \alpha_2z_{71}, \alpha_3z_{02}, \alpha_4z_{13}, \alpha_5z_{20}, \alpha_6z_{31}, \alpha_7z_{42} ]. \end{aligned} \]

因此 \(\delta=5\) 总计：

- 6种 plaintext masks；
- 每种 mask 作用于两个输入密文；
- 共12次 PMult。

但这里的 \(\operatorname{Rot}_9(C_g)\) 和 \(\operatorname{Rot}_1(C_g)\) 不是专门为 \(\delta=5\) 生成的：它们同时被 \(\delta=1\) 复用。最终的 giant rotations 也不是为 \(\delta=5\) 单独执行，而是在 \(\delta=4,5,6,7\) 全部累加进 \(B_4^{\mathrm N/W}\) 后，每个 bucket 统一旋转一次。