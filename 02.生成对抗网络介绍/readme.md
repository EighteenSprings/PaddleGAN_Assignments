# 生成对抗网络介绍



## 〇、本节大纲

![课程大纲](./assets/Content.png)



## 一、生成对抗网络概述

![GAN_Intro](./assets/GAN_Intro.png)



![GAN_History](./assets/GAN_History.png)

学习过程中应该注意的部分：

1. 网络结构（√）
2. 条件生成网络
3. 图像翻译
4. 归一化和限制
5. 损失函数（√）
6. 评价指标（√）

√ 为学习过程中应该着重注意的部分





## 二、生成对抗网络原理



### 生成对抗网络介绍

生成对抗网络由生成器和判别器构成

- 生成器：希望骗过判别器
- 判别器：希望鉴别生成器生成数据是 fake 的

![GAN_Arch](./assets/GAN_Arch.png)

图中假的图片（Fake Image）分数设为 0，真实图片（Real Image）分数设为 1。

看图中目标函数
$$
\min_{G}\max_{D} = E_{x\sim P_r}[\log {D(x)}] + E_{z \sim P_z}[\log (1 - D(G(z)))]
$$
其中

$ x\sim P_r $ 代表 $ x $ 是从真实数据中取出来的，即 $ x $ 满足真实数据分布

$ D(x) $ 即 $ x $ 传入判别器（Discriminator）得到的分数，这里我们会取 log，得到 $ \log {D(x)} $

$ x \sim P_z $ 代表 $ z $ 是从随机噪声中取出来的，即这里得到的 $ z $ 满足随机噪声的分布

$ D(G(z)) $ 代表从随机噪声分布中采样得到的 $ z $ ，通过生成器（Generator），得到生成图片（Fake Image），再通过判别器，最后得到的分数。

**生成器**

对于 生成器，我们先看 $ \min_{G} $

因为

​	通过 $ z $ 生成的图像我们希望它接近真实数据

所以

​	$ z $ 经过判别器得到的分数我们希望它能越大越好

​	同时接近真实数据通过判别器得到的分数，即与 $ D(x) $ 越接近越好。 

​	即课中老师说的 ”生成的图片和真实的图片几乎一样“

**判别器**

对于 判别器，我们看 $ \min_D $

因为

​	我们希望通过 $ z $ 生成的图像能被鉴别器判断出来

所以

​	$ z $ 经过判别器得到的分数我们希望它越小越好

​	同时得到的分数 $ D(z) $ 要与真实数据通过判别器得到的分数 $ D(x) $ 相差越大越好

​	即 ”生成的图片和真实的图片只要有一点点不同，就会被判断出来“



### 生成对抗网络的迭代过程



![GAN_Autocoder](./assets/GAN_Autoencoder.png)

**Autocoder**

原图像通过 Encoder 编码为固定维度的 vector，再通过 Decoder 解码为输入图像尺度相同的图片，并通过计算该图片和原图的 L2 Loss 来使得生成的图尽可能与原图接近。

这样得到的网络，我们取出其中 Decoder 部分，通过调整 vector 能生成图片。

![GAN_diff](./assets/GAN_diff.png)



### 生成对抗网络的一点点理论

![GAN_Simple_Theory](./assets/GAN_Simple_Theory.png)

通过采样 [0, 1] 之间的均匀分布，得到一个向量（vector），生成器根据这个向量，去拟合真实分布，从而得到和目标图像接近的生成图像



![GAN_Simple_Theory3](./assets/GAN_Simple_Theory3.png)

上图是 GAN 训练过程的可视化，可以看到随着参数的更新，我们通过生成器拟合出的数据分布会逐渐向着真实数据接近，并且判别器的分类效果会越来越差并最终为 0.5（ 1 为真实数据得分，0 为生成数据得分），即无法正确判断。

![GAN_Simple_Theory4](./assets/GAN_Simple_Theory4.png)

判别器训练可以看作二分类问题，其中  $ \tilde{x} $ 为随机噪声 $ z $ 通过生成器得到的生成图像



![GAN_Simple_Theory5](./assets/GAN_Simple_Theory5.png)

