训练一个神经网络模型，将一组低信噪比图像(mics)和一组对应的高信噪比图像(projs)分别编码到向量空间，做Contrastive learning

模型的工作目标是：给模型一张低信噪比的mic，模型从几万或十几万张高信噪比的projs中，找到和该mic最相似的高信噪比的proj。
注意：配对的mic和proj有不同的噪声水平，且可能存在相对的in plane shifts和in plane rotation，且mic经过了CTF调制(调制方法见@project.py中的ctf部分;另外project.py也可以用来从3D volume中产生大量模拟projs)

模型采用实空间+频域双分支：一个分支看实空间图像，另一个分支看Fourier空间图像
loss用InfoNCE / supervised contrastive loss

最推荐模型：Siamese / contrastive encoder。

训练时输入一对：

noisy particle 和 对应 clean projection

让 encoder 把它们映射到同一个 embedding 空间：

E_noisy(noisy) ≈ E_clean(clean_projection)

然后推理时：

预先生成几万/几十万张 clean projections；
对所有 clean projections 编码，建 FAISS index；
对 noisy particle 编码；
检索 top-k clean projections；
对 top-k 再用传统算法精修 in-plane rotation、XY shift、CTF/defocus。

这比 regression denoise 更合理。

模型可以选：

轻量 baseline：ResNet18 / ConvNeXt-Tiny encoder
最适合先验证可行性。

更强一点：ViT-small / Swin-T encoder
对全局结构更敏感，但训练数据要更多。

最适合 cryo-EM 的版本：频域 + 实空间双分支 encoder
一个分支看 real-space particle，一个分支看 Fourier amplitude / phase / bandpass channels。cryo-EM 的 CTF、低 SNR、频率壳层信息很重要，纯自然图像 encoder 不一定好。

loss 用：

InfoNCE / supervised contrastive loss
正样本：对应 clean projection。
负样本：其他方向/其他 in-plane rotation 的 projections。
最好加 hard negatives：相邻 viewing direction、相似 silhouette、对称相关 projection。

但有一个关键点：不要只检索一张。
极低 SNR 下 top-1 很容易错。应该检索 top-20/top-100，然后用物理评分重排：

score = masked NCC / whitened NCC / CTF-aware correlation

注意生成合理的项目目录结构

@emd_19110.map是一个3D volume，可以用python mrcfile读取它，并用@project.py产生simulated projs
