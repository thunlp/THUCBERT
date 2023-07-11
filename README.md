# THUCBERT

## 介绍

THUCBERT是由清华大学自然语言处理与社会人文计算实验室开发的字符级中文预训练BERT模型。模型具有如下特点：

1. 训练语料质量高，包括图书、百科、报纸、期刊等97G语料，共计378亿字。

2. tokenizer基于字符，字表齐全，对于繁体和异体字会自动映射到对应简体，对非中英字符会映射到对应语种token。目前主流的中文BERT模型中文字符不全，大多沿用谷歌<a href="https://huggingface.co/bert-base-chinese">bert-base-chinese</a>的词表，根据维基百科语料统计而来，而它缺失了国家通用规范汉字表8105字中的2765字，例如镊、馊、犟、囵、鲠、殄、箪、廪、勠等中低频字。

3. 使用了基于字频的降采样策略。对字符进行MASK时，降低高频字的MASK概率，防止大量的训练集中在高频常见字上，提升模型对于低频字的理解能力。

4. 使用了**层次化的字词混合MASK策略**。

   基于整词MASK的模型和基于字MASK的模型各有优势，我们采用了字词混合的MASK策略，在整词MASK提升性能的同时，也训练对层次化语义的理解能力。具体的做法是将语料进行分词后，每个词再利用wordpiece进行细分，形成一个词到字的层次结构（例如：计算机→计算+机→计+算+机），采样时依据归一化的概率整体MASK其中一部分（例如：计算机、计算或单字）。

## 模型地址

| 模型名称    | MASK策略 | Hugginface地址🤗                                              |
| ----------- | -------- | ------------------------------------------------------------ |
| THUCBERT-cm | 字       | <a href="https://huggingface.co/chengzl18/thucbert-cm">thucbert-cm</a> |
| THUCBERT-mm | 层次化   | <a href="https://huggingface.co/chengzl18/thucbert-cm">thucbert-mm</a> |

## 使用方式

可以通过如下代码使用THUCBERT：

```python
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("chengzl18/thucbert-mm", trust_remote_code=True)
model = AutoModel.from_pretrained("chengzl18/thucbert-mm")
```

使用方法和<a href="https://huggingface.co/bert-base-chinese">bert-base-chinese</a> 相同。

## 训练效果

##### PPL

在随机语料上进行验证，THUCBERT训练完成时的perplexity为2.20，显著低于bert-base-chinese的2.78。（需要注意perplexity也与词表有关，此对比仅供参考）

##### 字表示

字表示（采用embedding层的最近邻）如下：

THUCBERT

```
美: 靓 丑 韩 丽 英 艳 雅 魅 赏 绘
戏: 剧 玩 嬉 讽 娱 耍 舞 谑 棋 赌
麦: 稻 荞 薯 粱 枣 椰 稞 秫 麸 豌
今: 昨 昔 此 迄 前 咱 每 崭 现 迩
寻: 找 觅 追 溯 讨 谋 搜 探 挖 询
```

bert-base-chinese

```
美: 英 德 香 欧 雅 国 韩 歐 韓 國
戏: 戲 剧 娱 游 game 乐 艺 话 诗 玩 画
麦: 麥 玛 叶 马 兰 贝 荞 饼 凯 黄
今: 昨 2016 此 现 2015 現 前 2017 近 每
寻: 尋 觅 找 覓 讨 搜 询 尝 谋 选
```

##### 掩码预测

MASK预测效果如下：

THUCBERT

```
生活的真谛是[MASK]。: 爱 乐 诗 美 福 善 富 笑 渔 穷
我去吃了北京烤[MASK]。: 鸭 串 肉 鸡 饼 鱼 肠 鹅 兔 羊
唯江上之清风，与山间之明月，耳得之而为[MASK]，目遇之而成色。: 声 音 美 丽 佳 妙 香 乐 清 诗
凡事都有两面性，我们要[MASK][MASK]地看待。: 辩 正 辨 客 矛 科 认 冷 平 理
凡事都有两面性，我们要[MASK][MASK]地看待。: 证 观 确 性 学 静 面 衡 辩 极
```

bert-base-chinese

```
生活的真谛是[MASK]。: 美 爱 乐 人 ： 笑 - 玩 活 好
我去吃了北京烤[MASK]。: 肉 鸭 鱼 鴨 鸡 串 羊 饼 肠 的
唯江上之清风，与山间之明月，耳得之而为[MASK]，目遇之而成色。: 声 音 光 香 形 味 风 耳 心 闻
凡事都有两面性，我们要[MASK][MASK]地看待。: 正 客 冷 认 理 平 公 坦 科 清
凡事都有两面性，我们要[MASK][MASK]地看待。: 观 性 确 等 平 容 慎 面 理 心
```

##### 下游任务

在我们已进行的测试中，THUCBERT在各种文本分类任务上与[哈工大的BERT模型](https://huggingface.co/hfl/chinese-bert-wwm-ext)效果相当，在中文分词（[DeepTHULAC](https://github.com/thunlp/DeepTHULAC)基于THUCBERT-cm开发而成）、命名实体识别和语法改错任务上有明显的性能提升。

