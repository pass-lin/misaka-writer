# misaka-writer

## ai-续写小说

基于encoder-decoder结构的续写小说模型，模型比较小只有8kw，模型结构是魔改的transformer模型。

## 依赖环境

本项目的依赖有：tensorflow bert4keras jieba pandas。

如果使用GPU请安装 cuda 和 cudnn。

推荐的配置为 tensorflow 2.3.0，cuda 10.1，cudnn 7.6。

对于不支持 cuda 10 的 30 系显卡，建议使用 tensorflow 2.5.0，cuda 11.2，cudnn 8。

### 使用 conda 配置

对于 tensorflow 2.3.0：

```sh
conda create -n misaka-writer python=3.8
conda activate misaka-writer
conda install -c conda-forge pandas cudatoolkit=10.1 cudnn
pip install tensorflow==2.3.0 bert4keras jieba
```

对于 tensorflow 2.5.0：

```sh
conda create -n misaka-writer python=3.9
conda activate misaka-writer
conda install -c conda-forge pandas cudatoolkit=11.2 cudnn
pip install tensorflow==2.5.0 bert4keras jieba
```

## 使用方法

见 `main.py`。

![image](https://user-images.githubusercontent.com/62837036/169949572-b64ac754-e590-4cd3-bee5-08a597fa60b8.png)

`model_path` 是模型的权重路径，建议使用相对路径。

`num` 代表生成的下文的数量。 `text` 为输入，建议输入在20到250字之间。

**如果想写英文内容，修改 `load_model.py` 中 `get_writer_model` 函数的 `return` 语句，把 `start_token=4` 改为 `start_token=4`。**


## 训练语料

训练语料有100G中文和50G英文。

> 链接：https://pan.baidu.com/s/1WCiPA_tplI0AhdpDEuQ5ig <br/>
> 提取码：rlse  

## 预训练权重

### base/综合模型

> 链接：https://pan.baidu.com/s/1SdvL6W70np2qp9jDWbsGVQ <br/>
> 提取码：9sno


### 玄幻
> 链接：https://pan.baidu.com/s/1vGBJr6NOsWQAvJvxjqld-w <br/>
> 提取码：hszv

### 日轻
> 链接：https://pan.baidu.com/s/1n7vXu-1uLF6XKtizoJmQZg <br/>
> 提取码：miw0

如果受不了百度云的网速，建议加QQ群在群文件下载，更多类型目前现在Q群测试。

## 社区

如有问题可加Q群-143626394(大群，除了本项目还有 https://github.com/BlinkDL/AI-Writer 项目群）、905398734（本项目小群），本人qq 935499957

---

最后，misaka镇楼

![image](https://user-images.githubusercontent.com/62837036/170024801-1d10d8c5-266f-4ade-894c-67f30069f94f.png)
