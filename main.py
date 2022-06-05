# -*- coding: utf-8 -*-
import sys
from generate import generate
from load_model import get_writer_model
import textwrap
import itertools
try:
    import tqdm
except ImportError:
    tqdm = None

if __name__ == "__main__":
    model_path = "models/综合.h5"  # 模型路径
    support_english = False  # 英文/中文模式
    nums = 3  # 开头生成下文的数量
    # 开头，建议开头字数在50字到200字之间
    text = """
    却说张飞带着几人与那鲁智深厮杀起来，那鲁智深使了个开碑裂石的招数，把那张飞一把丢在地上，自己落了地，往外便跑。
    到了门口，被他的随从一个劲的追杀。张飞心中一怒，便一刀砍在一名将军的腿上，那名将军疼的大叫着往前一滚，张飞一个转身，一脚便踢在他的胸口上，那人一下子滚到了台阶边，也是摔了个四脚朝天。
    张飞不敢恋战，赶紧带了几个人从后门离去，一路上还不断的大叫：
    “鲁智深，我与你不共戴天！”
    """
    output = "out.txt"  # 输出文件名

    # 加载模型
    generator = get_writer_model(model_path, support_english=support_english)
    # 生成
    text = textwrap.dedent(text)

    if tqdm is not None:
        with tqdm.tqdm(total=512 * (len(text) // 400 + 1) + 10) as progress_bar:
            progress_bar.set_description("Generating")
            outputs, time_consumed = generate(generator, text, nums, step_callback=lambda: progress_bar.update(1))    
            while progress_bar.n < progress_bar.total:  # type: ignore
                progress_bar.update(1)
    else:
        spin = itertools.cycle("|/-\\")
        outputs, time_consumed = generate(generator, text, nums, step_callback=lambda: sys.stderr.write("\rGenerating {}".format(next(spin))))
        sys.stderr.write("\r")
    sys.stderr.write(f"Finished in {time_consumed:.2f}s.\n")

    # 输出
    with open(output, "w", encoding="utf-8") as f:
        for _ in range(nums):
            f.write(textwrap.indent(text, "\t") + "\n")
            for output in outputs:                
                f.write(textwrap.indent(output, "\t") + "\n")
            f.write("\n" + "*" * 80 + "\n")
