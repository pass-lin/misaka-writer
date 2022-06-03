import streamlit as st
from load_model import list_models, get_writer_model
from generate import generate
import time

if "current_model_path" not in st.session_state:
    st.session_state["current_model_path"] = None
if "current_model" not in st.session_state:
    st.session_state["current_model"] = None
if "current_support_english" not in st.session_state:
    st.session_state["current_support_english"] = False

model_path = st.sidebar.selectbox("选择模型：", list(list_models()))
support_english = st.sidebar.checkbox(
    "英文模式", value=st.session_state["current_support_english"]
)
model = st.session_state["current_model"]

st.sidebar.markdown("---")

max_len = int(
    st.sidebar.number_input("续写最大长度：", min_value=50, max_value=512, value=512)
)
nums = int(st.sidebar.number_input("生成下文的数量：", min_value=1, max_value=5, value=3))

text = st.text_area(
    "输入开头（建议50~250字）：",
    """
却说张飞带着几人与那鲁智深厮杀起来，那鲁智深使了个开碑裂石的招数，把那张飞一把丢在地上，自己落了地，往外便跑。
到了门口，被他的随从一个劲的追杀。张飞心中一怒，便一刀砍在一名将军的腿上，那名将军疼的大叫着往前一滚，张飞一个转身，一脚便踢在他的胸口上，那人一下子滚到了台阶边，也是摔了个四脚朝天。
张飞不敢恋战，赶紧带了几个人从后门离去，一路上还不断的大叫：
“鲁智深，我与你不共戴天！”
""".strip(),
    height=250,
)

if model_path and (
    model_path != st.session_state["current_model_path"]
    or model is None
    or support_english != st.session_state["current_support_english"]
):
    st.session_state["current_model_path"] = model_path
    st.session_state["current_support_english"] = support_english
    with st.spinner("加载模型中..."):
        model = get_writer_model(model_path, support_english)
        st.session_state["current_model"] = model


class ProgressBar:
    def __init__(self, total):
        self._pbar = st.progress(0)
        self._total = total
        self._count = 0

    def update(self, count=1):
        self._count += count
        self._pbar.progress(int(self._count / self._total * 100))

    def finish(self):
        while self._count < self._total:
            time.sleep(0.01)
            self.update(1)
        self._pbar.progress(100)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()


if model and st.button("开始生成"):
    with ProgressBar(total=max_len * (len(text) // 400 + 1) + 10) as pbar:
        outputs, time_consumed = generate(
            model, text, max_len=max_len, nums=nums, step_callback=pbar.update
        )
        st.success(f"生成完成！耗时：{time_consumed}s")
        for i, output in enumerate(outputs):
            st.subheader(f"续写{i + 1}")
            st.write(output.replace("\n", "\n\n"))
