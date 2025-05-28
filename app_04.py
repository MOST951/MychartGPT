import io
import uuid
import pickle
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain, ConversationalRetrievalChain
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

# 初始化环境变量
load_dotenv()
plt.style.use('ggplot')


# --------------------------
# 通用工具函数
# --------------------------
def init_session_state():
    """初始化所有会话状态"""
    if 'current_mode' not in st.session_state:
        st.session_state.current_mode = "💬 智能聊天"

    # 聊天模块状态
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = [{'role': 'ai', 'content': '你好！我是您的智能助手，请问有什么可以帮您？'}]
    if 'chat_memory' not in st.session_state:
        st.session_state.chat_memory = ConversationBufferMemory(return_messages=True)

    # 文档问答模块状态
    if 'rag_memory' not in st.session_state:
        st.session_state.rag_memory = ConversationBufferMemory(
            return_messages=True,
            memory_key='chat_history',
            output_key='answer'
        )
    if 'session_id' not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex
    if 'rag_db' not in st.session_state:
        st.session_state.rag_db = None
    if 'is_new_file' not in st.session_state:
        st.session_state.is_new_file = True

    # 数据分析模块状态
    if 'data_memory' not in st.session_state:
        st.session_state.data_memory = ConversationBufferMemory(return_messages=True)
    if 'data_df' not in st.session_state:
        st.session_state.data_df = None


def get_ai_response(memory, user_prompt, system_prompt=""):
    """通用AI响应生成函数"""
    try:
        model = ChatOpenAI(
            model=st.session_state.selected_model,
            api_key=st.secrets['API_KEY'],
            base_url='https://api.openai.com/v1',
            temperature=st.session_state.model_temperature,
            max_tokens=st.session_state.model_max_length
        )
        chain = ConversationChain(llm=model, memory=memory)
        full_prompt = f"System: {system_prompt}\nUser: {user_prompt}"
        response = chain.invoke({'input': full_prompt})['response']
        return response
    except Exception as e:
        st.error(f"AI请求失败，请检查您的网络连接或API密钥配置。错误信息：{str(e)}")
        return "无法生成响应，请检查配置"


# --------------------------
# 侧边栏功能
# --------------------------
def render_sidebar():
    with st.sidebar:
        st.title("⚙️ 控制面板")

        # 模型配置
        st.subheader("模型设置")
        st.session_state.selected_model = st.selectbox(
            "AI模型",
            ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo", "gpt-4"],
            index=0,
            help="选择要使用的AI模型"
        )
        st.session_state.model_temperature = st.slider(
            "温度值",
            0.0, 1.0, 0.7, 0.1,
            help="控制回答的创造性（0=保守，1=创新）"
        )
        st.session_state.model_max_length = st.slider(
            "最大长度",
            100, 2000, 1000,
            help="控制回答的最大长度"
        )
        st.session_state.system_prompt = st.text_area(
            "系统提示词",
            "你是一个专业的人工智能助手，用中文简洁准确地回答问题",
            height=100
        )

        # 设置环境变量
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

        # 文件上传
        st.subheader("数据管理")
        uploaded_file = st.file_uploader(
            "上传文件（支持CSV/Excel/TXT）",
            type=["csv", "xlsx", "txt"],
            help="根据当前模式自动处理文件类型"
        )
        handle_uploaded_file(uploaded_file)

        # 历史消息
        st.subheader("对话历史")
        render_history()

        if st.button("🗑️ 清空当前历史"):
            clear_current_history()


def handle_uploaded_file(file):
    if not file:
        return

    try:
        file_type = file.name.split('.')[-1]

        # 处理数据文件
        if file_type in ['csv', 'xlsx']:
            df = pd.read_csv(file) if file_type == 'csv' else pd.read_excel(file)
            st.session_state.data_df = df
            st.success("数据加载成功！")
            st.dataframe(df.head(8), use_container_width=True, height=300)

        # 处理文本文件
        elif file_type == 'txt':
            if st.session_state.current_mode == "📚 文档问答":
                st.session_state.is_new_file = True
                with open(f'{st.session_state.session_id}.txt', 'w', encoding='utf-8') as f:
                    f.write(file.read().decode('utf-8'))
                st.success("文档上传成功！")
            else:
                st.session_state.txt_content = file.read().decode('utf-8')
                st.text_area("文本内容", st.session_state.txt_content, height=300)

    except Exception as e:
        st.error(f"文件处理失败：{str(e)}")


def render_history():
    """渲染历史记录，并在失败时提供用户友好的提示"""
    try:
        current_history = []
        if st.session_state.current_mode == "💬 智能聊天":
            # 获取聊天模式下的历史记录
            current_history = [msg["content"] for msg in st.session_state.chat_messages if msg["role"] == "human"]
        elif st.session_state.current_mode == "📚 文档问答":
            # 获取文档问答模式下的历史记录
            rag_memory = st.session_state.get('rag_memory')
            if rag_memory is not None:
                chat_history = rag_memory.load_memory_variables({}).get('chat_history', [])
                current_history = [msg.content for msg in chat_history if isinstance(msg, HumanMessage)]
            else:
                current_history = []

        if current_history:
            # 显示历史记录选择框
            selected = st.selectbox(
                "选择历史记录",
                options=[msg[:30] + "..." for msg in current_history],
                key="history_select"
            )
            if selected:
                st.write(f"**选中记录**: {selected}")
        else:
            st.write("暂无历史记录")
    except Exception as e:
        # 捕获异常并显示友好提示
        st.error(f"加载历史记录时遇到问题：{str(e)}。请尝试刷新页面或重新启动应用。")
        st.caption("提示：如果问题持续存在，请检查应用日志以获取更多详细信息。")


def clear_current_history():
    try:
        if st.session_state.current_mode == "💬 智能聊天":
            st.session_state.chat_messages = [{'role': 'ai', 'content': '你好！我是您的智能助手，请问有什么可以帮您？'}]
            st.session_state.chat_memory.clear()
        elif st.session_state.current_mode == "📚 文档问答":
            st.session_state.rag_memory.clear()
        elif st.session_state.current_mode == "📊 数据分析":
            st.session_state.data_memory.clear()
        st.success("当前模式历史记录已清除！")
    except Exception as e:
        st.error(f"清空历史失败：{str(e)}")


# --------------------------
# 主功能模块
# --------------------------
def main():
    init_session_state()

    # 页面标题
    st.markdown("""
        <div style="text-align:center; margin-bottom:40px">
            <h1 style="margin-bottom:0">SuperAI 智能分析助手🚀</h1>
            <p style="color:#6C63FF; font-size:1.2rem">数据洞察从未如此简单</p>
        </div>
    """, unsafe_allow_html=True)

    # 渲染侧边栏
    render_sidebar()

    # 模式选择
    st.session_state.current_mode = st.sidebar.radio(
        "功能导航",
        ["💬 智能聊天", "📚 文档问答", "📊 数据分析"],
        horizontal=True,
        label_visibility="collapsed"
    )

    # 路由到对应模块
    if st.session_state.current_mode == "💬 智能聊天":
        render_chat()
    elif st.session_state.current_mode == "📚 文档问答":
        render_document_qa()
    elif st.session_state.current_mode == "📊 数据分析":
        render_data_analysis()


def render_chat():
    st.header("💬 智能聊天")

    # 显示消息历史
    for msg in st.session_state.chat_messages:
        role = "user" if msg["role"] == "human" else "assistant"
        with st.chat_message(role):
            st.write(msg["content"])

    # 处理用户输入
    if prompt := st.chat_input("请输入您的问题..."):
        handle_chat_input(prompt)


def handle_chat_input(prompt):
    # 添加用户消息
    st.session_state.chat_messages.append({'role': 'human', 'content': prompt})

    # 获取AI响应
    with st.spinner('思考中...'):
        response = get_ai_response(
            memory=st.session_state.chat_memory,
            user_prompt=prompt,
            system_prompt=st.session_state.system_prompt
        )

    # 添加AI响应
    st.session_state.chat_messages.append({'role': 'ai', 'content': response})
    st.rerun()


def render_document_qa():
    st.header("📚 文档智能问答")

    # 问题输入
    question = st.text_input("请输入您的问题", key="doc_question")

    if question and st.session_state.get('session_id'):
        with st.spinner('正在分析文档...'):
            try:
                if st.session_state.is_new_file:
                    loader = TextLoader(f'{st.session_state.session_id}.txt', encoding='utf-8')
                    docs = loader.load()

                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200,
                        separators=["\n\n", "\n", "。", "！", "？", "，"]
                    )
                    texts = text_splitter.split_documents(docs)

                    st.session_state.rag_db = FAISS.from_documents(
                        texts,
                        embedding=None  # 使用默认的embedding
                    )
                    st.session_state.is_new_file = False

                model = ChatOpenAI(
                    model=st.session_state.selected_model,
                    api_key=st.secrets['API_KEY'],
                    base_url='https://api.openai.com/v1',
                    temperature=0.2
                )

                qa_chain = ConversationalRetrievalChain.from_llm(
                    llm=model,
                    retriever=st.session_state.rag_db.as_retriever(
                        search_type="mmr",
                        search_kwargs={'k': 3}
                    ),
                    memory=st.session_state.rag_memory,
                    return_source_documents=True
                )

                result = qa_chain({'question': question})

                st.subheader("答案")
                st.write(result['answer'])

                with st.expander("参考文档片段"):
                    for doc in result['source_documents']:
                        st.markdown(f"**来源**: `{doc.metadata['source']}`")
                        st.text(doc.page_content[:300] + "...")
                        st.divider()

            except Exception as e:
                st.error(f"文档处理失败：{str(e)}")


def render_data_analysis():
    st.header("📊 智能数据分析")

    if st.session_state.data_df is not None:
        st.write("### 数据预览")
        st.dataframe(st.session_state.data_df.head(10), use_container_width=True)

        analysis_query = st.text_input("输入分析需求（示例：显示各月销售额趋势）")

        if analysis_query:
            with st.spinner('正在生成分析...'):
                try:
                    data_desc = f"""
                    数据集列名：{st.session_state.data_df.columns.tolist()}
                    数据类型：
                    {st.session_state.data_df.dtypes.to_string()}
                    数据样例：
                    {st.session_state.data_df.head(3).to_markdown()}
                    """

                    analysis_result = get_ai_response(
                        memory=st.session_state.data_memory,
                        user_prompt=f"分析需求：{analysis_query}\n{data_desc}",
                        system_prompt="你是一个数据分析专家，请根据提供的数据集进行专业分析，给出可视化建议"
                    )

                    st.write("### AI分析报告")
                    st.write(analysis_result)

                    # 自动生成图表
                    if "柱状图" in analysis_result:
                        st.bar_chart(st.session_state.data_df.select_dtypes(include='number'))
                    elif "折线图" in analysis_result:
                        st.line_chart(st.session_state.data_df.select_dtypes(include='number'))
                    elif "散点图" in analysis_result:
                        st.scatter_chart(st.session_state.data_df)

                except Exception as e:
                    st.error(f"分析失败：{str(e)}")
    else:
        st.info("请先在侧边栏上传数据文件")


if __name__ == "__main__":
    main()