import streamlit as st
import pandas as pd
from dataframe_utils import dataframe_agent
from chart_generator import create_chart


def render_data_analysis():
    """渲染数据分析模块"""
    st.header("📊 智能数据分析")

    if "df" not in st.session_state:
        st.info("请先在侧边栏上传数据文件")
        return

    st.write("### 数据预览")
    st.dataframe(st.session_state.data_df.head(10), use_container_width=True)

    analysis_query = st.text_area(
        "输入分析需求（示例：显示各月销售额趋势）",
        height=100
    )

    if st.button("生成回答"):
        with st.spinner("AI正在思考中，请稍等..."):
            result = dataframe_agent(st.session_state.data_df, analysis_query)

        if "answer" in result:
            st.write(result["answer"])

        if "table" in result:
            st.table(
                pd.DataFrame(
                    result["table"]["data"],
                    columns=result["table"]["columns"]
                )
            )

        if "bar" in result:
            create_chart(result["bar"], "bar")

        if "line" in result:
            create_chart(result["line"], "line")