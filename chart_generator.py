import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


def create_chart(input_data, chart_type):
    """
    生成统计图表

    Args:
        input_data: 输入数据，字典格式，包含 "columns" 和 "data" 两个键
        chart_type: 图表类型，支持 "bar"（柱状图）和 "line"（折线图）
    """
    df_data = pd.DataFrame(
        data={
            "x": input_data["columns"],
            "y": input_data["data"]
        }
    )
    df_data.set_index("x", inplace=True)

    # 根据数据内容生成标题
    title = ""
    if chart_type == "bar":
        title = "柱状图分析 - " + ", ".join(input_data["columns"])
    elif chart_type == "line":
        title = "折线图分析 - " + ", ".join(input_data["columns"])

    if chart_type == "bar":
        st.bar_chart(df_data)
    elif chart_type == "line":
        plt.plot(df_data.index, df_data["y"], marker="o", linestyle="--")
        plt.ylim(0, df_data["y"].max() * 1.1)
        plt.title(title)
        st.pyplot(plt.gcf())