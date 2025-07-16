import streamlit as st
import arxiv
import pandas as pd
import openai

# 设置 OpenAI API 密钥（从 secrets 中读取）
openai.api_key = st.secrets["api_key"]
client = openai.OpenAI()

# 页面设置
st.set_page_config(page_title="AI 文献分析 Agent", layout="wide")
st.title("📚 AI 文献分析 Agent")
st.markdown("请输入一个研究主题，我将抓取 arXiv 上的最新论文并使用 OpenAI GPT 进行分析总结。")

# 用户输入
query = st.text_input("🔍 搜索关键词（英文）", value="large language models in healthcare")

# 搜索按钮
if st.button("开始分析"):
    with st.spinner("正在搜索 arXiv 文献..."):
        search = arxiv.Search(
            query=query,
            max_results=10,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        client_arxiv = arxiv.Client()
        papers = []
        for result in client_arxiv.results(search):
            papers.append({
                "Title": result.title,
                "Authors": ", ".join([a.name for a in result.authors]),
                "Published": result.published.date(),
                "Summary": result.summary,
                "PDF": result.pdf_url
            })
        df = pd.DataFrame(papers)

    st.success(f"✅ 共找到 {len(df)} 篇文献")

    # 显示摘要
    with st.expander("📄 查看原始文献摘要"):
        st.dataframe(df)

    # 合并摘要
    combined_text = ""
    for i, row in df.iterrows():
        combined_text += f"Paper {i+1}:\nTitle: {row['Title']}\nSummary: {row['Summary']}\n\n"

    prompt = f"""
你是一个科研助手，请对以下10篇论文的摘要内容进行学术综述型分析，输出包括：

1. 按研究方向或技术路线对论文进行分组；
2. 每组的共同点总结；
3. 每篇论文的不同点（方法/模型/数据集等）；
4. 最终用 markdown 表格列出论文标题与分组标签。

以下是论文摘要内容：
{combined_text}
    """

    # GPT 分析
    with st.spinner("🤖 GPT 正在生成分析..."):
        try:
            chat_response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "你是一个擅长科研综述的学术助手。"},
                    {"role": "user", "content": prompt}
                ]
            )
            result_text = chat_response.choices[0].message.content
        except Exception as e:
            st.error(f"OpenAI 出错了：{e}")
            result_text = ""

    if result_text:
        st.markdown("### 🔍 GPT 总结与对比分析")
        st.markdown(result_text)

    # 下载原始数据
    csv = df.to_csv(index=False).encode('utf-8-sig')
    st.download_button("📥 下载文献数据 CSV", csv, "papers.csv", "text/csv")


