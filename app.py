import os
import streamlit as st
from dotenv import load_dotenv
from core.document_processor import DocumentProcessor
from core.feedback_db import FeedbackDB
from core.rag_pipeline import RAGPipeline

load_dotenv()

st.set_page_config(
    page_title="DocuLens | Document Intelligence ",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

DOCULENS_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        background-color: #07090E;
        color: #E2E8F0;
    }

    .stApp {
        background: radial-gradient(circle at 50% -20%, rgba(99, 102, 241, 0.18), transparent 55%),
                    radial-gradient(circle at 85% 30%, rgba(168, 85, 247, 0.08), transparent 45%),
                    #07090E;
    }

    .glass-card {
        background: rgba(15, 23, 42, 0.65);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.36);
        margin-bottom: 1.5rem;
    }

    .citation-tile {
        background: rgba(15, 23, 42, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-left: 3px solid #6366F1;
        border-radius: 10px;
        padding: 1rem;
        margin-top: 0.75rem;
        margin-bottom: 0.75rem;
        font-size: 0.88rem;
        line-height: 1.5;
        color: #CBD5E1;
    }

    .badge-glow-high {
        background: rgba(16, 185, 129, 0.12);
        color: #34D399;
        border: 1px solid rgba(52, 211, 153, 0.3);
        padding: 4px 14px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-flex;
        box-shadow: 0 0 12px rgba(52, 211, 153, 0.15);
    }
    .badge-glow-mod {
        background: rgba(245, 158, 11, 0.12);
        color: #FBBF24;
        border: 1px solid rgba(251, 191, 36, 0.3);
        padding: 4px 14px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-flex;
        box-shadow: 0 0 12px rgba(245, 158, 11, 0.15);
    }
    .badge-glow-low {
        background: rgba(239, 68, 68, 0.12);
        color: #F87171;
        border: 1px solid rgba(248, 113, 113, 0.3);
        padding: 4px 14px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-flex;
        box-shadow: 0 0 12px rgba(239, 68, 68, 0.15);
    }

    .stChatMessage {
        background: rgba(15, 23, 42, 0.45) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 14px !important;
        padding: 1.2rem !important;
        backdrop-filter: blur(12px) !important;
        margin-bottom: 1rem !important;
    }

    div.stButton > button {
        background: rgba(30, 41, 59, 0.7) !important;
        color: #F8FAFC !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
        transition: all 0.2s ease !important;
    }
    div.stButton > button:hover {
        background: rgba(99, 102, 241, 0.2) !important;
        border-color: #6366F1 !important;
        box-shadow: 0 0 14px rgba(99, 102, 241, 0.3) !important;
        transform: translateY(-1px);
    }
</style>
"""
st.markdown(DOCULENS_CSS, unsafe_allow_html=True)

if "feedback_db" not in st.session_state:
    st.session_state.feedback_db = FeedbackDB()
if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = None
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0
if "digest" not in st.session_state:
    st.session_state.digest = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "active_prompt" not in st.session_state:
    st.session_state.active_prompt = None
if "generated_report" not in st.session_state:
    st.session_state.generated_report = None

with st.sidebar:
    st.markdown("### ⚡ **DocuLens**")
    st.caption("AI-Powered Document Intelligence")
    st.markdown("---")

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        groq_api_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")

    model_option = st.selectbox(
        "Groq LLM Model",
        options=["openai/gpt-oss-20b", "qwen/qwen3.8-27b", "openai/gpt-oss-120b"],
        index=0
    )

    st.markdown("---")
    st.markdown("#### **Corpus Upload**")
    uploaded_files = st.file_uploader(
        "Select PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    if st.button("⚡ Index to Persistent DB", use_container_width=True, type="primary"):
        if not groq_api_key:
            st.error("Please provide a Groq API Key.")
        elif not uploaded_files:
            st.warning("Upload at least one document.")
        else:
            with st.spinner("Extracting text, computing FastEmbed vectors & storing on disk..."):
                processor = DocumentProcessor(chunk_size=800, chunk_overlap=150)
                all_chunks = []
                file_names = []

                for file in uploaded_files:
                    try:
                        chunks = processor.process_file(file)
                        all_chunks.extend(chunks)
                        file_names.append(file.name)
                    except Exception as e:
                        st.error(f"Error parsing {file.name}: {e}")

                if all_chunks:
                    pipeline = RAGPipeline(groq_api_key=groq_api_key, model_name=model_option)
                    pipeline.initialize_index(all_chunks, reset=True)
                    digest = pipeline.generate_document_digest(all_chunks)

                    st.session_state.rag_pipeline = pipeline
                    st.session_state.indexed_files = file_names
                    st.session_state.chunk_count = len(all_chunks)
                    st.session_state.digest = digest
                    st.session_state.chat_history = []
                    st.session_state.generated_report = None
                    st.success("Indexed successfully into persistent database!")

    st.markdown("---")
    st.markdown("#### 📊 **RLHF Telemetry**")
    stats = st.session_state.feedback_db.get_stats()
    c1, c2, c3 = st.columns(3)
    c1.metric("Queries", stats["total"])
    c2.metric("👍", stats["positive"])
    c3.metric("👎", stats["negative"])

    if st.button("📥 Export DPO Dataset", use_container_width=True):
        count = st.session_state.feedback_db.export_dpo_dataset()
        if count > 0:
            st.success(f"Exported {count} pairs to `data/dpo_feedback.jsonl`!")
        else:
            st.info("No negative queries with corrections found yet.")

col_title, col_stat = st.columns([3, 1])
with col_title:
    st.markdown("# **DocuLens**")
    st.caption("Grounded Document Intelligence & Multi-Section Report Generation.")

with col_stat:
    if st.session_state.rag_pipeline:
        st.markdown(
            f"""
            <div style="text-align: right; padding-top: 10px;">
                <span style="color: #94A3B8; font-size: 0.8rem;">PERSISTENT DB STATUS</span><br>
                <span style="color: #6366F1; font-weight: 700; font-size: 1.1rem;">● PERSISTED</span> 
                <span style="color: #E2E8F0; font-size: 0.9rem;">({len(st.session_state.indexed_files)} Docs / {st.session_state.chunk_count} Chunks)</span>
            </div>
            """,
            unsafe_allow_html=True
        )

if not st.session_state.rag_pipeline:
    st.markdown(
        """
        <div class="glass-card" style="text-align: center; padding: 3rem 1.5rem; margin-top: 2rem;">
            <h3 style="color: #F8FAFC; margin-bottom: 0.5rem;">Knowledge Base Empty</h3>
            <p style="color: #94A3B8; max-width: 480px; margin: 0 auto;">
                Provide your Groq API Key, upload your documents in the sidebar, and click <b>Index to Persistent DB</b> to begin.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.stop()

tab_qa, tab_report = st.tabs(["💬 Interactive Q&A", "📑 Intelligent Report Generator"])

with tab_qa:
    if st.session_state.digest:
        st.markdown(
            f"""
            <div class="glass-card">
                <div style="font-weight: 700; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; color: #818CF8; margin-bottom: 0.4rem;">
                    ✦ Context Snapshot
                </div>
                <div style="font-size: 0.92rem; color: #CBD5E1; line-height: 1.6;">
                    {st.session_state.digest['summary']}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("##### **Suggested Inquiries**")
        chip_cols = st.columns(len(st.session_state.digest["suggested_questions"]))
        for i, question in enumerate(st.session_state.digest["suggested_questions"]):
            if chip_cols[i].button(f"✨ {question}", key=f"chip_{i}", use_container_width=True):
                st.session_state.active_prompt = question

    for item in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(item["question"])

        with st.chat_message("assistant"):
            st.markdown(item["answer"])

            if item.get("matched_memory"):
                st.markdown(
                    f'<span style="background: rgba(99, 102, 241, 0.15); color: #818CF8; border: 1px solid rgba(99, 102, 241, 0.3); padding: 3px 10px; border-radius: 9999px; font-size: 0.72rem; font-weight: 600;">🧠 Semantic Memory Match: "{item["matched_memory"][0]["past_question"]}"</span>',
                    unsafe_allow_html=True
                )

            conf = item["confidence"]
            badge_style = "badge-glow-high" if conf == "High" else "badge-glow-mod" if conf == "Moderate" else "badge-glow-low"
            st.markdown(f'<div style="margin-top: 0.6rem;"><span class="{badge_style}">✦ Grounding Confidence: {conf}</span></div>', unsafe_allow_html=True)

            if item["sources"]:
                with st.expander("🔍 Verified Document Citations"):
                    for src in item["sources"]:
                        st.markdown(
                            f"""
                            <div class="citation-tile">
                                <strong style="color: #F8FAFC;">📄 {src['source']}</strong> 
                                <span style="color: #94A3B8;">&nbsp;•&nbsp; Page {src['page']} &nbsp;•&nbsp; Distance: {src['distance']}</span><br>
                                <div style="margin-top: 0.35rem; color: #94A3B8;">{src['content']}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

            col_fb1, col_fb2, col_fb3 = st.columns([1, 1, 8])
            if col_fb1.button("👍", key=f"up_{item['db_id']}"):
                st.session_state.feedback_db.record_feedback(item["db_id"], rating=1)
                st.toast("Approved as golden response!")
                st.rerun()

            if col_fb2.button("👎", key=f"down_{item['db_id']}"):
                st.session_state.feedback_db.record_feedback(item["db_id"], rating=-1)
                st.session_state[f"correcting_{item['db_id']}"] = True
                st.rerun()

            if st.session_state.get(f"correcting_{item['db_id']}", False):
                with st.form(key=f"correction_form_{item['db_id']}"):
                    correction_text = st.text_input(
                        "Provide the preferred answer (updates semantic memory):",
                        key=f"text_corr_{item['db_id']}"
                    )
                    submit_corr = st.form_submit_button("Save Correction")
                    if submit_corr and correction_text:
                        st.session_state.feedback_db.record_feedback(item["db_id"], rating=-1, correction=correction_text)
                        st.session_state[f"correcting_{item['db_id']}"] = False
                        st.success("Correction saved to semantic memory!")
                        st.rerun()

    query_input = st.chat_input("Ask a question about the uploaded documents...")
    resolved_query = query_input or st.session_state.active_prompt

    if resolved_query:
        st.session_state.active_prompt = None

        with st.chat_message("user"):
            st.write(resolved_query)

        with st.chat_message("assistant"):
            with st.spinner("Searching semantic query memory & running hybrid retrieval..."):
                semantic_matches = st.session_state.feedback_db.search_similar_queries(
                    incoming_question=resolved_query,
                    k=2,
                    threshold=0.80
                )

                result = st.session_state.rag_pipeline.query(
                    question=resolved_query, 
                    semantic_exemplars=semantic_matches
                )

                db_id = st.session_state.feedback_db.log_query(
                    question=resolved_query,
                    answer=result["answer"],
                    confidence=result["confidence"],
                    sources=result["sources"]
                )

                st.markdown(result["answer"])

                if result.get("matched_memory"):
                    st.markdown(
                        f'<span style="background: rgba(99, 102, 241, 0.15); color: #818CF8; border: 1px solid rgba(99, 102, 241, 0.3); padding: 3px 10px; border-radius: 9999px; font-size: 0.72rem; font-weight: 600;">🧠 Semantic Memory Match: "{result["matched_memory"][0]["past_question"]}"</span>',
                        unsafe_allow_html=True
                    )

                conf = result["confidence"]
                badge_style = "badge-glow-high" if conf == "High" else "badge-glow-mod" if conf == "Moderate" else "badge-glow-low"
                st.markdown(f'<div style="margin-top: 0.6rem;"><span class="{badge_style}">✦ Grounding Confidence: {conf}</span></div>', unsafe_allow_html=True)

                if result["sources"]:
                    with st.expander("🔍 Verified Document Citations"):
                        for src in result["sources"]:
                            st.markdown(
                                f"""
                                <div class="citation-tile">
                                    <strong style="color: #F8FAFC;">📄 {src['source']}</strong> 
                                    <span style="color: #94A3B8;">&nbsp;•&nbsp; Page {src['page']} &nbsp;•&nbsp; Distance: {src['distance']}</span><br>
                                    <div style="margin-top: 0.35rem; color: #94A3B8;">{src['content']}</div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

        st.session_state.chat_history.append({
            "db_id": db_id,
            "question": resolved_query,
            "answer": result["answer"],
            "confidence": result["confidence"],
            "sources": result["sources"],
            "matched_memory": result.get("matched_memory", [])
        })
        st.rerun()

with tab_report:
    st.markdown("### 📑 **Natural Language Executive Report Generator**")
    st.caption("Provide a high-level instruction or objective. DocuLens will scan the persistent corpus and compile a formal, multi-section whitepaper report.")

    col_r1, col_r2 = st.columns([3, 1])
    with col_r1:
        report_nl_prompt = st.text_area(
            "Enter Natural Language Report Directive:",
            placeholder="e.g. Generate a comprehensive security audit report detailing all cryptographic techniques, threat models, and student examination priorities discussed across the document.",
            height=90
        )
    with col_r2:
        report_profile = st.selectbox(
            "Report Profile / Structure",
            options=["Executive Briefing", "Compliance & Audit Report", "Technical Deep-Dive", "Comparative Summary"],
            index=0
        )
        generate_btn = st.button("🚀 Compile Full Report", use_container_width=True, type="primary")

    if generate_btn:
        if not report_nl_prompt.strip():
            st.warning("Please type a report directive or prompt.")
        else:
            with st.spinner("Executing multi-hop retrieval across corpus & synthesizing formal report..."):
                report_data = st.session_state.rag_pipeline.generate_report(
                    report_prompt=report_nl_prompt,
                    report_type=report_profile,
                    k=12
                )
                st.session_state.generated_report = {
                    "directive": report_nl_prompt,
                    "profile": report_profile,
                    "content": report_data["report"],
                    "sources": report_data["sources"],
                    "source_count": report_data["source_count"]
                }

    if st.session_state.generated_report:
        rep = st.session_state.generated_report
        st.markdown("---")

        d_col1, d_col2 = st.columns([3, 1])
        with d_col1:
            st.markdown(f"#### 📄 **Synthesized Report** (Synthesized from {rep['source_count']} contextual excerpts)")
        with d_col2:
            st.download_button(
                label="📥 Download Report (.md)",
                data=rep["content"],
                file_name="DocuLens_Executive_Report.md",
                mime="text/markdown",
                use_container_width=True
            )

        with st.container(border=True):
            st.markdown(rep["content"])

        with st.expander("🔍 Inspect Underlying Evidentiary Passages"):
            for s_idx, s in enumerate(rep["sources"]):
                st.markdown(f"**Excerpt {s_idx + 1}** — `{s['source']}` (Page {s['page']})")
                st.caption(s["content"])