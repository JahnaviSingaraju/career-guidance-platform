# app.py — COMPLETE STANDALONE FILE — ReAct Agent + LangGraph + RAG + Streamlit UI
import os
import glob
import streamlit as st
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from PyPDF2 import PdfReader
from typing import List, Dict, Any, TypedDict, Optional
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME   = "llama-3.1-8b-instant"
DOCS_FOLDER  = "career_docs"

# LANGGRAPH STATE
class AgentState(TypedDict):
    user_query:     str
    query_type:     Optional[str]
    branch:         Optional[str]
    retrieved_docs: Optional[List[str]]
    reranked_docs:  Optional[List[str]]
    resume_context: Optional[List[str]]
    final_answer:   Optional[str]
    has_resume:     bool

# RAG PIPELINE WITH LANGGRAPH + ReAct AGENT

class CareerGuidanceRAG:

    def __init__(self):
        os.environ["GROQ_API_KEY"] = GROQ_API_KEY
        self.llm = ChatGroq(temperature=0, model_name=MODEL_NAME)
        self.chroma_client     = chromadb.EphemeralClient()
        self.collection        = self.chroma_client.get_or_create_collection("career_knowledge_base")
        self.resume_collection = self.chroma_client.get_or_create_collection("student_resume")
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

        # Build LangGraph pipeline
        self.graph = self._build_graph()

        # Build ReAct Agent with tools
        self.react_agent = self._build_react_agent()

    # ─────────────────────────────────────────
    # REACT AGENT
    # ─────────────────────────────────────────

    def _build_react_agent(self):
        """
        ReAct Agent with 4 tools:
          - search_career_knowledge: search career docs
          - get_resume_info: fetch student resume context
          - classify_question: detect query type
          - get_final_advice: generate final answer
        """

        rag_ref = self  # reference to self for use inside tools

        @tool
        def search_career_knowledge(query: str) -> str:
            """Search the career knowledge base for relevant information about
            careers, jobs, internships, skills, companies, salary, GATE, MS abroad."""
            total = rag_ref.collection.count()
            if total == 0:
                return "Knowledge base is empty. Load career documents first."
            results = rag_ref.collection.query(
                query_texts=[query],
                n_results=min(5, total)
            )
            docs = results["documents"][0]
            return "\n\n---\n\n".join(docs[:3])

        @tool
        def get_resume_info(query: str) -> str:
            """Retrieve relevant information from the student's uploaded resume.
            Use this when the question is about the student's profile, skills, or background."""
            count = rag_ref.resume_collection.count()
            if count == 0:
                return "No resume uploaded by the student."
            results = rag_ref.resume_collection.query(
                query_texts=[query],
                n_results=min(3, count)
            )
            return "\n\n".join(results["documents"][0])

        @tool
        def classify_question(query: str) -> str:
            """Classify the student's question into one of these types:
            career, internship, skills, resume, higher_education, general.
            Returns the query type as a string."""
            q = query.lower()
            if any(w in q for w in ["internship", "intern", "drdo", "isro", "stipend"]):
                return "internship"
            elif any(w in q for w in ["skill", "learn", "course", "certification", "roadmap"]):
                return "skills"
            elif any(w in q for w in ["resume", "cv", "my profile", "my background"]):
                return "resume"
            elif any(w in q for w in ["gate", "mtech", "ms abroad", "mba", "phd"]):
                return "higher_education"
            elif any(w in q for w in ["job", "career", "salary", "company", "placement"]):
                return "career"
            return "general"

        @tool
        def get_final_advice(context: str) -> str:
            """Use this tool LAST to format and return the final career advice
            after gathering all necessary context from other tools."""
            return context

        tools = [
            search_career_knowledge,
            get_resume_info,
            classify_question,
            get_final_advice,
        ]

        return create_react_agent(self.llm, tools)

    # ─────────────────────────────────────────
    # LANGGRAPH PIPELINE
    # ─────────────────────────────────────────

    def _build_graph(self):
        builder = StateGraph(AgentState)
        builder.add_node("classify_query", self._node_classify)
        builder.add_node("retrieve",       self._node_retrieve)
        builder.add_node("rerank",         self._node_rerank)
        builder.add_node("personalize",    self._node_personalize)
        builder.add_node("generate",       self._node_generate)
        builder.set_entry_point("classify_query")
        builder.add_edge("classify_query", "retrieve")
        builder.add_edge("retrieve",       "rerank")
        builder.add_edge("rerank",         "personalize")
        builder.add_edge("personalize",    "generate")
        builder.add_edge("generate",       END)
        return builder.compile()

    # ── Node 1: Classify ──────────────────────

    def _node_classify(self, state: AgentState) -> AgentState:
        query = state["user_query"].lower()

        # Detect branch
        branch_map = {
            "ece":        ["ece", "electronics", "communication", "vlsi", "embedded", "iot", "signal"],
            "cse":        ["cse", "computer science", "software", "programming", "coding"],
            "mechanical": ["mechanical", "mech", "automobile", "manufacturing", "thermal", "aerospace"],
            "civil":      ["civil", "structural", "construction", "urban", "transportation"],
            "electrical": ["electrical", "eee", "power systems", "vlsi", "automation"],
            "chemical":   ["chemical", "petroleum", "process", "pharma"],
            "it":         ["it", "information technology"],
        }
        detected_branch = None
        for branch, keywords in branch_map.items():
            if any(k in query for k in keywords):
                detected_branch = branch
                break

        # Detect query type
        if any(w in query for w in ["internship", "intern", "drdo", "isro", "stipend", "internshala"]):
            query_type = "internship"
        elif any(w in query for w in ["skill", "learn", "course", "certification", "roadmap", "how to become"]):
            query_type = "skills"
        elif any(w in query for w in ["resume", "cv", "my profile", "my background", "my skills"]):
            query_type = "resume"
        elif any(w in query for w in ["gate", "mtech", "ms abroad", "mba", "higher study", "phd"]):
            query_type = "higher_education"
        elif any(w in query for w in ["job", "career", "salary", "company", "placement", "package"]):
            query_type = "career"
        else:
            query_type = "general"

        return {**state, "query_type": query_type, "branch": detected_branch}

    # ── Node 2: Retrieve ──────────────────────

    def _node_retrieve(self, state: AgentState) -> AgentState:
        query  = state["user_query"]
        branch = state.get("branch")
        total  = self.collection.count()
        if total == 0:
            return {**state, "retrieved_docs": []}

        # Enrich query with branch name so vector search targets right docs
        enriched_query = f"{branch} {query}" if branch else query

        n           = min(10, total)
        results     = self.collection.query(query_texts=[enriched_query], n_results=n)
        vector_docs = results["documents"][0]

        # Branch-specific keyword filter — only keep docs mentioning the branch
        branch_keywords = {
            "ece":        ["ece", "electronics", "vlsi", "embedded", "iot", "microcontroller", "signal", "rf", "robotics"],
            "cse":        ["cse", "software", "computer science", "python", "data science", "web", "cloud"],
            "mechanical": ["mechanical", "automobile", "manufacturing", "cad", "thermal", "aerospace", "mech"],
            "civil":      ["civil", "structural", "construction", "autocad", "urban", "staad"],
            "electrical": ["electrical", "power", "eee", "plc", "automation", "scada"],
            "chemical":   ["chemical", "process", "petroleum", "pharma", "refinery"],
            "it":         ["information technology", "it ", "software"],
        }

        if branch and branch in branch_keywords:
            bkeys = branch_keywords[branch]
            # Priority: docs that mention the branch first
            branch_docs  = [d for d in vector_docs if any(k in d.lower() for k in bkeys)]
            other_docs   = [d for d in vector_docs if d not in branch_docs]
            hybrid = list(dict.fromkeys(branch_docs + other_docs))[:10]
            # If no branch-specific docs found, fall back to general keyword filter
            if not branch_docs:
                keywords = [k for k in query.lower().split() if len(k) > 3]
                keyword_docs = [d for d in vector_docs if any(k in d.lower() for k in keywords)]
                hybrid = list(dict.fromkeys(keyword_docs + vector_docs))[:10]
        else:
            keywords     = [k for k in query.lower().split() if len(k) > 3]
            keyword_docs = [d for d in vector_docs if any(k in d.lower() for k in keywords)]
            hybrid = list(dict.fromkeys(keyword_docs + vector_docs))[:10]

        return {**state, "retrieved_docs": hybrid}

    # ── Node 3: Rerank ────────────────────────

    def _node_rerank(self, state: AgentState) -> AgentState:
        docs  = state.get("retrieved_docs", [])
        query = state["user_query"]
        if not docs:
            return {**state, "reranked_docs": []}
        prompt   = PromptTemplate.from_template(
            "Query: {query}\n\nDocuments (numbered):\n{docs}\n\n"
            "Return the numbers of the top 3 most relevant documents "
            "for answering this career question, comma separated. Example: 1,3,5"
        )
        numbered = "\n".join([f"{i+1}. {d[:200]}" for i, d in enumerate(docs)])
        try:
            result   = (prompt | self.llm).invoke({"query": query, "docs": numbered})
            indices  = [int(x.strip()) - 1 for x in result.content.split(",") if x.strip().isdigit()]
            reranked = [docs[i] for i in indices if 0 <= i < len(docs)]
            if not reranked:
                reranked = docs[:3]
        except Exception:
            reranked = docs[:3]
        return {**state, "reranked_docs": reranked}

    # ── Node 4: Personalize ───────────────────

    def _node_personalize(self, state: AgentState) -> AgentState:
        query        = state["user_query"]
        resume_count = self.resume_collection.count()
        if resume_count == 0:
            return {**state, "resume_context": [], "has_resume": False}
        results = self.resume_collection.query(
            query_texts=[query], n_results=min(3, resume_count)
        )
        return {**state, "resume_context": results["documents"][0], "has_resume": True}

    # ── Node 5: Generate ──────────────────────

    # ── Node 5: Generate ──────────────────────

    def _node_generate(self, state: AgentState) -> AgentState:
        query      = state["user_query"]
        reranked   = state.get("reranked_docs", [])
        context    = "\n\n---\n\n".join(reranked)
        query_type = state.get("query_type", "general")
        has_resume = state.get("has_resume", False)
        resume_ctx = state.get("resume_context", [])

        if not reranked or not context.strip():
            return {**state, "final_answer": "❌ This question is not related to career guidance."}

        query_keywords = [k for k in query.lower().split() if len(k) > 3]
        relevance_score = sum(1 for k in query_keywords if k in context.lower())
        career_keywords = [
            "career", "job", "intern", "skill", "course", "engineer", "salary",
            "company", "placement", "gate", "mtech", "resume", "learn", "work",
            "software", "data", "cloud", "mechanical", "civil", "ece", "cse",
            "electrical", "chemical", "degree", "college", "certification", "tech",
            "python", "project", "interview", "hire", "role", "domain", "field",
            "opportunity", "roadmap", "study", "exam", "higher", "abroad", "mba",
            "path", "future", "improve", "tips", "advice", "best", "top",
            "which", "recommend", "suggest", "prepare", "apply", "how to", "what are"
        ]
        career_score = sum(1 for k in career_keywords if k in query.lower())

        if relevance_score == 0 and career_score == 0:
            return {**state, "final_answer": "❌ This question is not related to career guidance."}

        if has_resume:
            resume_text  = " ".join(resume_ctx).lower()
            query_words  = [k for k in query.lower().split() if len(k) > 3]
            resume_match = sum(1 for w in query_words if w in resume_text)
            if career_score == 0 and resume_match == 0:
                return {**state, "final_answer": "❌ This question is not related to career guidance."}

            prompt = PromptTemplate.from_template(
                "You are an expert AI Career Guidance Counselor for engineering students in India.\n\n"
                "Student Profile (from resume):\n{resume}\n\n"
                "Career Knowledge Base:\n{context}\n\n"
                "The student asked: {query}\n"
                "Query type detected: {query_type}\n\n"
                "STRICT RULE: If the question is NOT related to careers, jobs, internships, skills, "
                "education, or the student profile — reply with exactly: NOT_CAREER_RELATED\n\n"
                "If the question IS career-related, answer with:\n"
                "1. Direct answer to their question\n"
                "2. Personalized recommendation based on their profile\n"
                "3. Specific next steps for THIS student\n"
            )
            advice = (prompt | self.llm).invoke({
                "resume":     "\n\n".join(resume_ctx),
                "context":    context,
                "query":      query,
                "query_type": query_type
            })
            if "NOT_CAREER_RELATED" in advice.content:
                return {**state, "final_answer": "❌ This question is not related to career guidance."}

        else:
            prompt = PromptTemplate.from_template(
                "You are an expert AI Career Guidance Counselor for engineering students in India.\n\n"
                "Career Knowledge Base:\n{context}\n\n"
                "The student asked: {query}\n"
                "Query type detected: {query_type}\n\n"
                "STRICT RULE: If the question is NOT related to careers, jobs, internships, skills, "
                "education, or engineering — reply with exactly: NOT_CAREER_RELATED\n\n"
                "If the question IS career-related, answer with:\n"
                "1. Direct answer to their specific question\n"
                "2. Key skills or steps relevant to their question\n"
                "3. Recommended resources or platforms\n"
                "4. 3 actionable next steps\n"
            )
            advice = (prompt | self.llm).invoke({
                "context":    context,
                "query":      query,
                "query_type": query_type
            })
            if "NOT_CAREER_RELATED" in advice.content:
                return {**state, "final_answer": "❌ This question is not related to career guidance."}

        return {**state, "final_answer": advice.content}

    # ─────────────────────────────────────────
    # PUBLIC QUERY — ReAct decides, LangGraph executes
    # ─────────────────────────────────────────

    def query(self, user_query: str) -> Dict[str, Any]:
        if self.collection.count() == 0:
            return {
                "answer":      "**Knowledge base is empty!** Click **Load Career Documents** in the sidebar.",
                "query_type":  "unknown",
                "has_resume":  False,
                "react_steps": []
            }

        # ── Step 1: ReAct Agent reasons and decides what to retrieve ──
        react_prompt = (
            f"A student asked: '{user_query}'\n\n"
            f"Use the available tools to:\n"
            f"1. Classify the question type\n"
            f"2. Search the career knowledge base for relevant information\n"
            f"3. Check if there is resume information available\n"
            f"4. Summarize all gathered context as the final answer input\n\n"
            f"Think step by step before answering."
        )

        react_steps = []
        react_context = ""

        try:
            react_result = self.react_agent.invoke(
                {"messages": [{"role": "user", "content": react_prompt}]}
            )
            # Extract tool usage steps and final message
            for msg in react_result["messages"]:
                role = getattr(msg, "type", "") or getattr(msg, "role", "")
                content = getattr(msg, "content", "")
                if role in ("tool", "ai") and content:
                    react_steps.append({"role": role, "content": str(content)[:300]})
            # Use last AI message as enriched context hint
            ai_msgs = [m for m in react_result["messages"] if getattr(m, "type", "") == "ai"]
            if ai_msgs:
                react_context = getattr(ai_msgs[-1], "content", "")
        except Exception as e:
            react_steps = [{"role": "error", "content": str(e)}]

        # ── Step 2: LangGraph pipeline runs with enriched state ──
        initial_state: AgentState = {
            "user_query":     user_query,
            "query_type":     None,
            "branch":         None,
            "retrieved_docs": None,
            "reranked_docs":  None,
            "resume_context": None,
            "final_answer":   None,
            "has_resume":     False,
        }
        final_state = self.graph.invoke(initial_state)

        return {
            "answer":      final_state["final_answer"],
            "query_type":  final_state["query_type"],
            "has_resume":  final_state["has_resume"],
            "react_steps": react_steps,
        }

    # ─────────────────────────────────────────
    # DOCUMENT LOADING
    # ─────────────────────────────────────────

    def load_career_docs_folder(self) -> int:
        folder_path = os.path.join(os.path.dirname(__file__), DOCS_FOLDER)
        txt_files   = sorted(glob.glob(os.path.join(folder_path, "*.txt")))
        if not txt_files:
            return 0
        all_chunks, all_ids = [], []
        for filepath in txt_files:
            filename = os.path.basename(filepath)
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
            chunks = self.splitter.split_text(text)
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_ids.append(f"{filename}_{i}")
        self.collection.add(documents=all_chunks, ids=all_ids)
        return len(all_chunks)

    # ─────────────────────────────────────────
    # RESUME MANAGEMENT
    # ─────────────────────────────────────────

    def load_resume(self, file) -> int:
        self.clear_resume()
        if file.name.endswith(".pdf"):
            reader = PdfReader(file)
            text   = "".join(p.extract_text() or "" for p in reader.pages)
        else:
            text = file.read().decode("utf-8")
        text = text.strip()
        if not text:
            return 0
        chunks = self.splitter.split_text(text)
        if not chunks:
            chunks = [text[:500]]
        self.resume_collection.add(
            documents=chunks,
            ids=[f"resume_{i}" for i in range(len(chunks))]
        )
        return len(chunks)

    def clear_resume(self):
        existing = self.resume_collection.get()
        if existing and existing["ids"]:
            self.resume_collection.delete(ids=existing["ids"])

    def has_resume(self) -> bool:
        return self.resume_collection.count() > 0

    def get_doc_count(self) -> int:
        return self.collection.count()


# ─────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Career Guidance Platform",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Sora', sans-serif; }
.main-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%);
    padding: 2rem; border-radius: 16px; color: white; margin-bottom: 2rem;
}
.response-box {
    background: #f8fafc; border-left: 4px solid #3b82f6;
    border-radius: 0 12px 12px 0; padding: 1.5rem; margin: 1rem 0;
}
.resume-badge {
    background: #dcfce7; border: 1px solid #16a34a; border-radius: 8px;
    padding: 0.5rem 1rem; color: #15803d; font-size: 0.85rem;
    margin: 0.5rem 0; display: block;
}
.qtype-badge {
    display: inline-block; padding: 3px 12px; border-radius: 20px;
    font-size: 0.78rem; font-weight: 600; margin-bottom: 8px;
}
.qtype-career           { background:#dbeafe; color:#1e40af; }
.qtype-internship       { background:#fef9c3; color:#854d0e; }
.qtype-skills           { background:#f3e8ff; color:#6b21a8; }
.qtype-resume           { background:#dcfce7; color:#166534; }
.qtype-higher_education { background:#ffedd5; color:#9a3412; }
.qtype-general          { background:#f1f5f9; color:#475569; }
.stButton>button {
    background: linear-gradient(135deg, #1e3a5f, #3b82f6);
    color: white; border: none; border-radius: 10px;
    padding: 0.6rem 2rem; font-weight: 600; transition: all 0.2s;
}
.stButton>button:hover { box-shadow: 0 4px 15px rgba(59,130,246,0.4); }
</style>
""", unsafe_allow_html=True)

# ── Init RAG once per session ──────────────────
if "rag" not in st.session_state:
    st.session_state["rag"] = CareerGuidanceRAG()
rag = st.session_state["rag"]


def run_query(query: str):
    if rag.get_doc_count() == 0:
        st.warning("⚠️ Please click **Load Career Documents** in the sidebar first!")
        return

    with st.spinner("ReAct Agent thinking + LangGraph pipeline running..."):
        result = rag.query(query)

    st.session_state["query_count"] = st.session_state.get("query_count", 0) + 1

    # Query type badge
    qtype = result.get("query_type", "general")
    qtype_labels = {
        "career":           "💼 Career Path Query",
        "internship":       "🏢 Internship Query",
        "skills":           "🛠️ Skills & Learning Query",
        "resume":           "📄 Resume / Profile Query",
        "higher_education": "🎓 Higher Education Query",
        "general":          "💬 General Career Query",
    }
    st.markdown(
        f"<span class='qtype-badge qtype-{qtype}'>{qtype_labels.get(qtype, '💬 General')}</span>",
        unsafe_allow_html=True
    )

    if result.get("has_resume"):
        st.success("✅ Answer personalized using your uploaded resume")

    st.subheader("💡 Career Advice")
    st.markdown(
        f"<div class='response-box'>{result['answer']}</div>",
        unsafe_allow_html=True
    )


# ── Sidebar ────────────────────────────────────
with st.sidebar:

    # 1. Resume upload
    st.markdown("### 👤 Student Resume")
   
    resume_file = st.file_uploader(
        "Upload resume", type=["pdf", "txt"], accept_multiple_files=False
    )
    if resume_file:
        file_key = f"resume_loaded_{resume_file.name}"
        if not st.session_state.get(file_key):
            chunks = rag.load_resume(resume_file)
            for k in [k for k in st.session_state if k.startswith("resume_loaded_")]:
                del st.session_state[k]
            if chunks == 0:
                st.error("❌ Could not extract text. Try a .txt file instead.")
            else:
                st.session_state[file_key] = True
                st.success(f"✅ Resume loaded ({chunks} chunks)")

    if rag.has_resume():
        st.markdown(
            "<span class='resume-badge'>✅ Resume active — answers will be personalized</span>",
            unsafe_allow_html=True
        )
        if st.button("🗑️ Clear Resume"):
            rag.clear_resume()
            for k in [k for k in st.session_state if k.startswith("resume_loaded_")]:
                del st.session_state[k]
            st.rerun()

    st.divider()

    # 2. Load career docs
    
    if st.button("📚 Load Career Documents"):
        with st.spinner("Loading..."):
            count = rag.load_career_docs_folder()
        if count > 0:
            st.success(f"✅ {count} chunks loaded from 15 documents!")
        else:
            st.error("❌ career_docs/ folder not found next to app.py")

    st.divider()

    # 3. Stats
    st.markdown("### 📊 Stats")
    c1, c2 = st.columns(2)
    c1.metric("Chunks", rag.get_doc_count())
    c2.metric("Queries", st.session_state.get("query_count", 0))


# ── Main UI ────────────────────────────────────
st.markdown("""
<div class='main-header'>
    <h1 style='margin:0; font-size:2.2rem;'>🎯 AI Career Guidance Platform</h1>
 
</div>
""", unsafe_allow_html=True)

st.markdown("### 🚀 Quick Questions")
st.caption("Click any question for an instant answer")

quick_questions = [
    "Career paths for CSE students?",
    "Best internships for ECE students?",
    "How to become a Data Scientist?",
    "How to become an AI/ML Engineer?",
    "Career options in Mechanical Engineering?",
    "How to get internship at DRDO or ISRO?",
    "Skills needed for cybersecurity jobs?",
    "How to crack campus placements?",
    "Career options after Civil Engineering?",
    "How to prepare for GATE exam?",
    "Best free courses for programming?",
    "How to write a good resume?",
]

cols = st.columns(3)
for i, q in enumerate(quick_questions):
    if cols[i % 3].button(q, key=f"quick_{i}"):
        st.session_state["active_query"] = q

st.divider()

user_input = st.text_input(
    "Or type your own question:",
    placeholder="e.g. What skills do I need for ML jobs as a CSE student?"
)
if st.button("🎯 Get Advice") and user_input.strip():
    st.session_state["active_query"] = user_input.strip()

if st.session_state.get("active_query"):
    query = st.session_state["active_query"]
    st.markdown(f"**Your question:** _{query}_")
    st.divider()
    run_query(query)
    st.session_state["active_query"] = ""