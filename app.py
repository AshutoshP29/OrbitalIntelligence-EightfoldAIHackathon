"""
Streamlit Application: Verified Capability Extraction & Matching
Matches GitHub profiles or resume text against job descriptions using FAISS embeddings
and provides Glass-Box explanations.
"""

import streamlit as st
import os
from dotenv import load_dotenv

from scraper import get_github_profile, load_resumes_from_csv, load_jobs_from_csv
from engine import EmbeddingEngine, create_profile_text
from utils import ExplainabilityEngine, generate_verified_signals, format_reasoning_chain

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Verified Capability Extraction - Eightfold.ai",
    page_icon="🚀",
    layout="wide"
)

# Title and description
st.title("🚀 Verified Capability Extraction & Matching")
st.markdown(
    """
    
    Extract language proficiency and commit quality from GitHub profiles or resume datasets,
    match against job descriptions, and receive explainable Glass-Box reports.
    """
)

# Sidebar for settings and mode selection
st.sidebar.header("⚙️ Configuration")
st.sidebar.markdown("---")

# Mode selection
mode = st.sidebar.radio(
    "Select Mode:",
    ["GitHub Profile Matching", "Resume Text Matching"]
)

# Check API keys
def check_api_keys():
    """Verify required API keys are configured."""
    github_token = os.getenv("GITHUB_TOKEN")
    
    if not github_token or github_token == "your_github_token_here":
        st.sidebar.error("❌ GitHub token not configured")
        return False
    
    st.sidebar.success("✓ API Keys configured")
    return True


# ==================== MODE 1: GITHUB PROFILE MATCHING ====================
if mode == "GitHub Profile Matching":
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Job Description")
        job_description = st.text_area(
            "Enter the job description to match against:",
            height=200,
            placeholder="Enter job requirements, skills, and responsibilities..."
        )
    
    with col2:
        st.subheader("👤 GitHub Profile")
        github_username = st.text_input(
            "Enter GitHub username:",
            placeholder="e.g., torvalds, gvanrossum"
        )
    
    # Process button
    if st.button("🔍 Analyze & Match", type="primary", key="github_match"):
        if not check_api_keys():
            st.error("Please configure API keys in .env file and restart the app")
            st.stop()
        
        if not job_description.strip():
            st.error("Please enter a job description")
            st.stop()
        
        if not github_username.strip():
            st.error("Please enter a GitHub username")
            st.stop()
        
        # Fetch GitHub data
        with st.spinner("🔄 Fetching GitHub profile..."):
            try:
                github_data = get_github_profile(github_username)
            except ValueError as e:
                st.error(f"❌ {str(e)}")
                st.stop()
        
        # Create embeddings and compute match score
        with st.spinner("⚙️ Computing embeddings and matching..."):
            try:
                engine = EmbeddingEngine()
                profile_text = create_profile_text(github_data=github_data)
                match_score = engine.compute_match_score(job_description, profile_text)
            except Exception as e:
                st.error(f"❌ Embedding error: {str(e)}")
                st.stop()
        
        # Generate verified signals
        with st.spinner("✓ Generating verified capability signals..."):
            try:
                verified_signals = generate_verified_signals(
                    github_data,
                    job_description,
                    match_score
                )
            except Exception as e:
                st.error(f"❌ Signal generation error: {str(e)}")
                st.stop()
        
        # Generate explanation
        with st.spinner("💭 Generating Glass-Box explanation..."):
            try:
                llm_engine = ExplainabilityEngine()
                explanation = llm_engine.generate_explanation(
                    job_description=job_description,
                    github_username=github_username,
                    match_score=match_score,
                    top_languages=github_data.get("top_5_languages", {}),
                    total_stars=github_data.get("total_stars", 0),
                    verification_signals=verified_signals
                )
            except Exception as e:
                st.error(f"❌ Explanation generation error: {str(e)}")
                st.stop()
        
        st.success("✓ Analysis complete!")
        st.markdown("---")
        
        # Display tabs
        tab1, tab2, tab3, tab4 = st.tabs(
            ["📊 Match Score", "✓ Verified Signals", "💭 Reasoning Chain", "👤 Profile Summary"]
        )
        
        with tab1:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.metric("Overall Match Score", f"{match_score:.0%}", delta=f"{match_score:.2f}", delta_color="off")
            
            st.markdown("### Match Interpretation")
            if match_score > 0.85:
                st.success("🟢 **Excellent Match**")
            elif match_score > 0.7:
                st.info("🟡 **Good Match**")
            elif match_score > 0.55:
                st.warning("🟠 **Moderate Match**")
            else:
                st.error("🔴 **Poor Match**")
        
        with tab2:
            st.markdown("### Real-World Artifacts")
            cols = st.columns(2)
            for idx, (signal_name, signal_value) in enumerate(verified_signals.items()):
                with cols[idx % 2]:
                    st.info(f"**{signal_name}**: {signal_value}")
        
        with tab3:
            st.markdown("### Glass-Box Explanation")
            st.markdown(f"> {explanation}")
        
        with tab4:
            st.markdown("### Developer Profile Summary")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Core Skills")
                top_langs = github_data.get("top_5_languages", {})
                for lang, count in list(top_langs.items())[:5]:
                    st.write(f"- **{lang}** ({count} repos)")
            with col2:
                st.markdown("#### Impact Metrics")
                st.write(f"- Total Stars: {github_data.get('total_stars', 0)}")
                st.write(f"- Public Repos: {github_data.get('public_repos', 0)}")


# ==================== MODE 2: RESUME TEXT MATCHING ====================
else:
    
    # Job description at top (always visible)
    st.subheader("📋 Job Description")
    jd_text = st.text_area(
        "Enter the job description:",
        height=150,
        placeholder="Enter job requirements, skills, and responsibilities...",
        key="jd_main"
    )
    
    st.markdown("---")
    st.markdown("### 📄 Select Resume Source")
    
    # Tab for dataset vs custom
    data_tab1, data_tab2 = st.tabs(["📁 Sample Data", "📝 Custom Text"])
    
    with data_tab1:
        st.markdown("**Use sample resumes from our dataset**")
        
        # Load datasets
        resumes = load_resumes_from_csv()
        
        if resumes:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Select Candidate")
                selected_resume_idx = st.selectbox(
                    "Choose a resume:",
                    range(len(resumes)),
                    format_func=lambda i: f"ID: {resumes[i]['id']} ({resumes[i]['category']})",
                    key="resume_select"
                )
                selected_resume = resumes[selected_resume_idx]
                st.write(f"**Category**: {selected_resume['category']}")
                st.write(f"**ID**: {selected_resume['id']}")
            
            with col1:
                resume_text = selected_resume['resume_text']
                st.markdown("#### Resume Preview")
                st.text(resume_text[:300] + "..." if len(resume_text) > 300 else resume_text)
            
            if st.button("🔍 Compare & Match", type="primary", key="sample_match"):
                if not jd_text.strip():
                    st.error("Please enter a job description")
                    st.stop()
                
                # Create embeddings and compute match score
                with st.spinner("⚙️ Computing embeddings and matching..."):
                    try:
                        engine = EmbeddingEngine()
                        match_score = engine.compute_match_score(jd_text, resume_text)
                    except Exception as e:
                        st.error(f"❌ Embedding error: {str(e)}")
                        st.stop()
                
                # Generate explanation
                with st.spinner("💭 Generating Glass-Box explanation..."):
                    try:
                        llm_engine = ExplainabilityEngine()
                        explanation = llm_engine.generate_explanation(
                            job_description=jd_text,
                            github_username=f"Resume ID: {selected_resume['id']}",
                            match_score=match_score,
                            top_languages={},
                            total_stars=0,
                            verification_signals={"Resume Type": "Sample Data", "Category": selected_resume['category']}
                        )
                    except Exception as e:
                        st.error(f"❌ Explanation generation error: {str(e)}")
                        st.stop()
                
                st.success("✓ Analysis complete!")
                st.markdown("---")
                
                # Display results
                tab1, tab2, tab3 = st.tabs(
                    ["📊 Match Score", "💭 Reasoning Chain", "📊 Comparison"]
                )
                
                with tab1:
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        st.metric("Overall Match Score", f"{match_score:.0%}", delta=f"{match_score:.2f}", delta_color="off")
                    
                    st.markdown("### Match Interpretation")
                    if match_score > 0.85:
                        st.success("🟢 **Excellent Match**")
                    elif match_score > 0.7:
                        st.info("🟡 **Good Match**")
                    elif match_score > 0.55:
                        st.warning("🟠 **Moderate Match**")
                    else:
                        st.error("🔴 **Poor Match**")
                
                with tab2:
                    st.markdown("### Glass-Box Explanation")
                    st.markdown(f"> {explanation}")
                
                with tab3:
                    st.markdown("### Candidate vs Job")
                    comp_col1, comp_col2 = st.columns(2)
                    
                    with comp_col1:
                        st.markdown("#### Resume")
                        st.text(resume_text[:400] + "..." if len(resume_text) > 400 else resume_text)
                    
                    with comp_col2:
                        st.markdown("#### Job Description")
                        st.text(jd_text[:400] + "..." if len(jd_text) > 400 else jd_text)
        else:
            st.error("❌ Could not load datasets")
    
    with data_tab2:
        st.markdown("**Paste a custom resume**")
        
        st.subheader("📄 Resume Text")
        resume_text = st.text_area(
            "Paste the resume:",
            height=200,
            placeholder="Paste candidate's resume or profile...",
            key="custom_resume"
        )
        
        if st.button("🔍 Compare & Match", type="primary", key="custom_match"):
            if not jd_text.strip():
                st.error("Please enter a job description above")
                st.stop()
            
            if not resume_text.strip():
                st.error("Please enter a resume")
                st.stop()
            
            # Create embeddings and compute match score
            with st.spinner("⚙️ Computing embeddings and matching..."):
                try:
                    engine = EmbeddingEngine()
                    match_score = engine.compute_match_score(jd_text, resume_text)
                except Exception as e:
                    st.error(f"❌ Embedding error: {str(e)}")
                    st.stop()
            
            # Generate explanation
            with st.spinner("💭 Generating Glass-Box explanation..."):
                try:
                    llm_engine = ExplainabilityEngine()
                    explanation = llm_engine.generate_explanation(
                        job_description=jd_text,
                        github_username="Resume Candidate",
                        match_score=match_score,
                        top_languages={},
                        total_stars=0,
                        verification_signals={"Resume Type": "Custom"}
                    )
                except Exception as e:
                    st.error(f"❌ Explanation generation error: {str(e)}")
                    st.stop()
            
            st.success("✓ Analysis complete!")
            st.markdown("---")
            
            # Display results
            tab1, tab2, tab3 = st.tabs(
                ["📊 Match Score", "💭 Reasoning Chain", "📊 Comparison"]
            )
            
            with tab1:
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.metric("Overall Match Score", f"{match_score:.0%}", delta=f"{match_score:.2f}", delta_color="off")
                
                st.markdown("### Match Interpretation")
                if match_score > 0.85:
                    st.success("🟢 **Excellent Match** - Strong alignment between resume and JD")
                elif match_score > 0.7:
                    st.info("🟡 **Good Match** - Solid alignment with some gaps")
                elif match_score > 0.55:
                    st.warning("🟠 **Moderate Match** - Significant alignment but notable gaps")
                else:
                    st.error("🔴 **Poor Match** - Limited alignment with JD requirements")
            
            with tab2:
                st.markdown("### Glass-Box Explanation")
                st.markdown(f"> {explanation}")
            
            with tab3:
                st.markdown("### Resume vs JD Comparison")
                comp_col1, comp_col2 = st.columns(2)
                
                with comp_col1:
                    st.markdown("#### Job Description")
                    st.text(jd_text[:400] + "..." if len(jd_text) > 400 else jd_text)
                
                with comp_col2:
                    st.markdown("#### Resume")
                    st.text(resume_text[:400] + "..." if len(resume_text) > 400 else resume_text)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
    Built using Streamlit, PyGithub, Sentence-Transformers and FAISS
    </div>
    """,
    unsafe_allow_html=True
)
