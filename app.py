from typing import Dict, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

st.set_page_config(
    page_title="AstroVector - Semantic Career Compatibility Engine",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

DIMENSION_INFO = {
    "vocation_10h": {
        "title": "10th House (Vocation & Public Standing)",
        "desc": "Career title, public authority, organizational rank, and industry sector.",
    },
    "daily_work_6h": {
        "title": "6th House (Operational Cadence)",
        "desc": "Day-to-day workflow, team environment, direct duties, and task autonomy.",
    },
    "intellect_mercury": {
        "title": "Mercury (Cognitive Alignment)",
        "desc": "Problem-solving style, communication, technical logic, and analysis.",
    },
    "action_mars": {
        "title": "Mars (Execution Drive & Pace)",
        "desc": "Operational tempo, stamina, crisis tolerance, and task turnaround.",
    },
}

DEFAULT_WEIGHTS = {
    "vocation_10h": 0.30,
    "daily_work_6h": 0.25,
    "intellect_mercury": 0.25,
    "action_mars": 0.20,
}


class CareerCompatibilityEngine:

    def __init__(
        self,
        dataset_name: str = "kalixlouiis/zodiac-career-archetypes",
        model_name: str = "all-MiniLM-L6-v2",
    ):
        self.device = "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)
        self.dataset_name = dataset_name
        self.df = self._fetch_dataset()
        self.vector_cache = self._build_cache()

    def _fetch_dataset(self) -> pd.DataFrame:
        ds = load_dataset(self.dataset_name, split="train")
        df = ds.to_pandas()
        return df.set_index("sign")

    def _build_cache(self) -> Dict[str, Dict[str, np.ndarray]]:
        cache = {}
        for sign in self.df.index:
            cache[sign] = {}
            for dim in DIMENSION_INFO.keys():
                text = str(self.df.loc[sign, dim])
                vec = self.model.encode(
                    text,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                cache[sign][dim] = vec
        return cache

    def sanitize_and_validate(self, text: str) -> Tuple[bool, str]:
        stripped = text.strip()
        
        try:
            stripped.encode("ascii")
        except UnicodeEncodeError:
            return (
                False,
                "Please enter the job title in English only (e.g., 'Data Analyst', 'Surgeon').",
            )

        if len(stripped) < 3:
            return False, "Input is too short. Please provide a full job title or role description."

        target_vec = self.model.encode(
            stripped, convert_to_numpy=True, normalize_embeddings=True
        )
        anchor_vec = self.model.encode(
            "Professional occupation, job title, career trade, or workplace industry.",
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        sim = float(np.dot(target_vec, anchor_vec))
        if sim < 0.12:
            return (
                False,
                "Invalid input detected. Please enter a valid job title or professional career.",
            )

        return True, ""

@staticmethod
    def scale_similarity(sim: float, base: float = 0.12, max_lim: float = 0.32) -> float:
        scaled = ((sim - base) / (max_lim - base)) * 100.0
        return float(np.clip(scaled, 0.0, 100.0))

    def compute_match(
        self,
        career_str: str,
        selections: Dict[str, str],
        active_weights: Dict[str, float],
    ) -> Dict:
        career_vec = self.model.encode(
            career_str.strip(), convert_to_numpy=True, normalize_embeddings=True
        )

        dim_scores = {}
        raw_cos = {}
        for dim, sign in selections.items():
            ref_vec = self.vector_cache[sign][dim]
            c_sim = float(np.dot(ref_vec, career_vec))
            raw_cos[dim] = c_sim
            dim_scores[dim] = self.scale_similarity(c_sim)

        weight_sum = sum(active_weights[d] for d in selections.keys())
        norm_weights = {d: active_weights[d] / weight_sum for d in selections.keys()}

        raw_composite = sum(
            raw_cos[d] * norm_weights[d] for d in selections.keys()
        )
        scaled_composite = self.scale_similarity(raw_composite)

        return {
            "composite_score": round(scaled_composite, 1),
            "composite_raw": round(raw_composite, 4),
            "dimension_scores": {k: round(v, 1) for k, v in dim_scores.items()},
            "raw_cos": {k: round(v, 4) for k, v in raw_cos.items()},
            "norm_weights": norm_weights,
        }


@st.cache_resource(show_spinner=False)
def load_engine():
    return CareerCompatibilityEngine()


def draw_radar(scores: Dict[str, float]) -> go.Figure:
    labels = [DIMENSION_INFO[k]["title"] for k in scores.keys()]
    vals = list(scores.values())
    labels.append(labels[0])
    vals.append(vals[0])

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=vals,
            theta=labels,
            fill="toself",
            fillcolor="rgba(37, 99, 235, 0.2)",
            line=dict(color="#1d4ed8", width=2),
            marker=dict(size=5, color="#1e40af"),
        )
    )
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=10)),
            angularaxis=dict(tickfont=dict(size=11)),
        ),
        showlegend=False,
        margin=dict(l=40, r=40, t=30, b=30),
        height=350,
    )
    return fig


def main():
    engine = load_engine()
    sign_list = list(engine.df.index)

    st.title("AstroVector: Semantic Career Compatibility Engine")
    st.markdown(
        "A vector-space matching pipeline mapping astrological career archetypes "
        "to professional titles via Dense Bi-Encoder representations."
    )

    with st.expander("Mathematical Formulation & Retrieval Architecture", expanded=False):
        st.markdown(
            r"""
            #### 1. Vector Projection & Cosine Similarity
            Let the target occupation text $C$ and each natal archetype dimension $D_i$ for placement sign $S$ 
            be mapped into a 384-dimensional unit hypersphere using a dense sentence encoder:
            $$\mathbf{v}_C = f_\theta(C), \quad \mathbf{v}_{D_i, S} = f_\theta(\text{text}(S, D_i)) \quad \text{where } \|\mathbf{v}\|_2 = 1$$
            
            The directional similarity for each astrological dimension $i$ is calculated as:
            $$S_i = \mathbf{v}_{D_i, S} \cdot \mathbf{v}_C = \cos(\theta_i)$$

            #### 2. Normalized Hierarchical Aggregation
            Given user-defined or default weights $w_i$, the composite semantic score is:
            $$S_{\text{raw}} = \frac{\sum_{i \in \text{Provided}} w_i \cdot S_i}{\sum_{i \in \text{Provided}} w_i}$$

            #### 3. Bounded Calibration Function
            Cosine similarities are calibrated into an intuitive percentage scale using empirical bounds:
            $$\text{Score}_{\text{final}} (\%) = \text{clip}\left(\frac{S_{\text{raw}} - 0.15}{0.70 - 0.15} \times 100, \ 0, \ 100\right)$$
            """
        )

    with st.sidebar:
        st.header("Natal Placements")
        st.caption("Select your signs below:")

        sun = st.selectbox("Sun Sign (Core Drive)", sign_list, index=0)
        mercury = st.selectbox("Mercury Sign (Intellect & Logic)", sign_list, index=2)
        mars = st.selectbox("Mars Sign (Action & Execution)", sign_list, index=7)

        st.markdown("---")
        st.markdown("**Precision Placements (Optional)**")
        use_mc = st.checkbox("Specify 10th House / MC", value=True)
        mc = st.selectbox("10th House Sign", sign_list, index=9) if use_mc else sun

        use_6h = st.checkbox("Specify 6th House", value=False)
        h6 = st.selectbox("6th House Sign", sign_list, index=5) if use_6h else sun

        st.markdown("---")
        st.markdown("### Creator Profile")
        st.markdown(
            """
            - [Hugging Face](https://huggingface.co/kalixlouiis)
            - [LinkedIn](https://www.linkedin.com/in/khant-sint-heinn)
            - [GitHub](https://github.com/kalixlouiis)
            """
        )

    target_career = st.text_input(
        "Enter Target Career / Job Role (English only)",
        placeholder="e.g., Natural Language Processing Engineer, Corporate Litigator, Trauma Surgeon",
    )

    if st.button("Evaluate Compatibility", type="primary", use_container_width=True):
        if not target_career:
            st.error("Please enter a target career.")
            return

        valid, msg = engine.sanitize_and_validate(target_career)
        if not valid:
            st.warning(msg)
            return

        selections = {
            "vocation_10h": mc,
            "daily_work_6h": h6,
            "intellect_mercury": mercury,
            "action_mars": mars,
        }

        weights = DEFAULT_WEIGHTS.copy()
        if not use_mc:
            weights["vocation_10h"] = 0.15
            weights["intellect_mercury"] = 0.35
            weights["action_mars"] = 0.30

        result = engine.compute_match(target_career, selections, weights)

        st.markdown("---")
        col_res, col_plot = st.columns([1, 2])

        with col_res:
            st.metric(
                label="Overall Alignment Index",
                value=f"{result['composite_score']}%",
                delta=f"Cosine Sim: {result['composite_raw']}",
            )
            st.markdown("##### Configuration Overview")
            st.write(f"- **Vocation (10H):** {selections['vocation_10h']}")
            st.write(f"- **Routine (6H):** {selections['daily_work_6h']}")
            st.write(f"- **Intellect (Mercury):** {selections['intellect_mercury']}")
            st.write(f"- **Execution (Mars):** {selections['action_mars']}")

        with col_plot:
            fig = draw_radar(result["dimension_scores"])
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Dimensional Diagnostics")
        cols = st.columns(4)
        for i, (dim, score) in enumerate(result["dimension_scores"].items()):
            meta = DIMENSION_INFO[dim]
            with cols[i]:
                st.metric(label=meta["title"], value=f"{score}%")
                st.caption(f"**Sign:** {selections[dim]}")
                st.caption(meta["desc"])
                st.progress(score / 100.0)

    st.markdown("---")
    st.warning(
        """
        **Disclaimer / သတိပေးချက်**
        
        * **EN:** This system is an experimental machine learning application built for portfolio demonstration. Astrological archetypes are processed via semantic vector models for informational and entertainment purposes only. It should not be used as definitive professional, career, legal, or financial advice.
        * **MY:** ဤစနစ်သည် Machine Learning နှင့် NLP နည်းပညာများကို သရုပ်ပြရန် ဖန်တီးထားသော စမ်းသပ်မှု ပရောဂျက်တစ်ခုသာ ဖြစ်ပါသည်။ နက္ခတ်ဗေဒင်ဆိုင်ရာ အချက်အလက်များနှင့် အလုပ်အကိုင် ကိုက်ညီမှု ရလဒ်များသည် အသိပညာပေးခြင်းနှင့် ဖျော်ဖြေရေး သဘောတရားအတွက်သာ ဖြစ်ပြီး၊ တကယ့် ဘဝအလုပ်အကိုင်၊ ဥပဒေ သို့မဟုတ် ငွေကြေးဆိုင်ရာ အာမခံချက်ရှိသော ဆုံးဖြတ်ချက်များအဖြစ် အသုံးမပြုသင့်ပါ။
        """
    )


if __name__ == "__main__":
    main()
