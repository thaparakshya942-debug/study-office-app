"""Study office · who should we talk to this week?

A Streamlit app for the study office advisers. It ranks this week's first-year students by their risk of
leaving later in the semester, shows what a rule would have got right and wrong on last year's students (2025),
the same per group, and what the rule is worth with the costs the office chooses.

The model sits in model/: booster.json (XGBoost) + preprocess.json (preparation as plain numbers), loaded with
portable.py from the hotel app (session 10). The data is read straight from the course's GitHub repository.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from portable import Model

MODEL_DIR = Path(__file__).parent / "model"
URL = "https://raw.githubusercontent.com/aaubs/ds-master/main/assignments/study-office/data/"

# How each feature is described to an adviser (used for "why is this student on the list?")
LABEL = {
    "fees_owed": lambda v: "owes tuition fees" if v == 1 else "no fees owed",
    "su_scholarship": lambda v: "no SU grant" if v == 0 else "has an SU grant",
    "logins_total": lambda v: f"{v:.0f} logins in weeks 1-6",
    "logins_last3": lambda v: f"{v:.0f} logins in the last 3 weeks",
    "logins_trend": lambda v: "logging in less and less" if v < 0 else "logins steady or rising",
    "submitted_share": lambda v: f"handed in {v:.0%} of assignments",
    "missed_last3": lambda v: f"missed {v:.0f} assignments in the last 3 weeks",
    "quiz_mean": lambda v: f"quiz average {v:.0f}",
    "weeks_since_login": lambda v: f"{v:.0f} weeks since last login",
    "programme": lambda v: f"programme: {v}",
    "age": lambda v: f"age {v:.0f}",
    "admission_grade": lambda v: f"admission grade {v:.1f}",
    "international": lambda v: "international student" if v == 1 else "domestic student",
    "first_gen": lambda v: "first in family at university" if v == 1 else "parents went to university",
    "moved_from_home": lambda v: "moved away from home" if v == 1 else "lives at home",
    "married": lambda v: "married" if v == 1 else "not married",
    "evening_programme": lambda v: "evening programme" if v == 1 else "day programme",
    "gender": lambda v: f"gender: {v}",
}

st.set_page_config(page_title="Study office · who to talk to", page_icon="🎓", layout="wide")


# ---------------------------------------------------------------- model and data, loaded once
def find_model_dir():
    """Use the model/ folder; if only model.zip was uploaded to the repository, unpack it first."""
    if (MODEL_DIR / "booster.json").exists():
        return MODEL_DIR
    zipped = Path(__file__).parent / "model.zip"
    if zipped.exists():
        import zipfile
        with zipfile.ZipFile(zipped) as z:
            z.extractall(MODEL_DIR)
        for folder in [MODEL_DIR, MODEL_DIR / "model"]:     # the zip may or may not contain a model/ folder
            if (folder / "booster.json").exists():
                return folder
    st.error("No model found. Upload the model/ folder (or model.zip) from the notebook to this repository.")
    st.stop()


@st.cache_resource
def load_model():
    model_dir = find_model_dir()
    model = Model(model_dir)
    config_file = model_dir / "config.json"
    config = json.loads(config_file.read_text()) if config_file.exists() else {}
    card_file = model_dir / "README.md"
    card = card_file.read_text() if card_file.exists() else ""
    return model, config, card


@st.cache_data
def load_data():
    # Use the repository's own data/ folder if it is there; otherwise read it from the course repository
    local = Path(__file__).parent / "data"
    base = str(local) + "/" if (local / "history_week6.csv").exists() else URL
    history = pd.read_csv(base + "history_week6.csv")
    new = pd.read_csv(base + "new_week6.csv")
    return history, new


model, config, card = load_model()
history, new = load_data()

# 2025 = last year's students: the model never trained on them, and we know who left
last_year = history[history["cohort"] == 2025].copy()
last_year["risk"] = model.predict_proba(last_year)
new = new.copy()
new["risk"] = model.predict_proba(new)


def reasons(rows, top=3):
    """The features that push each student's risk up most, in plain words (SHAP values from XGBoost)."""
    contrib = model.contributions(rows).reset_index(drop=True)
    out = []
    for i, (_, row) in enumerate(rows.iterrows()):
        biggest = contrib.iloc[i].sort_values(ascending=False).head(top)
        words = [LABEL.get(f, lambda v, f=f: f"{f}: {v}")(row[f]) for f, c in biggest.items() if c > 0]
        out.append(" · ".join(words))
    return out


def suggested_talk(row):
    """A first suggestion for what kind of conversation to offer. The adviser decides."""
    kinds = []
    if row["fees_owed"] == 1:
        kinds.append("💰 money advice")
    if row["missed_last3"] >= 2 or row["submitted_share"] < 0.5 or row["quiz_mean"] < 45:
        kinds.append("📚 study support")
    if row["weeks_since_login"] >= 2 or row["logins_last3"] <= 3:
        kinds.append("📞 personal call, soon")
    return " + ".join(kinds) if kinds else "💬 general check-in"


def flag(df, rule, n, cut):
    """Which students the rule contacts."""
    if rule == "number":
        return df["risk"].rank(ascending=False, method="first") <= n
    return df["risk"] >= cut


def boxes(df, contacted):
    left = df["left"] == 1
    return {"reached": int((contacted & left).sum()), "worried": int((contacted & ~left).sum()),
            "missed": int((~contacted & left).sum()), "alone": int((~contacted & ~left).sum())}


def show_boxes(b, key=""):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🎯 Reached in time", b["reached"], help="Contacted, and would have left: the conversation had a chance to help.")
    c2.metric("📞 Worried for nothing", b["worried"], help="Contacted, but would have stayed anyway.")
    c3.metric("🚪 Missed", b["missed"], help="Not contacted, and left the university.")
    c4.metric("✅ Correctly left alone", b["alone"], help="Not contacted, and stayed.")
    contacted = b["reached"] + b["worried"]
    total_left = b["reached"] + b["missed"]
    precision = b["reached"] / contacted if contacted else 0
    recall = b["reached"] / total_left if total_left else 0
    st.markdown(
        f"Of the **{contacted}** students contacted, **{b['reached']}** were really going to leave "
        f"(**{precision:.0%}**, *precision*). Of the **{total_left}** students who left, the office reached "
        f"**{b['reached']}** (**{recall:.0%}**, *recall*).")


# ---------------------------------------------------------------- the rule, in the sidebar
with st.sidebar:
    st.header("📏 The rule")
    choice = st.radio("How does the office choose who to talk to?",
                      ["A fixed number of conversations", "Everyone above a risk level"])
    rule = "number" if choice.startswith("A fixed") else "cut"
    n = st.slider("Number of conversations", 5, 200, int(config.get("capacity", 40)), 1,
                  disabled=rule != "number", help="Three advisers can hold about 40 conversations at the end of week 6.")
    cut = st.slider("Contact everyone with a risk of at least (%)", 1, 90,
                    int(round(100 * config.get("best_cut_off", 0.14))), 1, disabled=rule != "cut") / 100
    st.caption("40 conversations = three advisers. A fourth adviser adds about 13.")
    st.divider()
    st.caption(f"Model: XGBoost, trained on the 2023-24 students, checked on 2025 · "
               f"AUC 2025: {config.get('auc_2025', {}).get('xgboost', 'n/a')}")

st.title("🎓 Study office · who should we talk to this week?")
st.markdown("The model ranks first-year students by their **risk of leaving** later this semester, from what the "
            "office knows at the end of **week 6**. It only makes a suggestion: **an adviser decides** who to "
            "contact and how.")

tab_list, tab_mistakes, tab_groups, tab_costs, tab_about = st.tabs(
    ["📋 This week's list", "🔍 What the rule gets wrong", "⚖️ Per group", "💶 Costs", "ℹ️ About"])

# ---------------------------------------------------------------- 1 · this week's list
with tab_list:
    new["contact"] = flag(new, rule, n, cut)
    k = int(new["contact"].sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Students this week", len(new))
    c2.metric("Suggested conversations", k)
    c3.metric("Expected to leave without help", f"{new['risk'].sum():.0f}",
              help="The sum of all risks: roughly how many of this week's students the model expects to leave.")

    only_list = st.toggle("Show only the students the rule suggests", value=True)
    view = new.sort_values("risk", ascending=False)
    if only_list:
        view = view[view["contact"]]
    view = view.head(300).copy()
    view["why on the list"] = reasons(view)
    view["suggested conversation"] = view.apply(suggested_talk, axis=1)
    view["risk %"] = (100 * view["risk"]).round()
    view["group"] = view["international"].map({0: "domestic", 1: "international"})
    st.dataframe(
        view[["student_id", "risk %", "contact", "why on the list", "suggested conversation", "programme", "group"]],
        column_config={"risk %": st.column_config.ProgressColumn("risk", min_value=0, max_value=100, format="%d%%"),
                       "contact": st.column_config.CheckboxColumn("on the list"),
                       "student_id": "student"},
        hide_index=True, width="stretch")
    st.download_button("⬇️ Download the list (CSV)",
                       view[["student_id", "risk %", "why on the list", "suggested conversation"]].to_csv(index=False),
                       "students_to_contact.csv", "text/csv")
    st.info("💡 **Before contacting anyone:** check the reasons. A student who logs in rarely but hands in every "
            "assignment and does well in quizzes may simply study differently, especially international students. "
            "Invite openly (\"we'd like to hear how your semester is going\"), never \"the system says you will drop out\".")

# ---------------------------------------------------------------- 2 · the mistakes on 2025
with tab_mistakes:
    st.markdown(f"**What if the office had used this rule last year?** The 2025 students: "
                f"{len(last_year)} students, of whom **{int(last_year['left'].sum())} left** later in the semester.")
    contacted_2025 = flag(last_year, rule, n, cut)
    b = boxes(last_year, contacted_2025)
    show_boxes(b)
    all_left = int(last_year["left"].sum())
    st.markdown(
        f"In plain words: **{b['reached']} students reached in time, {b['worried']} worried for nothing, "
        f"{b['missed']} missed.** For comparison, contacting {int(contacted_2025.sum())} students *at random* would "
        f"have reached about **{contacted_2025.sum() * all_left / len(last_year):.0f}** of the students who left.")

    st.markdown("#### How the mistakes change with the number of conversations")
    ranked = last_year.sort_values("risk", ascending=False)["left"].values
    ks = np.arange(1, 301)
    reached = np.cumsum(ranked)[ks - 1]
    curve = pd.DataFrame({"conversations": ks, "reached in time": reached, "worried for nothing": ks - reached,
                          "missed": all_left - reached}).set_index("conversations")
    st.line_chart(curve, color=["#E07B39", "#9AA5B1", "#211A52"])
    st.caption("Each extra conversation either reaches a student who would leave or worries one who would stay. "
               "No rule avoids both mistakes.")

# ---------------------------------------------------------------- 3 · per group
with tab_groups:
    st.markdown("**Is the rule equally good for international and domestic students?** (2025 students)")
    last_year["contacted"] = contacted_2025
    cols = st.columns(2)
    for col, (code, name) in zip(cols, [(0, "🇩🇰 Domestic students"), (1, "🌍 International students")]):
        part = last_year[last_year["international"] == code]
        with col:
            st.subheader(name)
            st.caption(f"{len(part)} students · {part['left'].mean():.1%} really left · "
                       f"average predicted risk {part['risk'].mean():.1%}")
            show_boxes(boxes(part, part["contacted"]))
    st.warning("⚠️ **Watch out:** international students log in much less on the learning platform "
               f"({history[history.international == 1].logins_total.mean():.0f} vs "
               f"{history[history.international == 0].logins_total.mean():.0f} logins in six weeks), yet they hand in "
               "as many assignments and score as well in quizzes. Few logins is a weaker warning sign for them, so "
               "the model can worry the wrong international students and miss others. Only about 14 international "
               "students left in 2025, so these numbers move a lot by chance: check them every year.")

# ---------------------------------------------------------------- 4 · costs (our own addition)
with tab_costs:
    st.markdown("**What is the rule worth?** These numbers are **assumptions**. The study office and university "
                "management should set them, ideally together with student representatives.")
    defaults = config.get("costs_dkk", {})
    c1, c2, c3, c4 = st.columns(4)
    talk = c1.number_input("A conversation costs (DKK)", 0, 10_000, int(defaults.get("conversation", 500)), 100)
    worry = c2.number_input("A worried student costs (DKK)", 0, 50_000, int(defaults.get("false_alarm", 2_000)), 500)
    leave = c3.number_input("A student who leaves costs (DKK)", 0, 500_000, int(defaults.get("student_leaves", 60_000)), 5_000)
    helps = c4.slider("A conversation keeps this share of students at risk", 0, 100,
                      int(100 * defaults.get("conversation_helps", 0.30)), 5) / 100

    def value(contacted):
        bx = boxes(last_year, contacted)
        return bx["reached"] * helps * leave - (bx["reached"] + bx["worried"]) * talk - bx["worried"] * worry

    cuts = np.round(np.arange(0.02, 0.91, 0.01), 2)
    values = [value(last_year["risk"] >= c) for c in cuts]
    best_cut = float(cuts[int(np.argmax(values))])
    best_n = int((last_year["risk"] >= best_cut).sum())
    now = value(contacted_2025)
    m1, m2, m3 = st.columns(3)
    m1.metric("Your rule, on 2025", f"{now:,.0f} DKK")
    m2.metric("Best risk level with these costs", f"{best_cut:.0%}")
    m3.metric("…which means conversations", best_n, f"{best_n - 40:+d} vs 40 (three advisers)", delta_color="off")
    st.line_chart(pd.DataFrame({"risk level (%)": (100 * cuts).astype(int), "net value (DKK)": values})
                  .set_index("risk level (%)"))
    breakeven = (talk + worry) / (helps * leave + worry) if (helps * leave + worry) else 1
    st.caption(f"A conversation pays off on average for students with a risk above about {breakeven:.0%} "
               "(break-even). Try a higher cost for a worried student, or conversations that help less, and watch "
               "the best rule move.")

# ---------------------------------------------------------------- 5 · about
with tab_about:
    left, right = st.columns(2)
    with left:
        st.markdown("""
### Before a university uses this
- **A person decides.** The list is a suggestion; an adviser checks every student before contacting them.
- **Students are told** that the university uses such a system, what data it uses and why.
- **Students can object** and be left out, without any consequence.
- **Only data known at week 6** is used. ECTS points, deregistration forms and the last login week were
  removed: they are only known later.
- **Mistakes are checked every year**, also per group, and the model is retrained on new students.
- **The model shows patterns, not causes.** Owing fees goes with leaving; it does not make a student leave.
""")
    with right:
        st.markdown("### Model card")
        st.markdown(card or "_No model card found in model/README.md._")
