# 🎓 Study office · who should we talk to this week?

A Streamlit app for the study office advisers (AAU Business Data Science, module 1, assignment 4).
It ranks this week's first-year students by their risk of leaving later in the semester, shows what a rule
would have got right and wrong on the 2025 students, the same per group, and what the rule is worth.

| Piece | Where |
|---|---|
| App | `app.py` |
| Model | `model/`: `booster.json` + `preprocess.json` (loaded by `portable.py`), `config.json`, `README.md` (model card) |
| Data | read from the course repository on GitHub |
| Hosting | Streamlit Community Cloud |

The model only suggests: an adviser decides who to contact, students are told about the system and can object.
