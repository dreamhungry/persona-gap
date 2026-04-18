# Persona-Gap

**Do LLM Agents Act as They Say?**  
A toolkit for measuring the gap between expressed and behavioral personality in LLM agents.

---

## 🔍 Overview

Large Language Model (LLM) agents are often prompted with specific personalities (e.g., aggressive, cooperative). However, it remains unclear whether these *expressed traits* are reflected in their actual decision-making behavior.

This project introduces a framework to:

- Quantify **expressed personality** from language
- Measure **behavioral personality** from actions
- Evaluate the **alignment (or gap)** between them
- Analyze **consistency over time**

---

## 🧠 Key Features

- 🎭 Personality-conditioned LLM agents
- 🎮 Pluggable environments (RLCard / TextArena / custom Gym)
- 📊 Behavioral trait extraction (risk, aggression, cooperation, deception)
- 📏 Alignment metrics (cosine similarity, KL divergence)
- 🔁 Counterfactual evaluation via state replay

---

## ⚙️ Installation

```bash
git clone https://github.com/yourname/persona-gap
cd persona-gap
pip install -r requirements.txt
