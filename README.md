a-b-testing-analyzer
Built an interactive app to analyze A/B/n experiments using both Frequentist (z-tests, p-values) and Bayesian (posterior distributions, probability best) methods. Features CSV upload, visualizations, and clear winner identification to support data-driven product decisions.

Demo: https://huggingface.co/spaces/aakash-malhan/AB-Testing-Simulator-Analyzer

<img width="728" height="762" alt="Screenshot 2025-10-03 162113" src="https://github.com/user-attachments/assets/f1b45c95-4e01-4b5c-abf0-aaee1fccd222" />


Features

- 📂 Upload CSV (with `user_id, group, converted`) or generate demo data
- 📊 **Frequentist Analysis**: conversion rates, z-scores, p-values, winner
- 🔮 **Bayesian Analysis**: posterior means, probability best, winner
- 🏆 Winner clearly displayed (both approaches)
- 📈 Visualizations:
  - Conversion rates by group
  - Bayesian posterior distributions

BUSINESS IMPACT:

Helps product teams test and validate hypotheses (e.g., new UI design vs old).

Reduces decision-making risk with probability-driven insights.

Supports multi-variant experiments (A/B/C/D), not just binary A/B.

Bayesian analysis gives more intuitive understanding for non-statisticians.
