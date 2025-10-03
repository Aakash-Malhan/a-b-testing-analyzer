a-b-testing-analyzer
Built an interactive app to analyze A/B/n experiments using both Frequentist (z-tests, p-values) and Bayesian (posterior distributions, probability best) methods. Features CSV upload, visualizations, and clear winner identification to support data-driven product decisions.

Demo: https://huggingface.co/spaces/aakash-malhan/AB-Testing-Simulator-Analyzer

Features

- 📂 Upload CSV (with `user_id, group, converted`) or generate demo data
- 📊 **Frequentist Analysis**: conversion rates, z-scores, p-values, winner
- 🔮 **Bayesian Analysis**: posterior means, probability best, winner
- 🏆 Winner clearly displayed (both approaches)
- 📈 Visualizations:
  - Conversion rates by group
  - Bayesian posterior distributions

**Business Impact**<img width="787" height="596" alt="Screenshot 2025-10-03 162152" src="https://github.com/user-attachments/assets/e50e115a-1d93-4c2c-8964-9421c5d4d393" />
<img width="782" height="589" alt="Screenshot 2025-10-03 162143" src="https://github.com/user-attachments/assets/2a4ba168-322f-47b6-84f0-e7f5706e97e4" />
<img width="728" height="762" alt="Screenshot 2025-10-03 162113" src="https://github.com/user-attachments/assets/4a1535c0-d425-4e55-bef8-794e47e64348" />


Helps product teams test and validate hypotheses (e.g., new UI design vs old).

Reduces decision-making risk with probability-driven insights.

Supports multi-variant experiments (A/B/C/D), not just binary A/B.

Bayesian analysis gives more intuitive understanding for non-statisticians.
