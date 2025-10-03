import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ------------------------------
# Generate sample data
# ------------------------------
def generate_sample_data(n_per_group=50, groups=['A','B','C']):
    rng = np.random.default_rng(42)
    data = []
    conversion_rates = {'A': 0.40, 'B': 0.50, 'C': 0.70}
    
    for g in groups:
        converted = rng.binomial(1, conversion_rates[g], n_per_group)
        for i, c in enumerate(converted):
            data.append({
                'user_id': f"{g}{i+1}",
                'group': g,
                'converted': c
            })
    return pd.DataFrame(data)

# ------------------------------
# Frequentist A/B/n Test
# ------------------------------
def frequentist_abtest(df):
    results = []
    groups = df['group'].unique()
    
    for g in groups:
        subset = df[df['group'] == g]
        conv_rate = subset['converted'].mean()
        n = len(subset)
        results.append([g, n, subset['converted'].sum(), conv_rate])
    
    results_df = pd.DataFrame(results, columns=["Group", "Sample Size", "Conversions", "Conversion Rate"])
    
    comparisons = []
    for i in range(len(groups)):
        for j in range(i+1, len(groups)):
            g1, g2 = groups[i], groups[j]
            d1, d2 = df[df['group']==g1], df[df['group']==g2]
            
            p1, n1 = d1['converted'].mean(), len(d1)
            p2, n2 = d2['converted'].mean(), len(d2)
            p_pool = (d1['converted'].sum() + d2['converted'].sum()) / (n1 + n2)
            se = np.sqrt(p_pool*(1-p_pool)*(1/n1 + 1/n2))
            
            z = (p1 - p2) / se if se > 0 else 0
            pval = 2*(1-stats.norm.cdf(abs(z)))
            better = g1 if p1 > p2 else g2
            comparisons.append([f"{g1} vs {g2}", round(p1,3), round(p2,3), round(z,3), round(pval,4), f"{better} better"])
    
    comp_df = pd.DataFrame(comparisons, columns=["Comparison","Conv A","Conv B","Z-Score","P-Value","Likely Better"])
    
    # Best group overall
    best_group = results_df.loc[results_df['Conversion Rate'].idxmax(), "Group"]
    
    return results_df, comp_df, best_group

# ------------------------------
# Bayesian A/B/n Test
# ------------------------------
def bayesian_abtest(df, samples=5000):
    groups = df['group'].unique()
    probs = {}
    posterior_means = {}
    
    for g in groups:
        data = df[df['group']==g]['converted']
        alpha = 1 + data.sum()
        beta = 1 + len(data) - data.sum()
        samples_g = np.random.beta(alpha, beta, samples)
        posterior_means[g] = samples_g.mean()
        probs[g] = samples_g
    
    draws = np.vstack([probs[g] for g in groups])
    best_counts = (draws.argmax(axis=0))
    best_probs = {groups[i]: (best_counts==i).mean() for i in range(len(groups))}
    
    results = pd.DataFrame({
        "Group": groups,
        "Posterior Mean": [posterior_means[g] for g in groups],
        "Prob Best": [best_probs[g] for g in groups]
    })
    
    best_group = max(best_probs, key=best_probs.get)
    return results, probs, best_group

# ------------------------------
# Main Analysis Function
# ------------------------------
def analyze_abtest(file=None, n=50):
    # Load data
    if file is not None:
        df = pd.read_csv(file.name)
    else:
        df = generate_sample_data(n)
    
    # Frequentist
    freq_summary, freq_comparisons, best_f = frequentist_abtest(df)
    
    fig1, ax = plt.subplots()
    ax.bar(freq_summary["Group"], freq_summary["Conversion Rate"], color="skyblue")
    ax.set_ylabel("Conversion Rate")
    ax.set_title("Conversion Rates by Group")
    plot1 = fig1
    
    # Bayesian
    bayes_summary, probs, best_b = bayesian_abtest(df)
    
    fig2, ax = plt.subplots()
    for g in probs:
        ax.hist(probs[g], bins=50, density=True, alpha=0.5, label=g)
    ax.set_title("Bayesian Posterior Distributions")
    ax.legend()
    plot2 = fig2
    
    # Build readable text summary
    text_summary = f"""
### Frequentist Summary
{freq_summary.to_string(index=False)}

### Frequentist Comparisons
{freq_comparisons.to_string(index=False)}

🏆 Frequentist Best Group: {best_f}

---

### Bayesian Summary
{bayes_summary.to_string(index=False)}

🏆 Bayesian Best Group: {best_b}
"""
    
    return text_summary, plot1, plot2

# ------------------------------
# Gradio UI
# ------------------------------
with gr.Blocks() as demo:
    gr.Markdown("# 🚀 A/B/n Testing Analyzer\nUpload CSV or generate demo data to compare groups.")
    
    with gr.Row():
        file_input = gr.File(label="Upload CSV (columns: user_id, group, converted)", file_types=[".csv"])
        n_input = gr.Slider(20, 200, value=50, step=10, label="Users per Group (for demo data)")
    
    run_button = gr.Button("Run Analysis")
    
    text_output = gr.Textbox(label="Analysis Summary", lines=25)
    plot1_output = gr.Plot(label="Conversion Rates (Frequentist)")
    plot2_output = gr.Plot(label="Bayesian Posterior Distributions")
    
    run_button.click(analyze_abtest, inputs=[file_input, n_input], outputs=[text_output, plot1_output, plot2_output])

demo.launch()
