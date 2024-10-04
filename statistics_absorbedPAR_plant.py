import pandas as pd
import numpy as np
import statsmodels.api as sm
import scikit_posthocs as sp
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
from scipy.stats import levene, shapiro


# Define a function to process ANOVA, assumptions, and plotting for each dataset.
def process_data(input_file, output_anova, output_assumptions, qqplot_path):
    # Load the data.
    data = pd.read_csv(input_file)
    print(data)

    # Create output.txt for anova results and assumptions.
    anova_output = open(output_anova, 'w')
    assumptions = open(output_assumptions, 'w')

    # ANOVA 1-way.
    model = ols('absorbedPAR_umol_m2_s1 ~ C(architecture)', data=data).fit()
    anova_results = sm.stats.anova_lm(model, typ=2)
    print(anova_results)

    # Function to format p-values with a minimum display value of 0.001
    def format_p_value(p):
        if p < 0.001:
            return "<0.001"
        else:
            return round(p, 3)

    # Apply the function to the ANOVA results' p-values
    anova_results['PR(>F)'] = anova_results['PR(>F)'].apply(format_p_value)

    # Get descriptive statistics.
    descriptive_statistics = data.describe()
    print(descriptive_statistics)

    # Write ANOVA results, model summary and descriptive statistics in txt file.
    anova_output.write(f"\nANOVA results for total absorbedPAR: \n {anova_results} \n")
    anova_output.write("\nModel summary: \n")
    anova_output.write(model.summary().as_text())
    anova_output.write(f"\nDescriptive statistics: {descriptive_statistics} \n")

    # Post-hoc test (TukeyHSD).
    if model.f_pvalue < 0.05:  # Check if the overall model is significant.
        posthoc = sp.posthoc_tukey_hsd(data['absorbedPAR_umol_m2_s1'], data['architecture'])
        posthoc = posthoc.map(lambda x: f"{x:.3f}")
        anova_output.write(f'\nPost-hoc (Tukeyhsd) Results:\n{posthoc}\n')

    # Assumption Checks: Levene's Test and Shapiro-Wilk Test for normality
    levene_stat, levene_p = levene(*[group['absorbedPAR_umol_m2_s1'] for name, group in data.groupby('architecture')])
    shapiro_stat, shapiro_p = shapiro(model.resid)
    assumptions.write(f'\nLevene Test: Stat={levene_stat}, p={levene_p}\n')
    assumptions.write(f'\nShapiro-Wilk Test: Stat={shapiro_stat}, p={shapiro_p}\n')

    # Plotting residuals for assumption checks.
    residuals = model.resid
    plt.figure()
    sm.qqplot(residuals, line='s')
    plt.title(f'Q-Q plot for residuals')
    plt.savefig(qqplot_path)

    # Close output files.
    anova_output.close()
    assumptions.close()


# Define input files and corresponding output paths for high and low densities.
files = [
    {
        'input_file': 'total_absorbedPAR_high.csv',
        'output_anova': 'output_statistics/ANOVA_total_absorbedPAR_high.txt',
        'output_assumptions': 'output_statistics/assumptions_total_absorbedPAR_high.txt',
        'qqplot_path': 'qq_plots/total_absorbedPAR_high.png',
    },
    {
        'input_file': 'total_absorbedPAR_low.csv',
        'output_anova': 'output_statistics/ANOVA_total_absorbedPAR_low.txt',
        'output_assumptions': 'output_statistics/assumptions_total_absorbedPAR_low.txt',
        'qqplot_path': 'qq_plots/total_absorbedPAR_low.png',
    }
]

# Loop through the files and apply the function to each file
for file_info in files:
    process_data(
        input_file=file_info['input_file'],
        output_anova=file_info['output_anova'],
        output_assumptions=file_info['output_assumptions'],
        qqplot_path=file_info['qqplot_path'],
    )

print("Analysis completed for both datasets.")


# Plotting part.
# Load datasets for high and low densities
data_high = pd.read_csv('total_absorbedPAR_high.csv')
data_low = pd.read_csv('total_absorbedPAR_low.csv')

# Function to calculate mean, std dev, and confidence intervals
def calculate_statistics(data):
    grouped = data.groupby(['architecture'])['absorbedPAR_umol_m2_s1']
    mean_values = grouped.mean().reset_index(name='mean')
    mean_values['std_dev'] = grouped.std().reset_index(name='std_dev')['std_dev']
    mean_values['n'] = grouped.count().reset_index(name='count')['count']
        
    return mean_values

# Calculate statistics for both high-density and low-density datasets
mean_values_high = calculate_statistics(data_high)
mean_values_low = calculate_statistics(data_low)

def significance_letters(dataframe: pd.DataFrame) -> pd.DataFrame:
    letter = ord('a')
    for i in range(len(dataframe)):
        dataframe.iloc[i,i] = 0
        dataframe.iloc[i,:i] = [1 for x in dataframe.iloc[i,:i]]
        dataframe.iloc[i,:] = [chr(letter) if x == 0 else '' for x in dataframe.iloc[i,:]]
        letter += 1

    return dataframe[:-1]


# Plotting function to avoid repetition
def barplot_absorbedPAR(mean_values, density_label, output_file):
    fig, ax = plt.subplots(figsize=(10, 6))
    significants =  significance_letters(pd.read_csv(f'output_statistics\Tukey_HSD_{density_label.lower()}_ranks.csv'))
    print(significants)
    # Plotting lineplot for each architecture type within the current density
    for architecture in mean_values['architecture'].unique():
        architecture_data = mean_values[mean_values['architecture'] == architecture]
        ax.bar(architecture_data['architecture'], architecture_data['mean'], yerr=architecture_data['std_dev'], 
        width=0.5, edgecolor='black', linewidth=0.5, capsize=5, color='green')
        ax.text(architecture_data['architecture'], architecture_data['mean'] + architecture_data['std_dev']+ 50, s="".join(significants.loc[:,architecture].values), ha='center', va='bottom', fontsize=12)

    # Set the title and labels
    ax.set_title(f'Total Absorbed PAR for {density_label} density')
    ax.set_xlabel('Architecture')
    ax.set_ylim(0, 1200)
    ax.set_ylabel('Absorbed PAR (umol m-2 s-1)')

    plt.savefig(output_file)
    plt.close(fig)


# Plot for high-density data
barplot_absorbedPAR(mean_values_high, 'High', 'plots/total_absorbedPAR_high_density.png')

# Plot for low-density data
barplot_absorbedPAR(mean_values_low, 'Low', 'plots/total_absorbedPAR_low_density.png')

print("Plots generated successfully for high and low densities.")



