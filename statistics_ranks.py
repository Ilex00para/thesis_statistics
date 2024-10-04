import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import scikit_posthocs as sp
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import levene, shapiro
import numpy as np
from scipy.stats import sem, t


# Define the ANOVA function.
def process_anova(input_file, output_anova, output_assumptions, qqplot_path):
    # Load the data.
    data = pd.read_csv(input_file)

    # Open output files once at the start
    anova_output = open(output_anova, 'w')
    assumptions_output = open(output_assumptions, 'w')

    # Process data by rank.
    for rank in data['rank'].unique():
        # Filter data for the current rank.
        rank_data = data[data['rank'] == rank]

    # ANOVA model.
        model = ols('absorbedPAR_umol_m2_s1 ~ C(architecture)', data=rank_data).fit()
        anova_results = sm.stats.anova_lm(model, typ=2)

        # Function to format p-values with a minimum display value of 0.001
        def format_p_value(p):
            if p < 0.001:
                return "<0.001"
            else:
                return round(p, 3)

        # Apply the function to the ANOVA results' p-values
        anova_results['PR(>F)'] = anova_results['PR(>F)'].apply(format_p_value)

        # Write ANOVA results for this rank to the file
        anova_output.write(f"\nANOVA results for rank {rank}:\n{anova_results}\n")

        # Post-hoc test (TukeyHSD).
    if model.f_pvalue < 0.05:  # Check if the overall model is significant.
        posthoc = sp.posthoc_tukey_hsd(data['absorbedPAR_umol_m2_s1'], data['architecture'])
        posthoc.to_csv(output_anova.split('.')[0].replace('ANOVA','Tukey_HSD')+'.csv', index=False) #Name needs to be chnged
        posthoc = posthoc.map(lambda x: f"{x:.3f}") ##??
        anova_output.write(f'\nPost-hoc (Tukeyhsd) Results:\n{posthoc}\n')

        # Assumption Checks: Levene's Test and Shapiro-Wilk Test for normality
        levene_stat, levene_p = levene(*[group['absorbedPAR_umol_m2_s1'] for name, group in data.groupby('architecture')])
        shapiro_stat, shapiro_p = shapiro(model.resid)
        
        # Write assumption test results for this rank
        assumptions_output.write(f'\nAssumptions for rank {rank}:\n')
        assumptions_output.write(f'Levene Test: Stat={levene_stat}, p={levene_p}\n')
        assumptions_output.write(f'Shapiro-Wilk Test: Stat={shapiro_stat}, p={shapiro_p}\n')

        # Plotting residuals for assumption checks.
        residuals = model.resid
        plt.figure()
        sm.qqplot(residuals, line='s')
        plt.title(f'Q-Q plot for residuals of rank {rank}')
        plt.savefig(f'{qqplot_path}_rank_{rank}.png')
        plt.close()

    # Close the output files after all ranks are processed
    anova_output.close()
    assumptions_output.close()


# Define input files and corresponding output paths
files = [
    {
        'input': 'combined_high_ranks.csv',
        'output_anova': 'output_statistics/ANOVA_high_ranks.txt',
        'output_assumption': 'output_statistics/assumptions_high_ranks.txt',
        'qqplot': 'qq_plots/high_density_qqplot.png'
    },
    {
        'input': 'combined_low_ranks.csv',
        'output_anova': 'output_statistics/ANOVA_low_ranks.txt',
        'output_assumption': 'output_statistics/assumptions_low_ranks.txt',
        'qqplot': 'qq_plots/low_density_qqplot.png'
    }
]

# Loop through each file and apply the ANOVA function
for file_info in files:
    process_anova(file_info['input'], file_info['output_anova'], file_info['output_assumption'], file_info['qqplot'])

print("ANOVA processing complete for all files.")


# Plotting part.
# Load datasets for high and low densities
data_high = pd.read_csv('combined_high_ranks.csv')
data_low = pd.read_csv('combined_low_ranks.csv')

# Function to calculate mean, std dev, and confidence intervals
def calculate_statistics(data, confidence=0.95):
    grouped = data.groupby(['architecture', 'rank'])['absorbedPAR_umol_m2_s1']
    mean_values = grouped.mean().reset_index(name='mean')
    mean_values['std_dev'] = grouped.std().reset_index(name='std_dev')['std_dev']
    mean_values['n'] = grouped.count().reset_index(name='count')['count']
    
    # Calculate the t-value for 95% confidence interval
    mean_values['t_value'] = mean_values['n'].apply(lambda x: t.ppf((1 + confidence) / 2., x - 1))
    
    # Calculate the margin of error
    mean_values['margin_error'] = mean_values['t_value'] * (mean_values['std_dev'] / np.sqrt(mean_values['n']))
    
    # Calculate lower and upper confidence bounds
    mean_values['ci_lower'] = mean_values['mean'] - mean_values['margin_error']
    mean_values['ci_upper'] = mean_values['mean'] + mean_values['margin_error']
    
    return mean_values

# Calculate statistics for both high-density and low-density datasets
mean_values_high = calculate_statistics(data_high)
mean_values_low = calculate_statistics(data_low)

# Plotting function to avoid repetition
def plot_absorbedPAR(mean_values, density_label, output_file):
    plt.figure(figsize=(10, 6))
    
    # Plotting lineplot for each architecture type within the current density
    for architecture in mean_values['architecture'].unique():
        architecture_data = mean_values[mean_values['architecture'] == architecture]
        sns.lineplot(x='rank', y='mean', data=architecture_data, label=architecture, marker='o')
        
        # Adding the confidence interval as a shaded area
        plt.fill_between(architecture_data['rank'], architecture_data['ci_lower'], architecture_data['ci_upper'], alpha=0.3)
    
    plt.title(f'Absorbed PAR over Leaf Ranks for {density_label} Density')
    plt.xlabel('Rank')
    plt.ylabel('Absorbed PAR (umol m-2 s-1)')
    plt.legend(title='Architecture Type')
    plt.savefig(output_file)
    plt.close()

# Plot for high-density data
plot_absorbedPAR(mean_values_high, 'High', 'plots/mean_absorbedPAR_for_high_density.png')

# Plot for low-density data
plot_absorbedPAR(mean_values_low, 'Low', 'plots/mean_absorbedPAR_for_low_density.png')

print("Plots generated successfully for high and low densities.")
