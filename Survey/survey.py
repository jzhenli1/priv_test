import pandas as pd
import numpy as np
from scipy import stats


df = pd.read_csv('survey.csv', sep=';')

#################################
## DATA PREPROCESSING
#################################

# Define preprocessing function
def preprocess(data):
    # Remove participants who took less than 100 seconds to complete the survey
    data = data[data['duration'] > 100]
    # Remove irrelevant columns
    data = data.drop(data.columns[0:7], axis=1)
    data = data.drop(data.columns[819:], axis=1)
    # Replace all -77 with NaN
    data = data.replace(-77, np.nan)

    # Reshaping the data
    # Identify all v_66_... columns
    cols = [col for col in data.columns if col.startswith('v_66_')]
    data_list = []
    # Loop over each v_66_ column and calculate the means
    for col in cols:
        row = {
            'image': col,
            'mean': data[col].mean(),
            'stuttgart_yes': data[data['v_56'] == 1][col].mean(),
            'stuttgart_no': data[data['v_56'] == 2][col].mean(),
            'commute': data[data['v_67'] == 1][col].mean(),
            'recreation': data[data['v_67'] == 2][col].mean(),
            'equally': data[data['v_67'] == 3][col].mean(),
            'rarely': data[data['v_67'] == 4][col].mean()
        }
        data_list.append(row)

    # Create a DataFrame from the list of dictionaries
    data = pd.DataFrame(data_list)

    return data

df_new = preprocess(df)


#################################
## HYPOTHESIS TESTING
#################################

# Define hypothesis testing function
def hypothesis_testing(data, column1, column2):
    # H0: The mean of the columns are equal
    # Sig. level
    alpha = 0.05
    
    # Perform an Independent Samples t-test
    t_stat, p_val = stats.ttest_ind(data[column1], data[column2],
                                    equal_var=False, nan_policy='omit')

    # Print the results
    print('t-statistic: ', t_stat)
    print('p-value: ', p_val)
    if p_val < alpha:
        print('Reject the null hypothesis')
    else:
        print('Fail to reject the null hypothesis')

        
hypothesis_testing(df_new, 'mean', 'stuttgart_yes')
# t-statistic:  -2.8888115017587555
# p-value:  0.003926827267292419
# Reject the null hypothesis

hypothesis_testing(df_new, 'stuttgart_yes', 'stuttgart_no')
# t-statistic:  3.37113226546661
# p-value:  0.0007689703590340253
# Reject the null hypothesis



hypothesis_testing(df_new, 'mean', 'recreation')
# t-statistic:  -2.134236249734194
# p-value:  0.03298612528163634
# Reject the null hypothesis

hypothesis_testing(df_new, 'commute', 'recreation')
# t-statistic:  -2.55847448447655
# p-value:  0.010607484225381554
# Reject the null hypothesis
