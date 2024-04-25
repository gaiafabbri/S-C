import pandas as pd
import matplotlib.pyplot as plt

''' -------------- SAVING METRICS AND TRAINING TIME --------------------'''
'''
 1) This function saves the results of multiple models as a table.
 2) Visualizes it using Matplotlib.
 3) If no model results are provided, it prints a message: "No results to save.".
 4) It first checks if a DataFrame is provided, and if not, it creates a new DataFrame with predefined columns.
 5) Then, it iterates through the provided model results.
 6) Adds them to the DataFrame.
 7) Draws a table using Matplotlib.
 8) The function highlights the cells containing the maximum values of specified columns.
 9) Finally, it saves the table as a PNG image and prints a confirmation message.
'''
def save_results_table(model_results, results_df):
    if not model_results:
        print("No results to save.")
        return

    # If a DataFrame is not provided, create a new DataFrame
    if results_df is None:
        results_df = pd.DataFrame(columns=['Modello', 'Precision', 'Accuracy', 'F1 Score', 'Training Time (s)'])

    # Add the results of each model to the DataFrame
    for model_name, model_metrics in model_results.items():
        accuracy, precision, f1, training_time = model_metrics
        results_df.loc[len(results_df)] = [model_name, accuracy, precision, f1, training_time]

    # Draw the table
    plt.figure(figsize=(10, 5))

    table = plt.table(cellText=results_df.values, colLabels=results_df.columns, loc='center')

    # Aesthetic
    for j in range(len(results_df.columns)):
        table._cells[0, j].set_facecolor('lightgrey')
    for i in range(len(results_df)):
        table._cells[i+1, 0].set_facecolor('lightblue')

    # Serching for best results
    columns_of_interest = ['Accuracy', 'Precision', 'F1 Score']

    # Dictionary to store the indices of maximum values for each column
    max_indices = {}

    # Iterating through the columns of interest
    for column in columns_of_interest:
        max_index = results_df[column].idxmax()
        max_indices[column] = max_index

    # Coloring the cells with maximum values
    for column, max_index in max_indices.items():
        table._cells[max_index + 1, results_df.columns.get_loc(column)].set_facecolor('lightgreen')  # +1 to offset the table header


    # Saving table
    plt.savefig('plot_results/Results.png', bbox_inches='tight', pad_inches=0.05)

    plt.close()

    print("La tabella dei risultati è stata aggiornata e salvata come Results.png")
    return results_df
