import pandas as pd
import matplotlib.pyplot as plt

def save_results_table(model_results, results_df):
    if not model_results:
        print("No results to save.")
        return

    # Se il DataFrame non è stato fornito, crea un nuovo DataFrame
    if results_df is None:
        results_df = pd.DataFrame(columns=['Modello', 'Precision', 'Accuracy', 'F1 Score', 'Training Time (s)'])

    # Aggiungi i risultati di ogni modello al DataFrame
    for model_name, model_metrics in model_results.items():
        accuracy, precision, f1, training_time = model_metrics
        results_df.loc[len(results_df)] = [model_name, accuracy, precision, f1, training_time]

    # Disegna la tabella utilizzando matplotlib# Disegna la tabella utilizzando matplotlib
    plt.figure(figsize=(10, 5))

    # Crea la tabella con i dati e le etichette delle colonne
    table = plt.table(cellText=results_df.values, colLabels=results_df.columns, loc='center')

    # Colora la prima riga
    for j in range(len(results_df.columns)):
        table._cells[0, j].set_facecolor('lightgrey')
    
    # Colora la prima colonna
    for i in range(len(results_df)):
        table._cells[i+1, 0].set_facecolor('lightblue')

    # Lista delle colonne di interesse
    columns_of_interest = ['Accuracy', 'Precision', 'F1 Score']

    # Dizionario per memorizzare gli indici dei valori massimi per ciascuna colonna
    max_indices = {}

    #   Iterazione attraverso le colonne di interesse
    for column in columns_of_interest:
        # Trova l'indice della riga con il valore massimo nella colonna corrente
        max_index = results_df[column].idxmax()
        # Aggiungi l'indice al dizionario
        max_indices[column] = max_index

    # Colorazione delle celle con i valori massimi di verde
    for column, max_index in max_indices.items():
        table._cells[max_index + 1, results_df.columns.get_loc(column)].set_facecolor('lightgreen')  # +1 per compensare l'header della tabella

    '''
    # Imposta la famiglia del carattere per tutte le celle
        for i in range(len(results_df) + 1):
            for j in range(len(results_df.columns)):
                cell = table._cells[i, j]
                cell.set_fontproperties(font_manager.FontProperties(family='sans-serif'))
    '''
    # Salva la tabella come un'immagine PNG
    plt.savefig('plot_results/Results.png', bbox_inches='tight', pad_inches=0.05)

    plt.close()

    print("La tabella dei risultati è stata aggiornata e salvata come Results.png")
    return results_df
