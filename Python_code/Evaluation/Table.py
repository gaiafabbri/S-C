from PIL import Image, ImageDraw, ImageFont
import pandas as pd


''' -------------- SAVING METRICS AND TRAINING TIME --------------------'''
'''
 1) This function saves the results of multiple models as a table.
 2) Visualizes it using Matplotlib.
 3) If no model results are provided, it prints a message: "No results to save.".
 4) It first checks if a DataFrame is provided, and if not, it creates a new DataFrame with predefined columns.
 5) Then, it iterates through the provided model results: they include the model name, the accuracy, the precision, the f1-score and the training time
 6) Adds them to the DataFrame.
 7) Draws the table: it relies on Pillow to create a void images with specified dimensions (img_width e img_height) and then draws the table (and its content) wiht ImageDraw
 8) The best results are found and the corresponding cell is coloured in yellow
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

    # Create a blank image with specified dimensions
    img_width = 800  # Set as per your requirement
    img_height = 400  # Set as per your requirement
    img = Image.new('RGB', (img_width, img_height), color='white')
    #Draw the image
    draw = ImageDraw.Draw(img)

    # Define table dimensions
    table_width = 700
    table_height = 300
    cell_width = table_width / len(results_df.columns)
    cell_height = table_height / (len(results_df) + 1)  # +1 for the header row

    # Define font properties
    font = ImageFont.load_default()
    font_size = 12

    # Draw table headers
    for i, column in enumerate(results_df.columns):
        draw.rectangle([i * cell_width, 0, (i + 1) * cell_width, cell_height], outline='black', fill='lightgrey')
        draw.text((i * cell_width + 5, 5), column, fill='black', font=font)

  # Draw table content and find best results
    best_accuracy_index = results_df['Accuracy'].idxmax()
    best_precision_index = results_df['Precision'].idxmax()
    best_f1_index = results_df['F1 Score'].idxmax()
    fastest_time_index = results_df['Training Time (s)'].idxmin()

    for i, row in enumerate(results_df.values):
        for j, cell in enumerate(row):
            cell_color = 'white'  # Default color
            if i == best_accuracy_index and j == results_df.columns.get_loc('Accuracy'):
                cell_color = 'yellow'  # Yellow for best accuracy
            elif i == best_precision_index and j == results_df.columns.get_loc('Precision'):
                cell_color = 'yellow'  # Yellow for best precision
            elif i == best_f1_index and j == results_df.columns.get_loc('F1 Score'):
                cell_color = 'yellow'  # Yellow for best F1 Score
            elif i == fastest_time_index and j == results_df.columns.get_loc('Training Time (s)'):
                cell_color = 'yellow'  # Yellow for fastest training time

            draw.rectangle([j * cell_width, (i + 1) * cell_height, (j + 1) * cell_width, (i + 2) * cell_height],
                           outline='black', fill=cell_color)
            draw.text((j * cell_width + 5, (i + 1) * cell_height + 5), str(cell), fill='black', font=font)

    # Save the image
    img.save('plot_results/Results.png')

    print("The results have been updated in the table saved as Results.png")
    return results_df

