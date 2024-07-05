#!/bin/bash

######################################################
# Function to check the ROOT and python availability #
######################################################

# Function to ask the user what software is installed
check_software() {
    while true; do
        echo "Select which software is installed on your PC:"
        echo "1. Only ROOT"
        echo "2. Only Python"
        echo "B. Both (ROOT and Python)"
        echo "N. Neither of them"
        read -p "Insert your choice (1/2/B/N): " choice
    
        case $choice in
            1)
                echo "Use the dockerfile for Python analysis e chose only the ROOT analysis."
                return 1
                ;;
            2)
                echo "Use the dockerfile for ROOT analysis e chose only the Python analysis"
                return 2
                ;;
            B|b)
                echo "You can go ahead and perform both analysis."
                return 3
                ;;
            N|n)
                echo "Use dockerfile to run ROOT and Python scripts."
                exit 1
                ;;
            *)
                echo "Not valid choice. Please, insert 1, 2, B, or N."
                ;;
        esac
    done
}


##############################################
# Function to check the presence of ROOT files #
##############################################

check_root_file() {
    local tmva_ml_folder="TMVA_ML/images"
    local Python_code_folder="Python_code/images"

    echo "Checking for ROOT files in '$tmva_ml_folder' folder:"
    tmva_ml_root_files=$(find "$tmva_ml_folder" -maxdepth 1 -type f -iname '*.root')
    if [ -n "$tmva_ml_root_files" ]; then
        echo "ROOT files found in '$tmva_ml_folder' folder:"
        echo "$tmva_ml_root_files"
    else
        echo "No ROOT files found in '$tmva_ml_folder' folder."
        exit 1
    fi

    echo "Checking for ROOT files in '$Python_code_folder' folder:"
    Python_code_root_files=$(find "$Python_code_folder" -maxdepth 1 -type f -iname '*.root')
    if [ -n "$Python_code_root_files" ]; then
        echo "ROOT files found in '$Python_code_folder' folder:"
        echo "$Python_code_root_files"
    else
        echo "No ROOT files found in '$Python_code_folder' folder."
        exit 1
    fi
}


#################################################
# Function to generate delete command           #
#################################################

generate_delete_command() {
    local folder_path="$1"
    eval "rm -rf \"$folder_path\""
}


##################################
# Function to generate ROOT files #
##################################

Gen_files(){
    while true; do
        echo "Choose whether to use the backup file or generate it; if you choose the backup, only 10 000 events are present in the file"
        read -p "Do you want to generate or use the file in the backup folder? Please check the filename in the code! (g = generate/b = backup): " choice
        case $choice in
            [Gg]* ) root -l -q "ROOT_Gen/Generation.C(100000, 16, 16)"; break;;
            [Bb]* )
                    cp -r images_backup images
                    break
                    ;;
            * ) echo "Please choose g to generate or b for backup.";;
        esac
    done
}


####################################################
# Function to delete all '__pycache__' folders     #
# from subdirectories of a folder                  #
####################################################

delete_all_pycache_folders() {
    local parent_folder="$1"
    
    # Use the 'find' command to find all '__pycache__' folders and delete them
    find "$parent_folder" -type d -name "__pycache__" -exec rm -rf {} +
}


###################################
# Function to move 'images' folders #
###################################

move_images_folders() {

    # Make a copy and move it into "TMVA_ML"
    cp -r "images" "TMVA_ML/"

    # Make a copy and move it into "Python_code"
    cp -r "images" "Python_code/"
    
    # Make a copy and move it into "analysis_results"
    cp -r "images" "analysis_results/"

    # Delete "images" at the top level
    rm -rf "images"
}


##################
# Main function #
##################

main() {
    # Stampa le versioni di ROOT e Python
    root_version=$(root-config --version 2>/dev/null)
    python_version=$(python3 --version 2>&1)

    echo "This code has been tested with the following versions of ROOT and Python:"
    echo "ROOT version: $root_version"
    echo "Python version: $python_version"

    generate_delete_command "Python_code/images"
    generate_delete_command "TMVA_ML/images"
    generate_delete_command "analysis_results/images"
    generate_delete_command "Python_code/plot_results"
    generate_delete_command "TMVA_ML/dataset"
    delete_all_pycache_folders "Python_code"
    rm "TMVA_ML/TMVA_CNN_ClassificationOutput.root"

    check_software
    software_status=$?
    
    echo "Please note that the code has been tested with the following Python libraries: pandas 2.2.1, numpy 1.26.4, tensorflow 2.16.1, keras 3.1.1, torch 2.2.2, uproot 5.3.2, scikit-learn 1.4.1"

    echo "Alright, let's start."
    
    generate_delete_command "ROOT_Gen/images"
    Gen_files
    move_images_folders

    while true; do
        if [ $software_status -eq 3 ]; then
            read -p "To start the analysis please press one of the following button (1 = ROOT, 2 = Python, B = run both): " choice_root
        elif [ $software_status -eq 1 ]; then
            read -p "ROOT is installed. Press 1 to start the ROOT analysis: " choice_root
        elif [ $software_status -eq 2 ]; then
            read -p "Python is installed. Press 2 to start the Python analysis: " choice_root
        fi

        case $choice_root in
            1)
                echo "Running the ROOT program..."
                cd TMVA_ML
                root -l TMVA_Classification.C
                cd ..
                if [ $software_status -eq 3 ]; then
                    while true; do
                        read -p "Do you want to start the Python program now? (y/n): " rerun_py
                        case $rerun_py in
                            [yY]* )
                                echo "Running the Python program..."
                                cd Python_code
                                python3 Program_Start.py
                                cd ..
                                echo "You have completed the analysis. Goodbye!"
                                exit;;
                            [nN]* )
                                echo "Goodbye!"
                                exit;;
                            * )
                                echo "Invalid input. Please respond with 'y' or 'n'."
                                ;;
                        esac
                    done
                else
                    echo "You have completed the analysis. Goodbye!"
                    exit
                fi
                ;;
            2)
                echo "Running the Python program..."
                cd Python_code
                python3 Program_Start.py
                cd ..
                if [ $software_status -eq 3 ]; then
                    while true; do
                        read -p "Do you want to start the ROOT program now? (y/n): " rerun_root
                        case $rerun_root in
                            [yY]* )
                                    echo "Running the ROOT program..."
                                    cd TMVA_ML
                                    root -l TMVA_Classification.C
                                    cd ..
                                    echo "You have completed the analysis. Goodbye!"
                                    exit;;
                            [nN]* )
                                    echo "Goodbye!"
                                    exit;;
                                * )
                                    echo "Invalid input. Please respond with 'y' or 'n'."
                                    ;;
                        esac
                    done
                else
                    echo "You have completed the analysis. Goodbye!"
                    exit
                fi
                ;;
            B)
                if [ $software_status -eq 3 ]; then
                    echo "Running the ROOT program..."
                    cd TMVA_ML
                    root -l TMVA_Classification.C
                    cd ..
                    echo "Running the Python program..."
                    cd Python_code
                    python3 Program_Start.py
                    cd ..
                    echo "You have completed the analysis. Goodbye!"
                    exit
                else
                    echo "Invalid selection."
                fi
                ;;
            *)
                echo "Invalid selection."
                ;;
        esac
    done
}

# Execute the main function
main
