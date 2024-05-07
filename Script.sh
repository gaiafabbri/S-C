#!/bin/bash

#######################################
# Function to check the ROOT version #
#######################################

check_root_version() {
    root_version=$(root-config --version 2>/dev/null)
    if [ $? -eq 0 ]; then
        echo "ROOT version: $root_version"
        required_version="6.30.06"
        if [ "$root_version" != "$required_version" ]; then
            echo "Warning: Your ROOT version ($root_version) is not compatible (required: $required_version). Please update or use the Dockerfile."
            exit 1
        fi
    else
        echo "ROOT not found on the system."
        exit 1
    fi
}

####################################
# Function to check Python version #
####################################

check_python() {
    echo "Checking Python version:"
    python_version=$(python3 --version 2>&1)
    if [ $? -eq 0 ]; then
        echo "Python version: $python_version"
        required_python_version="3.11.5"
        if [ "$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')" != "$required_python_version" ]; then
            echo "Your Python version is not compatible (required: $required_python_version)."
            echo "Consider using a Dockerfile to manage the environment."
            exit 1
        fi
    else
        echo "Python not found on the system. Unable to proceed. Please use the Dockerfile."
        exit 1
    fi
}


############################################
# Function to update Python libraries     #
# and ensure compatibility                #
############################################

update_python_libraries() {
    echo "Checking Python libraries and compatibility:"
    incompatible_libraries=()

    check_library() {
        local lib_name=$1
        local required_version=$2
        local lib_version=$(python3 -c "import $lib_name; print($lib_name.__version__)" 2>/dev/null)

        if [ $? -eq 0 ]; then
            # Use packaging.version to compare library versions
            if [ "$(python3 -c "from packaging.version import parse; print(parse('$lib_version') < parse('$required_version'))")" = "True" ]; then
                incompatible_libraries+=("$lib_name (current version: $lib_version, required version: $required_version)")
            elif [ "$(python3 -c "from packaging.version import parse; print(parse('$lib_version') > parse('$required_version'))")" = "True" ]; then
                read -p "You have a newer version of $lib_name ($lib_version) compared to the required one ($required_version). Do you want to update to the required version? (y/n): " update_to_required_version
                if [ "$update_to_required_version" = "y" ]; then
                    echo "Updating $lib_name to the required version..."
                    pip3 install --upgrade $lib_name==$required_version
                    echo "$lib_name successfully updated to version $required_version!"
                else
                    incompatible_libraries+=("$lib_name (current version: $lib_version, required version: $required_version)")
                fi
            fi
        else
            incompatible_libraries+=("$lib_name (not installed)")
            read -p "$lib_name not found. Do you want to install it? (y/n): " install_lib
            if [ "$install_lib" = "y" ]; then
                echo "Installing $lib_name..."
                pip3 install $lib_name==$required_version
                echo "$lib_name successfully installed!"
            else
                echo "Consider using a Dockerfile to manage the environment."
                exit 1
            fi
        fi
    }

    check_library "pandas" "2.2.1"
    check_library "numpy" "1.26.4"
    check_library "tensorflow" "2.16.1"
    check_library "keras" "3.3.3"
    check_library "torch" "2.2.2"
    check_library "uproot" "5.3.2"

    if [ ${#incompatible_libraries[@]} -gt 0 ]; then
        echo "The following libraries are either incompatible with the required versions or not installed:"
        printf '%s\n' "${incompatible_libraries[@]}"
        read -p "Do you want to update these libraries? (y/n): " update_libraries
        if [ "$update_libraries" = "y" ]; then
            echo "Updating libraries..."
            for lib_info in "${incompatible_libraries[@]}"; do
                lib_name=$(echo "$lib_info" | cut -d' ' -f1)
                echo "Updating $lib_name..."
                pip3 install --upgrade $lib_name
                echo "$lib_name successfully updated!"
            done
        else
            echo "Continuing without updating libraries."
        fi
    else
        echo "All libraries are compatible."
    fi
}






##################################
# Function to check the pip version #
##################################

check_pip() {
    echo "Checking pip version:"
    pip_version=$(pip3 --version | cut -d ' ' -f 2)
    if [ $? -eq 0 ]; then
        echo "pip version: $pip_version"
        required_pip_version="24.0"
        if [ "$pip_version" != "$required_pip_version" ]; then
            while true; do
                read -p "Your pip version is not $required_pip_version. Do you want to update pip? (y/n): " update_pip
                case $update_pip in
                    [yY]* )
                        echo "Updating pip..."
                        python3 -m pip install --upgrade pip==$required_pip_version
                        echo "pip successfully updated to version $required_pip_version!"
                        break;;
                    [nN]* )
                        echo "Consider using a Dockerfile to manage the environment."
                        exit 1;;
                    * )
                        echo "Invalid response. Please respond with 'y' or 'n'.";;
                esac
            done
        fi
    else
        echo "pip not found on the system. Unable to proceed. Please use the Dockerfile."
        exit 1
    fi
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
        read -p "Do you want to generate or download a dataset? Please check the filename in the code! (g = generate/d = download): " choice
        case $choice in
            [Gg]* ) root -l -q "ROOT_Gen/Generation.C(100000, 16, 16)"; break;;
            [Dd]* )
                mkdir -p ROOT_Gen/images
                wget "https://drive.google.com/uc?export=download&id=1U3NjuMTeNWjFe9Rgen64FauayAMxZTel" -O ROOT_Gen/images/images_data_16x16_10000.root
                break;;
            * ) echo "Please choose g to generate or d to download.";;
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
    # First objective: Move the "images" folder out of "ROOT"
    mv "ROOT_Gen/images" "./"

    # Second objective: Make a copy and move it into "TMVA_ML"
    cp -r "images" "TMVA_ML/"

    # Third objective: Make a copy and move it into "Python_code"
    cp -r "images" "Python_code/"

    # Fourth objective: Delete "images" at the top level
    rm -rf "images"
}


##################
# Main function #
##################

main() {
    generate_delete_command "ROOT_Gen/images"
    generate_delete_command "Python_code/images"
    generate_delete_command "TMVA_ML/images"
    generate_delete_command "Python_code/plot_results"
    generate_delete_command "dataset"
    delete_all_pycache_folders "Python_code"
    rm "TMVA_CNN_ClassificationOutput.root"

    check_python
    check_root_version
    check_pip

    # If all checks passed without requiring Dockerfile usage
    if [ -n "$python_version" ] && [ -n "$root_version" ] && [ -n "$pip_version" ]; then
        while true; do
            read -p "Do you want to check Python libraries availability? (y/n): " check_libraries
            case $check_libraries in
                [yY]* )
                    update_python_libraries
                    break;;
                [nN]* )
                    echo "Please note that the code has been tested with the following Python libraries: pandas 2.2.1, numpy 1.26.4, tensorflow 2.16.1, keras 3.1.1, torch 2.2.2, uproot 5.3.2, scikit-learn 1.4.1"
                    break;;
                * )
                    echo "Invalid response. Please respond with 'y' or 'n'."
            esac
        done

        echo "Alright, let's start."
        Gen_files
        move_images_folders

        while true; do
            read -p "Do you want to start the ROOT program now? (1 = yes, 2 = no, I prefer starting with Python, 3 = run both): " choice_root
            case $choice_root in
                1)
                    echo "Running the ROOT program..."
                    root -l TMVA_ML/TMVA_Classification.C
                    while true; do
                        read -p "Do you want to start the Python program now? (y/n): " rerun_py
                        case $rerun_py in
                            [yY]* )
                                echo "Running the Python program..."
                                python3 Python_code/Program_Start.py
                                break;;
                            [nN]* )
                                echo "Goodbye!"
                                exit;;
                            * )
                                echo "Invalid input. Please respond with 'y' or 'n'."
                                ;;
                        esac
                    done
                    ;;
                2)
                    echo "Running the Python program..."
                    python3 Python_code/Program_Start.py
                    while true; do
                        read -p "Do you want to start the ROOT program now? (y/n): " rerun_root
                        case $rerun_root in
                            [yY]* )
                                    echo "Running the ROOT program..."
                                    root -l TMVA_ML/TMVA_Classification.C
                                    break;;
                            [nN]* )
                                    echo "Goodbye!"
                                    exit;;
                                * )
                                    echo "Invalid input. Please respond with 'y' or 'n'."
                                    ;;
                        esac
                    done
                    ;;
                3)
                    echo "Running the ROOT program..."
                    root -l TMVA_ML/TMVA_Classification.C
                    echo "Running the Python program..."
                    python3 Python_code/Program_Start.py
                    ;;
                *)
                    echo "Invalid selection."
                    ;;
            esac
        done
    fi
}

# Execute the main function
main
