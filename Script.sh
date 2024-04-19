#!/bin/bash

# Funzione per controllare la versione di ROOT
check_root_version() {
    root_version=$(root-config --version 2>/dev/null)
    if [ $? -eq 0 ]; then
        echo "ROOT version: $root_version"
        required_version="6.20/02"
        if [ "$root_version" != "$required_version" ]; then
            echo "Attenzione: La tua versione di ROOT ($root_version) non è compatibile ($required_version richiesta). Aggiorna o usa il dockerfile" #aggiustare versione
            exit 1
        fi
    else
        echo "ROOT non trovato sul sistema."
        exit 1
    fi
}

# Funzione per controllare la presenza e la versione di Python
check_python() {
    echo "Verifica della versione di Python:"
    python_version=$(python3 --version 2>&1)
    if [ $? -eq 0 ]; then
        echo "Python version: $python_version"
        required_python_version="3.11.6"
        if [ "$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')" != "$required_python_version" ]; then
            echo "La tua versione di Python non è compatibile ($required_python_version richiesta)."
            echo "Si consiglia di utilizzare un Dockerfile per gestire l'ambiente."
            exit 1
        fi
    else
        echo "Python non trovato sul sistema. Impossibile procedere. Usare il Dockerfile."
        exit 1
    fi
}

update_python_libraries() {
    echo "Verifica delle librerie Python e compatibilità:"
    incompatible_libraries=()

    check_library() {
        local lib_name=$1
        local required_version=$2
        local lib_version=$(python3 -c "import $lib_name; print($lib_name.__version__)" 2>/dev/null)

        if [ $? -eq 0 ]; then
            if dpkg --compare-versions "$lib_version" lt "$required_version"; then
                incompatible_libraries+=("$lib_name (versione attuale: $lib_version, versione richiesta: $required_version)")
            elif dpkg --compare-versions "$lib_version" gt "$required_version"; then
                while true; do
                    read -p "Hai una versione più recente di $lib_name ($lib_version) rispetto a quella richiesta ($required_version). Vuoi aggiornare alla versione richiesta? (s/n): " update_to_required_version
                    case $update_to_required_version in
                        [sS]* )
                            echo "Aggiornamento di $lib_name alla versione richiesta..."
                            pip3 install --upgrade $lib_name==$required_version
                            echo "$lib_name aggiornato con successo alla versione $required_version!"
                            break;;
                        [nN]* )
                            incompatible_libraries+=("$lib_name (versione attuale: $lib_version, versione richiesta: $required_version)")
                            break;;
                        * )
                            echo "Risposta non valida. Per favore, rispondi con 's' o 'n'.";;
                    esac
                done
            fi
        else
            while true; do
                read -p "$lib_name non trovato. Vuoi installarlo? (s/n): " install_lib
                case $install_lib in
                    [sS]* )
                        echo "Installazione di $lib_name..."
                        pip3 install $lib_name==$required_version
                        echo "$lib_name installato con successo!"
                        break;;
                    [nN]* )
                        echo "Si consiglia di utilizzare un Dockerfile per gestire l'ambiente."
                        exit 1;;
                    * )
                        echo "Risposta non valida. Per favore, rispondi con 's' o 'n'.";;
                esac
            done
        fi
    }

    check_library "pandas" "2.2.1"
    check_library "numpy" "1.26.4"
    check_library "tensorflow" "2.16.1"
    check_library "keras" "3.1.1"
    check_library "torch" "2.2.2+cu121"
    check_library "uproot" "5.3.2"

    if [ ${#incompatible_libraries[@]} -gt 0 ]; then
        echo "Le seguenti librerie non sono compatibili con le versioni richieste o non sono installate:"
        printf '%s\n' "${incompatible_libraries[@]}"
        while true; do
            read -p "Vuoi aggiornare queste librerie? (s/n): " update_libraries
            case $update_libraries in
                [sS]* )
                    echo "Aggiornamento delle librerie..."
                    for lib_info in "${incompatible_libraries[@]}"; do
                        lib_name=$(echo "$lib_info" | cut -d' ' -f1)
                        echo "Aggiornamento di $lib_name..."
                        pip3 install --upgrade $lib_name
                        echo "$lib_name aggiornato con successo!"
                    done
                    break;;
                [nN]* )
                    echo "Continuando senza aggiornare le librerie."
                    break;;
                * )
                    echo "Risposta non valida. Per favore, rispondi con 's' o 'n'.";;
            esac
        done
    else
        echo "Tutte le librerie sono compatibili."
    fi
}


check_pip() {
    echo "Verifica della versione di pip:"
    pip_version=$(pip3 --version | cut -d ' ' -f 2)
    if [ $? -eq 0 ]; then
        echo "pip version: $pip_version"
        required_pip_version="24.0"
        if [ "$pip_version" != "$required_pip_version" ]; then
            while true; do
                read -p "La tua versione di pip non è $required_pip_version. Vuoi aggiornare pip? (s/n): " update_pip
                case $update_pip in
                    [sS]* )
                        echo "Aggiornamento di pip..."
                        python3 -m pip install --upgrade pip==$required_pip_version
                        echo "pip aggiornato con successo alla versione $required_pip_version!"
                        break;;
                    [nN]* )
                        echo "Si consiglia di utilizzare un Dockerfile per gestire l'ambiente."
                        exit 1;;
                    * )
                        echo "Risposta non valida. Per favore, rispondi con 's' o 'n'.";;
                esac
            done
        fi
    else
        echo "pip non trovato sul sistema. Impossibile procedere. Usare il Dockerfile."
        exit 1
    fi
}

check_root_file() {
    local tmva_ml_folder="TMVA_ML/images"
    local Python_code_folder="Python_code/images"

    echo "Verifica della presenza di file ROOT nella cartella '$tmva_ml_folder':"
    tmva_ml_root_files=$(find "$tmva_ml_folder" -maxdepth 1 -type f -iname '*.root')
    if [ -n "$tmva_ml_root_files" ]; then
        echo "File ROOT trovati nella cartella '$tmva_ml_folder':"
        echo "$tmva_ml_root_files"
    else
        echo "Nessun file ROOT trovato nella cartella '$tmva_ml_folder'."
        exit 1
    fi

    echo "Verifica della presenza di file ROOT nella cartella '$Python_code_folder':"
    Python_code_root_files=$(find "$Python_code_folder" -maxdepth 1 -type f -iname '*.root')
    if [ -n "$Python_code_root_files" ]; then
        echo "File ROOT trovati nella cartella '$Python_code_folder':"
        echo "$Python_code_root_files"
    else
        echo "Nessun file ROOT trovato nella cartella '$Python_code_folder'."
        exit 1
    fi
}



# Funzione per generare il comando di eliminazione
generate_delete_command() {
    local folder_path="$1"
    eval "rm -rf \"$folder_path\""
}

Gen_files(){
    while true; do
        # Richiesta all'utente se preferisce generare il dataset o utilizzare uno già fatto
        read -p "Vuoi generare il dataset o utilizzare uno già fatto? (generare/utilizzare): " choice
        case $choice in
            generare)
                echo "Generiamo il dataset, ti chiederò di scegliere le dimensioni delle immagini, il numero di dati per il file di bkg e sig, e la distribuzione di dati:"
                
                while true; do
                    read -p "Inserisci il numero di dati (n): " n
                    if [[ $n =~ ^[0-9]+$ ]]; then
                        break
                    else
                        echo "Errore: inserisci un numero valido."
                    fi
                done
                while true; do  # Aggiunto 'do' qui
                    # Chiedi all'utente di inserire l'altezza delle immagini (nh) e controlla se è un numero
                    while true; do
                        read -p "Inserisci l'altezza delle immagini compresa tra 8 e 24 (nh): " nh
                        if [[ $nh =~ ^[0-9]+$ && $nh -ge 8 && $nh -le 24 ]]; then
                            break
                        else
                            echo "Errore: inserisci un numero valido."
                        fi
                    done

                    # Chiedi all'utente di inserire la larghezza delle immagini (nw) e controlla se è un numero
                    while true; do
                        read -p "Inserisci la larghezza delle immagini compresa tra 8 e 24 (nw): " nw
                        if [[ $nw =~ ^[0-9]+$ && $nw -ge 8 && $nw -le 24 ]]; then
                            break
                        else
                            echo "Errore: inserisci un numero valido."
                        fi
                    done
                     if [[ $(($nh * $nw)) -ge 64 && $(($nh * $nw)) -le 576 ]]; then
                        break
                    else
                        echo "Errore: il prodotto di nh * nw deve essere compreso tra 64 e 576."
                    fi
                done
                 # Esegui il comando ROOT per generare il dataset
                root -l -q "ROOT/Gauss.C($n, $nh, $nw)"
                break  # Esci dal ciclo while esterno dopo aver completato la scelta
                ;;
            utilizzare)
                echo "Scaricamento del dataset già fatto..."
                # Crea la cartella "images" se non esiste già
                mkdir -p ROOT/images
                # Scarica il file del dataset già fatto e salvalo nella cartella "images"
                wget -P ROOT/images "link_del_tuo_dataset_già_fatto"
                break  # Esci dal ciclo while esterno dopo aver completato la scelta
                ;;
            *)
                echo "Selezione non valida. Riprova."
                ;;
        esac
    done
}



# Funzione per cancellare tutte le cartelle '__pycache__' dalle sottocartelle di una cartella
delete_all_pycache_folders() {
    local parent_folder="$1"
    
    # Utilizza il comando 'find' per trovare tutte le cartelle '__pycache__' e cancellarle
    find "$parent_folder" -type d -name "__pycache__" -exec rm -rf {} +
    }

move_images_folders() {
    # Primo obiettivo: Muovere la cartella "images" fuori da "ROOT"
    mv "ROOT/images" "./"

    # Secondo obiettivo: Fare una copia e spostarla dentro a "TMVA_ML"
    cp -r "images" "TMVA_ML/"

    # Terzo obiettivo: Fare una copia e spostarla dentro a "Python_code"
    cp -r "images" "Python_code/"

    # Quarto obiettivo: Cancella "images" al livello "zero"
    rm -rf "images"
}





main() {
    generate_delete_command "ROOT/images"
    generate_delete_command "Python_code/images"
    generate_delete_command "TMVA_ML/images"
    generate_delete_command "Python_code/plot_results"
    # Esegui la funzione specificando la cartella principale dalla quale vuoi cancellare le sottocartelle '__pycache__'
    delete_all_pycache_folders "Python_code"

    check_python
    check_root_version
    check_pip
    
    
    # Se tutte le verifiche sono state superate senza richiedere l'uso del Dockerfile
    if [ -n "$python_version" ] && [ "$root_version" = "6.20/02" ] && [ "$pip_version" = "24.0" ]; then
        while true; do
            read -p "Diamo un'occhiata alle librerie di Python per vedere se sono disponibili? (s/n): " check_libraries
            case $check_libraries in
                [sS]* )
                    update_python_libraries
                    break;;
                [nN]* )
                    echo "Occhio però che il codice è stato testato con le seguenti librerie di python: pandas 2.2.1, numpy 1.26.4, tensorflow 2.16.1, keras 3.1.1, torch 2.2.2, uproot 5.3.2, scikit-learn 1.4.1"
                    break;;
                * )
                    echo "Risposta non valida. Si prega di rispondere con 's' o 'n'."
            esac
        done
        # Continua con il resto del codice...

        echo "Bene, possiamo cominciare."
        Gen_files
        move_images_folders
        read -p "Vuoi cominciare il programma ROOT? (1 = sì, 2 = no, preferisco cominciare con Python, 3 = esegui entrambi): " choice_root
        check_root_file
        case $choice_root in
            1)
                echo "Esecuzione del programma ROOT..."
                root -l -q Code/prova_TMVA.C
                read -p "Vuoi avviare ora il programma Python? (s/n): " rerun_py
                if [ "$rerun_py" = "s" ]; then
                    echo "Esecuzione del programma Python..."
                    python3 Code/CNN_pc.py #path della cartella
                fi
                ;;
            2)
                echo "Esecuzione del programma Python..."
                python3 Code/CNN_pc.py #path della cartella
                read -p "Vuoi avviare ora il programma in ROOT? (s/n): " rerun_root
                if [ "$rerun_root" = "s" ]; then
                    echo "Esecuzione del programma ROOT..."
                    root -l -q Code/prova_TMVA.C #path della cartella
                fi
                ;;
            3) 
                echo "Esecuzione del programma ROOT..."
                root -l -q Code/prova_TMVA.C
                echo "Esecuzione del programma Python..."
                python3 Code/CNN_pc.py #path della cartella
                ;;
            *)
                echo "Selezione non valida. Saltando l'esecuzione."
                ;;
        esac
    fi
}

# Esecuzione della funzione principale
main

