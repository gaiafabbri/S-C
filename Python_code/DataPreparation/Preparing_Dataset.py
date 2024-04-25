from .PCA import find_optimal_num_components, apply_pca
from .Data_Preparation import apply_shuffle


# The basic idea is to apply PCA to daset independentely from the model you choose, that's the reason for this function, which purpose is to hande with data making PCA and shuffle
def Control_PCA (X, y, width, height, choice):
    X_new = None  # Initialise X_PCA to None
    n_principal_components = None  # Also initialise n_principal_components to None

    if choice == "1" or choice == "2" or choice == "4":
        # Desired variance
        target_variance_explained = 0.95
        
        # Computing the number of principal components
        n_principal_components=find_optimal_num_components(X,0.95, width, height)
        print("Number of principal components:", n_principal_components)
        
        # Apply PCA
        X_PCA = apply_pca(X, n_principal_components)
        
        # Data shuffle
        X_new=apply_shuffle(X_PCA)
        y_new=apply_shuffle(y)
        
    else: #otherwise do nothing
        
        X_new  = X
        y_new = y
        
    
    return X_new,y_new, n_principal_components
