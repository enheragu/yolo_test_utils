
#!/usr/bin/env python3
# encoding: utf-8
"""
    Creates different approachs of mixing RGB with Thermal images:
    UMAP:
    t-SNE:
    MDS:
    All three methods seem to be impracticable in terms of time consumed or even memory consumed.

    Finished fit transform
    UMAP output min: -16.170766830444336, max: 25.416061401367188
    UMAP denormalized min: 0.0, max: 1.0
    Output image min: 0, max: 255
    UMAP fussion method for one image took 0h 30min 10.74ss
    UMAP image shape: (512, 640, 3)

    Generate TSNE reducer
    Fit transform
    Finished fit transform
    TSNE fussion method for one image took 0h 42min 39.01ss
    TSNE image shape: (512, 640, 3)

    Generate MDS reducer
    Fit transform
    Traceback (most recent call last):
        numpy.core._exceptions._ArrayMemoryError: Unable to allocate 800. GiB for an array with shape (327680, 327680) and data type float64
    
"""

import os
import time
import numpy as np
import cv2 as cv

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE, MDS
import umap


# Small hack so packages can be found
if __name__ == "__main__":
    import sys
    sys.path.append('./src')

from utils import log, bcolors
from Dataset.decorators import time_execution_measure, save_image_if_path, save_npmat_if_path



def getFourChannelImage(visible_image, thermal_image):
    b, g, r = cv.split(visible_image)
    th_channel = thermal_image
    ch4_image = cv.merge([b, g, r, th_channel])
    
    height, width, channels = ch4_image.shape
    ch4_image_reshaped = ch4_image.reshape(-1, channels)

    # Standarize data
    ch4_image_reshaped_std = (ch4_image_reshaped - np.mean(ch4_image_reshaped, axis=0)) / (np.std(ch4_image_reshaped, axis=0) + 1e-8)

    return ch4_image_reshaped_std, height, width

def getThreeChannelImageShaped(ch3_shaped, height, width, n_channels_out = 3):
    X_reshaped = ch3_shaped.reshape(height, width, n_channels_out)
    if abs(X_reshaped.min()-X_reshaped.max()) < 0.01:
        print(f"[ERROR] [getThreeChannelImageShaped] Min: {X_reshaped.min()}, Max: {X_reshaped.max()}")

    X_normalized = ((X_reshaped - X_reshaped.min()) / (X_reshaped.max() - X_reshaped.min()) * 255).astype(np.uint8)
    return X_normalized

"""
    n_neighbors: Controla el equilibrio entre la estructura local y global.
    min_dist: Influye en la densidad de los puntos proyectados.
    metric: La métrica de distancia a utilizar.
"""
@save_image_if_path
def combine_rgbt_umap_to3ch(visible_image, thermal_image, n_channels_out=3):
    ch4_image, height, width = getFourChannelImage(visible_image, thermal_image)

    print(f"Generate UMAP reducer")
    umap_reducer = umap.UMAP(n_components=n_channels_out, n_neighbors=15, min_dist=0.5, metric='euclidean', transform_queue_size=10, n_jobs=-1)
    print(f"Fit transform")
    X_umap = umap_reducer.fit_transform(ch4_image)
    print(f"Finished fit transform")
    # X_umap_denormalized = X_umap * np.std(ch4_image[:, :n_channels_out], axis=0) + np.mean(ch4_image[:, :n_channels_out], axis=0)
    X_umap_scaled = (X_umap - X_umap.min()) / (X_umap.max() - X_umap.min())
    image_normalized = getThreeChannelImageShaped(X_umap_scaled, height, width, n_channels_out)

    print(f"UMAP output min: {X_umap.min()}, max: {X_umap.max()}")
    print(f"UMAP denormalized min: {X_umap_scaled.min()}, max: {X_umap_scaled.max()}")
    print(f"Output image min: {image_normalized.min()}, max: {image_normalized.max()}")

    return image_normalized   


"""
    perplexity: Equilibra la atención entre las características locales y globales (con valores más altos, se simplifica la relación entre los puntos).
    learning_rate: Controla la velocidad de aprendizaje.
"""
@save_image_if_path
def combine_rgbt_tsne_to3ch(visible_image, thermal_image, n_channels_out=3):
    ch4_image, height, width = getFourChannelImage(visible_image, thermal_image)
    print(f"Generate TSNE reducer")
    tsne = TSNE(n_components=n_channels_out, perplexity=30, learning_rate=500, max_iter=250, n_jobs=-1)
    print(f"Fit transform")
    X_tsne = tsne.fit_transform(ch4_image)
    print(f"Finished fit transform")
    X_tsne_denormalized = X_tsne * np.std(ch4_image[:, :n_channels_out], axis=0) + np.mean(ch4_image[:, :n_channels_out], axis=0)
    image_normalized = getThreeChannelImageShaped(X_tsne_denormalized, height, width, n_channels_out)
    return image_normalized   


"""
    metric: Booleano que indica si usar MDS métrico o no métrico.
    n_init: Número de ejecuciones con diferentes inicializaciones aleatorias.
"""
@save_image_if_path
def combine_rgbt_mds_to3ch(visible_image, thermal_image, n_channels_out=3):
    ch4_image, height, width = getFourChannelImage(visible_image, thermal_image)
    print(f"Generate MDS reducer")
    mds = MDS(n_components=n_channels_out, metric=False, n_init=4)
    print(f"Fit transform")
    X_mds = mds.fit_transform(ch4_image)
    print(f"Finished fit transform")
    X_mds_denormalized = X_mds * np.std(ch4_image[:, :n_channels_out], axis=0) + np.mean(ch4_image[:, :n_channels_out], axis=0)
    image_normalized = getThreeChannelImageShaped(X_mds_denormalized, height, width, n_channels_out)
    return image_normalized   



# Format seconds time to h min s format
def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}h {m}min {s:.2f}s"

if __name__ == '__main__':
    from scipy.spatial.distance import pdist, squareform
    from scipy.stats import spearmanr
    from sklearn.neighbors import NearestNeighbors
    from sklearn.metrics import pairwise_distances
    import matplotlib.pyplot as plt
    
    test_umap = False
    test_tsne = False
    test_mds = True
    print(f"Configuration: {test_umap = }; {test_tsne = }; {test_mds = }")

    # example_image
    lwir_image_path = "/home/arvc/eeha/kaist-yolo-annotated/train-day-80_20/lwir/images/set00_V001_I00103.png"
    visible_image_path = lwir_image_path.replace('/lwir/', '/visible/')
    visible_image = cv.imread(visible_image_path)
    lwir_image = cv.imread(lwir_image_path, cv.IMREAD_GRAYSCALE)    

    # visible_image = cv.resize(visible_image, (320, 240))
    # lwir_image = cv.resize(lwir_image, (320, 240))

    ch4_image, height, width = getFourChannelImage(visible_image, lwir_image)

    print(f"Visible image shape: {visible_image.shape}")
    print(f"LWIR image shape: {lwir_image.shape}")
    print(f"CH4 image shape: {ch4_image.shape}")

    def logImageGenerated(method, exec_time, image):
        print(f"{method.upper()} fussion method for one image took {format_time(exec_time)}s")
        print(f"{method.upper()} image shape: {image.shape}")
        cv.imshow(f"{method.upper()} image", image)
        cv.pollKey()

        colors = ['blue', 'green', 'red']
        plt.figure(figsize=(10, 5))
        for i, color in enumerate(colors):
            if image.shape[-1] > i:  # Asegura que existan suficientes canales
                channel = image[:, :, i]
                plt.hist(channel.ravel(), bins=50, color=color, alpha=0.7, label=f'{color.capitalize()} Channel')
        
        plt.title(f'Histogram for {method.upper()} image')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()

    print(f"Combine images from {lwir_image_path.replace('lwir','lwir-visible')} with all three methods")
    if test_umap:
        decorated_function = time_execution_measure(combine_rgbt_umap_to3ch)
        umap_image, umap_time_execution = decorated_function(visible_image, lwir_image)
        logImageGenerated('umap', umap_time_execution, umap_image)
    if test_tsne:
        decorated_function = time_execution_measure(combine_rgbt_tsne_to3ch)
        tsne_image, tsne_time_execution = decorated_function(visible_image, lwir_image)
        logImageGenerated('tsne', tsne_time_execution, tsne_image)
    if test_mds:
        decorated_function = time_execution_measure(combine_rgbt_mds_to3ch)
        mds_image, mds_time_execution = decorated_function(visible_image, lwir_image)
        logImageGenerated('mds', mds_time_execution, mds_image)

    """
    Visualización: Crear gráficos de dispersión 3D para cada método.
    Preservación de distancias: Calcular la correlación entre las distancias en el espacio original y el reducido.
    """
    def evaluate_distance_preservation(X_original, X_reduced):
        X_original_flat = X_original.reshape(-1, X_original.shape[-1])
        X_reduced_flat = X_reduced.reshape(-1, X_reduced.shape[-1])
        dist_original = pairwise_distances(X_original_flat, metric='euclidean')
        dist_reduced = pairwise_distances(X_reduced_flat, metric='euclidean')
        correlation, _ = spearmanr(dist_original, dist_reduced)
        return correlation

    print(f"Evaluate distance preservation of the methods:")
    if test_umap:
        corr_umap = evaluate_distance_preservation(ch4_image, umap_image)
        print(f"\t· Correlación de distancias UMAP: {corr_umap:.4f}")
    if test_tsne:
        corr_tsne = evaluate_distance_preservation(ch4_image, tsne_image)
        print(f"\t· Correlación de distancias t-SNE: {corr_tsne:.4f}")
    if test_mds:
        corr_mds = evaluate_distance_preservation(ch4_image, mds_image)
        print(f"\t· Correlación de distancias MDS: {corr_mds:.4f}")
  


    """
    Reconstrucción del error: Para UMAP, puede usar la función de reconstrucción inversa.
    """
    if test_umap:
        umap_reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean')
        ch4_image_reconstructed = umap_reducer.inverse_transform(umap_image)
        reconstruction_error = np.mean(np.sum((ch4_image - ch4_image_reconstructed)**2, axis=1))

        print(f"\nError de reconstrucción UMAP: {reconstruction_error:.4f}")

    """
    Preservación de vecinos: Calcular cuántos de los k vecinos más cercanos en el espacio original se mantienen en el espacio reducido.
    """
    def neighbor_preservation(X_original, X_reduced, k=10):
        X_original_flat = X_original.reshape(-1, X_original.shape[-1])
        X_reduced_flat = X_reduced.reshape(-1, X_reduced.shape[-1])
        nn_original = NearestNeighbors(n_neighbors=k).fit(X_original_flat)
        nn_reduced = NearestNeighbors(n_neighbors=k).fit(X_reduced_flat)
        
        _, indices_original = nn_original.kneighbors(X_original_flat)
        _, indices_reduced = nn_reduced.kneighbors(X_reduced_flat)
        
        preservation = np.mean([len(set(orig) & set(red)) / k 
                                for orig, red in zip(indices_original, indices_reduced)])
        return preservation

    print(f"Evaluate neighbor preservation with all three methods:")
    if test_umap:
        preservation_umap = neighbor_preservation(ch4_image, umap_image)
        print(f"\t· Preservación de vecinos UMAP: {preservation_umap:.4f}")
    if test_tsne:
        preservation_tsne = neighbor_preservation(ch4_image, tsne_image)
        print(f"\t· Preservación de vecinos t-SNE: {preservation_tsne:.4f}")
    if test_mds:
        preservation_mds = neighbor_preservation(ch4_image, mds_image)
        print(f"\t· Preservación de vecinos MDS: {preservation_mds:.4f}")

    
    """
    Estas métricas le ayudarán a evaluar cuánta información preserva cada método. Un valor más alto en la correlación de distancias y la preservación de vecinos indica una mejor preservación de la estructura original de los datos
    """

    # Visualizar resultados
    methods = []
    correlations = []
    preservations = []
    if test_umap: 
        methods.append('UMAP')
        correlations.append(corr_umap)
        preservations.append(preservation_umap)
    if test_tsne: 
        methods.append('t-SNE')
        correlations.append(corr_tsne)
        preservations.append(preservation_tsne)
    if test_mds: 
        methods.append('MDS')
        correlations.append(corr_mds)
        preservations.append(preservation_mds)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.bar(methods, correlations)
    ax1.set_title('Correlación de distancias')
    ax1.set_ylim(0, 1)

    ax2.bar(methods, preservations)
    ax2.set_title('Preservación de vecinos')
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    plt.show()

    cv.destroyAllWindows()

