
# Hack for fusion in less time that generates both versions at the same time. Need to manually update symlinks to images
#!/bin/bash

### OLD VERSION
# cd /home/arvc/eeha/llvip-yolo-annotated/train-night-80_20/wavelet/images
# mkdir -p "/home/arvc/eeha/llvip-yolo-annotated/train-night-80_20/wavelet_max/images/"
# for link in *; do
#     if [ -L "$link" ]; then
#         target=$(readlink -f "$link")
#         ln -s "$target" "/home/arvc/eeha/llvip-yolo-annotated/train-night-80_20/wavelet_max/images/$link"
#     fi
# done

# ln -s /home/arvc/eeha/llvip-yolo-annotated/test-night-80_20/wavelet/labels /home/arvc/eeha/llvip-yolo-annotated/test-night-80_20/wavelet_max/labels
# ln -s /home/arvc/eeha/llvip-yolo-annotated/train-night-80_20/wavelet/labels /home/arvc/eeha/llvip-yolo-annotated/train-night-80_20/wavelet_max/labels
# ln -s /home/arvc/eeha/llvip-yolo-annotated/train-night-half1-80_20/wavelet/labels /home/arvc/eeha/llvip-yolo-annotated/train-night-half1-80_20/wavelet_max/labels
# ln -s /home/arvc/eeha/llvip-yolo-annotated/train-night-half2-80_20/wavelet/labels /home/arvc/eeha/llvip-yolo-annotated/train-night-half2-80_20/wavelet_max/labels


# Define tu directorio raíz aquí
ROOTS=(
    "/home/arvc/eeha/llvip-yolo-annotated"
    "/home/arvc/eeha/kaist-yolo-annotated"
)

TYPES=("wavelet" "curvelet")

## GENERATE MISSING _max folders
for ROOT in "${ROOTS[@]}"; do
    for TYPE in "${TYPES[@]}"; do
        # Buscar todas las carpetas del tipo actual
        find "$ROOT" -type d -name "$TYPE" | while read type_dir; do
            # Ruta del directorio _max asociado
            type_max_dir="${type_dir}_max"
            images_dir="$type_dir/images"
            images_max_dir="$type_max_dir/images"

            # Solo si existe el original y NO existe el _max
            if [ -d "$images_dir" ] && [ ! -d "$type_max_dir" ]; then
                mkdir -p "$images_max_dir"
                echo "Created new folder $images_max_dir." 

                # Enlazar los archivos/symlinks:
                for file in "$images_dir"/*; do
                    [ -e "$file" ] || continue
                    filename=$(basename "$file")
                    dest="$images_max_dir/$filename"
                    if [ -L "$file" ]; then
                        real_target=$(readlink -f "$file")
                        real_target_max=${real_target/$TYPE/"${TYPE}_max"}
                        ln -s "$real_target_max" "$dest"
                    else
                        max_file_name=${file/$TYPE/"${TYPE}_max"}
                        ln -s "$max_file_name" "$dest"
                    fi
                done
                echo "All images were linked to $images_max_dir."
            fi
        done
    done
done

## LINK LABELS FOLDERS
for ROOT in "${ROOTS[@]}"; do
    for TYPE in "${TYPES[@]}"; do
        MAX_DIRNAME="${TYPE}_max"
        # Busca carpetas *_max
        find "$ROOT" -type d -name "$MAX_DIRNAME" | while read max_dir; do
            # Verifica si ya hay un enlace/carpeta de labels
            if [ ! -e "$max_dir/labels" ]; then
                # Construye el nombre del origen tipo "wavelet" o "curvelet"
                original_dir="${max_dir/$MAX_DIRNAME/$TYPE}"
                if [ -d "$original_dir/labels" ]; then
                    echo ln -s "$original_dir/labels" "$max_dir/labels"
                    ln -s "$original_dir/labels" "$max_dir/labels"
                    echo "Linked labels from: $original_dir/labels to $max_dir/labels."
                else
                    echo "Original to link does not exist: $original_dir/labels."
                fi
            fi
        done
    done
done