
#!/usr/bin/env bash

export NTFY_TOPIC="eeha_training_test_battery"

## Path of current file
SOURCE=${BASH_SOURCE[0]}
while [ -L "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
DIR=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )
SOURCE=$(readlink "$SOURCE")
[[ $SOURCE != /* ]] && SOURCE=$DIR/$SOURCE # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
export EEHA_SCHEDULER_SCRIPTFILE_PATH=$SOURCE
export EEHA_SCHEDULER_SCRIPT_PATH=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )

function eeha_env(){
    source $EEHA_SCHEDULER_SCRIPT_PATH/../../venv/bin/activate
}


eeha_env


# Advanced early fusion methods:
python $EEHA_SCHEDULER_SCRIPT_PATH/../src/gui_render_preset.py --preset gui_presets/advanced_early_fusion/kaist_day_all_fusion_methods.yaml --output /home/quique/umh/latex_docs/07_Advanced_early_fusion_thermal_visible/data_and_code/results/kaist_day/ --load_from_cache

python $EEHA_SCHEDULER_SCRIPT_PATH/../src/gui_render_preset.py --preset gui_presets/advanced_early_fusion/kaist_night_all_fusion_methods.yaml --output /home/quique/umh/latex_docs/07_Advanced_early_fusion_thermal_visible/data_and_code/results/kaist_night/ --load_from_cache

python $EEHA_SCHEDULER_SCRIPT_PATH/../src/gui_render_preset.py --preset gui_presets/advanced_early_fusion/llvip_night_all_fusion_methods.yaml --output /home/quique/umh/latex_docs/07_Advanced_early_fusion_thermal_visible/data_and_code/results/llvip_night/ --load_from_cache

# Focused subsets (top-6 by mAP50 → PR curve comparison):
python $EEHA_SCHEDULER_SCRIPT_PATH/../src/gui_render_preset.py --preset gui_presets/advanced_early_fusion/kaist_day_focus_fusion_methods.yaml --output /home/quique/umh/latex_docs/07_Advanced_early_fusion_thermal_visible/data_and_code/results/kaist_day/ --load_from_cache

python $EEHA_SCHEDULER_SCRIPT_PATH/../src/gui_render_preset.py --preset gui_presets/advanced_early_fusion/kaist_night_focus_fusion_methods.yaml --output /home/quique/umh/latex_docs/07_Advanced_early_fusion_thermal_visible/data_and_code/results/kaist_night/ --load_from_cache

python $EEHA_SCHEDULER_SCRIPT_PATH/../src/gui_render_preset.py --preset gui_presets/advanced_early_fusion/llvip_night_focus_fusion_methods.yaml --output /home/quique/umh/latex_docs/07_Advanced_early_fusion_thermal_visible/data_and_code/results/llvip_night/ --load_from_cache

# CSV with all data
python $EEHA_SCHEDULER_SCRIPT_PATH/../src/gui_render_preset.py --preset gui_presets/full_table_dump.yaml --output /home/quique/umh/latex_docs/07_Advanced_early_fusion_thermal_visible/data_and_code/results/ --load_from_cache