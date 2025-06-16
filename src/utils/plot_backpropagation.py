

import os

from pathlib import Path

import math
import matplotlib.pyplot as plt

import numpy as np

if __name__ == "__main__":
    import sys
    sys.path.append('./src')
    sys.path.append('./src/utils')

from utils import parseYaml
from utils import log, bcolors

cmap_name='turbo'

## Custom color definitions
c_blue = "#0171ba"
c_green = "#78b01c"
c_yellow = "#f6ae2d"
c_red = "#f23535" 
c_purple = "#a66497"
c_grey = "#769393"
c_darkgrey = "#2a2b2e"

# Extended HEX color with 50% transparency (last 80 number)
c_alpha_blue = "#0171ba4D"
c_alpha_green = "#78b01c4D"
c_alpha_yellow = "#f6ae2d4D"
c_alpha_red = "#f235354D"
c_alpha_purple = "#a664974D"
c_alpha_grey = "#7693934D"
c_alpha_darkgrey = "#2a2b2e4D"

yoloCh4_late_v1_subgraphs = {
    # Range does not include the last number :)
    'Backbone_Visible': list(range(1, 11+1)),          # 1-11
    'Backbone_Thermal': list(range(12, 22+1)),         # 12-22
    # Head Visible (bloques funcionales, sin solapamiento)
    'Head_Visible': {#list(range(23, 24+1)),             # 23-23
        'HV': list(range(23, 25+1)),             # 23-24
        'Head_Visible_1': list(range(26, 28+1)),           # 26-27
        'Head_Visible_2': list(range(29, 31+1)),           # 29-30
        'Head_Visible_3': list(range(32, 34+1))           # 32-33
    },
    # Head Thermal (bloques funcionales, sin solapamiento)
    'Head_Thermal': {#list(range(35, 46+1)),             # 35-45
        'HT': list(range(35, 37+1)),             # 35-36
        'Head_Thermal_1': list(range(38, 40+1)),           # 38-39
        'Head_Thermal_2': list(range(41, 43+1)),           # 41-42
        'Head_Thermal_3': list(range(44, 46+1))           # 44-45
    },
    'Detection': list(range(47, 51+1))                 # 47-50
}


def export_colormap_legend(cmap_name=cmap_name, vmin=0, vmax=1, width=6, height=1, filename='colormap_legend.png'):
    fig, ax = plt.subplots(figsize=(width, height))
    
    cmap = plt.get_cmap(cmap_name)
    norm = plt.Normalize(vmin, vmax)
    cb = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=ax, 
        orientation='horizontal'
    )
    
    cb.set_label('Gradient Value', fontsize=14, labelpad=10)
    
    cb.set_ticks([vmin, vmax])
    cb.set_ticklabels(['Low', 'High'])
    cb.ax.tick_params(labelsize=12)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches='tight', transparent=True)
    plt.close(fig)
    log(f"Colormap legend saved as {filename}")




def grad_to_hex(grad, vmin, vmax, cmap_name=cmap_name):
    if math.isnan(grad):
        return c_alpha_darkgrey
    
    elif grad == float('inf'):
        grad = vmax
    elif grad == float('-inf'):
        grad = vmin

    if vmax == vmin:
        vmax = vmin + 1e-8
    ratio = (grad - vmin) / (vmax - vmin)
    ratio = max(0.0, min(1.0, ratio))
    # Obtiene el colormap de matplotlib
    cmap = plt.get_cmap(cmap_name)
    rgba = cmap(ratio)
    # Convierte a HEX
    return '#{:02x}{:02x}{:02x}'.format(int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255))


def extract_layers_and_connections(model_yaml):
    model_cfg = parseYaml(model_yaml)
    layers = []
    connections = []

    idx = 0
    for section in ['backbone', 'head']:
        for layer in model_cfg[section]:
            name = f"{idx}_{layer[2]}"
            layers.append(name)
            froms = layer[0] if isinstance(layer[0], list) else [layer[0]]
            for f in froms:
                if f == -1:
                    if idx > 0:
                        connections.append((layers[idx-1], name))
                else:
                    connections.append((layers[f], name))
            idx += 1
            
    # Return layers, connections, channels and number of classes
    subgraph_dict = {}
    if 'subgraph_config' in model_cfg:
        subgraph_dict = model_cfg['subgraph_config']
    else:
        log("No subgraph_config found in model YAML", bcolors.ERROR)
    return layers, connections, model_cfg['ch'], model_cfg['nc'], subgraph_dict

def get_layer_param_mean(gradients, layer_n):
    prefix = f"model.{layer_n}."
    values = [
        v for k, v in gradients.items()
        if k.startswith(prefix)
        and (k.endswith(".weight") or k.endswith(".bias"))
        and isinstance(v, (float, int))
        and not math.isnan(v)
    ]
    if values:
        return sum(values) / len(values)
    return math.nan

def add_subgraph(diagram, subgraph_name, subgraph_value, layers, gradients, color1=c_alpha_purple, color2=c_purple):
    diagram += f'    subgraph {subgraph_name}["**{subgraph_name.replace("_"," ")}**"]\n'
    
    # If its a dictionary, iterate through its items as its a subgraph again
    if isinstance(subgraph_value, dict):
        for k, v in subgraph_value.items():
            diagram = add_subgraph(diagram, k, v, layers, gradients, c_alpha_grey, c_grey)
    else:
        for idx in subgraph_value:
            if idx < len(layers):
                layer_name = layers[idx].replace('.', '_')
                current_grad = get_layer_param_mean(gradients, idx)
                diagram += f'        %% grad for {layer_name} layer = {current_grad}{" (probably not trainable)" if math.isnan(current_grad) else ""}\n'
                diagram += f'        {layer_name}["{layers[idx].replace("_", ": ")}"]:::blockStyle\n'
    diagram += '    end\n'
    
    if color1 and color2:
        diagram += f'    style {subgraph_name} fill:{color1},stroke:{color2},stroke-width:2px,rx:10px,ry:10px\n'
    return diagram

def extract_all_layer_index(subgraphs):
    idx = set()
    for value in subgraphs.values():
        if isinstance(value, dict):
            idx |= extract_all_layer_index(value)
        else:
            idx |= set(value)
    return idx

def generateMermaidModel(gradients, model_yaml):
    layers, connections, channels, num_classes, subgraph_config = extract_layers_and_connections(model_yaml)
    
    grad_values = []
    for _, v in connections:
        idx = int(v.split('_')[0])
        grad = get_layer_param_mean(gradients, idx)
        grad_values.append(grad)

    filtered_nan_grad_values = [v for v in grad_values if not math.isnan(v) and not math.isinf(v)]
    vmin, vmax = min(filtered_nan_grad_values), max(filtered_nan_grad_values)
    if vmin == vmax:
        vmin, vmax = 0, 1

    # log(f"Gradients min: {vmin}, max: {vmax}")
    # log(f"Gradients: {grad_values}")
    diagram = "%%{init: {'theme':'default'}}%%\ngraph TD\n"

    diagram += '\n    %% Model layer nodes in mermaid format\n'
    diagram += f'    Input["{channels} channel Image"]:::blockStyle'+'@{ shape: st-rect}\n'
    connections.insert(0, ('Input', layers[0]))

    for subgraph_name, subgraph_value in subgraph_config.items():
        diagram = add_subgraph(diagram, subgraph_name, subgraph_value, layers, gradients)

    # Add layers that are not part of any subgraph
    all_subgraph_idx = extract_all_layer_index(subgraph_config)
    for layer in layers:
        idx = int(layer.split('_')[0])
        if idx not in all_subgraph_idx:
            diagram += f'    {layer.replace(".", "_")}[{layer.replace("_", ": ")}]:::blockStyle\n'

    connections_mermaid = []
    connection_format = []
    for idx, (u, v) in enumerate(connections):
        layer_idx = int(v.split('_')[0])
        grad = get_layer_param_mean(gradients, layer_idx)
        color = grad_to_hex(grad, vmin, vmax)
        width = 5 if not math.isnan(grad) else 3
        line_type = '===' if not math.isnan(grad) else '-.-'
        u_clean = u.replace('.', '_')
        v_clean = v.replace('.', '_')
        connections_mermaid.append(f'    {u_clean} {line_type} {v_clean}\n')
        connection_format.append(f'    linkStyle {idx} stroke:{color},stroke-width:{width}px;\n')

    diagram += '\n    %% Connections\n'
    diagram += ''.join(connections_mermaid)
    diagram += '\n    %% Connection Styles based on gradient\n'
    diagram += ''.join(connection_format)
    diagram += '\n'

    diagram += f"    classDef blockStyle fill:{c_alpha_blue},stroke:{c_blue},stroke-width:2px\n"
    diagram += f"    classDef noBox fill:none,stroke:none;\n"
    return diagram

def generateBackpropagationGraph(graphs_dir, model_yaml):
    graphs_dir = Path(graphs_dir)
    yaml_files = sorted(graphs_dir.glob('graph_*.yaml'))

    for yaml_file in yaml_files:
        gradients = parseYaml(yaml_file)
        diagram = generateMermaidModel(gradients, model_yaml)

        file_path = graphs_dir.parent / yaml_file.name
        file_path = file_path.with_suffix('.mmd')

        with open(file_path, "w") as file:
            file.write(diagram)
        log(f"Generated Mermaid diagram: {file_path}")

    legend_path = graphs_dir.parent / 'colormap_legend.png'
    export_colormap_legend(cmap_name='turbo', vmin=0, vmax=1, filename=legend_path)

if __name__ == "__main__":
    from pathlib import Path

    base_dir = Path('/home/arvc/eeha/kaist-cvpr15/runs/detect/variance_llvip_Ch4_late_split_v1')
    latest = sorted([d for d in base_dir.iterdir() if d.is_dir()])[-1]
    graphs_dir = latest / 'gradients' / 'raw_data'
    model_yaml = '/home/arvc/eeha/yolo_test_utils/yolo_config/yoloCh4_late_split_v1.yaml' 
    generateBackpropagationGraph(graphs_dir, model_yaml)
