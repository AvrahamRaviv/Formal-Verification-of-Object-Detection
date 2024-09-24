import os
import re
import sys
import torch
import shutil
import subprocess
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from networks.object_detection_classifier import NeuralNetwork_OL_v2_old
from generate_vnnlib import generate_threshold_property
from custom_from_MNIST import CustomMnistDataset_OL


def add_general(yaml_file, device="cpu", root_dir="./abcrown_dir", tab=" "*2):
    yaml_file.write("\n".join([
        "general:\n"
        f"{tab}device: {device}",
        f"{tab}loss_reduction_func: min",
        f"{tab}root_path: {os.path.abspath(root_dir)}",
        # f"{tab}root_path: ../vnncomp2023_benchmarks/benchmarks/traffic_signs_recognition",
        f"{tab}csv_name: instances.csv",
        f"{tab}graph_optimizer: Customized('custom_graph_optimizer', 'merge_sign')",
        f"{tab}sparse_interm: false",
        f"{tab}save_adv_example: true",
    ]) + "\n")

def add_model(yaml_file, shape="[1,1,90,90]", tab=" "*2):
    yaml_file.write("\n".join([
        "model:",
        f"{tab}input_shape: {shape}",
        f"{tab}onnx_loader: Customized('custom_model_loader', 'customized_my_loader')",
    ])  + "\n")


def add_attack(yaml_file, tab=" "*2):
    yaml_file.write("\n".join([
        "attack:",
        f"{tab}pgd_order: before",
        f"{tab}pgd_restarts: 50",
        f"{tab}pgd_batch_size: 50",
        f"{tab}cex_path: ./cex.txt",
        # f"{tab}attack_func: Customized('custom_attacker', 'use_LiRPANet')",
        # f"{tab}adv_saver: Customized('custom_adv_saver', 'customized_gtrsb_saver')",
        # f"{tab}early_stop_condition: Customized('custom_early_stop_condition', 'customized_gtrsb_condition')",
        # f"{tab}pgd_loss: Customized('custom_pgd_loss', 'customized_gtrsb_loss')",
        # f"{tab}adv_example_finalizer: Customized('custom_adv_example_finalizer', 'customized_gtrsb_adv_example_finalizer')",
    ]) + "\n")

def add_solver(yaml_file, tab=" "*2):
    yaml_file.write("\n".join([
        "solver:", 
        f"{tab}batch_size: 1", 
        f"{tab}min_batch_size_ratio: 1", 
        f"{tab}alpha-crown:", 
        f"{tab}{tab}disable_optimization: ['MaxPool']",
        f"{tab}beta-crown:",
        f"{tab}{tab}iteration: 20",
        f"{tab}{tab}lr_beta: 0.03",
        f"{tab}mip:",
        f"{tab}{tab}parallel_solvers: 8",
        f"{tab}{tab}solver_threads: 4",
        f"{tab}{tab}refine_neuron_time_percentage: 0.8",
        f"{tab}{tab}skip_unsafe: True",
    ]) + "\n")


def add_bab(yaml_file, tab=" "*2):
    yaml_file.write("\n".join([
        "bab:", 
        f"{tab}pruning_in_iteration: False", 
        f"{tab}sort_domain_interval: 1", 
        f"{tab}branching:", 
        f"{tab}{tab}method: nonlinear",
        f"{tab}{tab}candidates: 3",
        f"{tab}{tab}nonlinear_split:",
        f"{tab}{tab}{tab}num_branches: 2",
        f"{tab}{tab}{tab}method: shortcut",
        f"{tab}{tab}{tab}filter: true"
    ]) + "\n")


def generate_instances_file(root_dir, model_path, vnnlib_paths, timeout):
    instances_filename = f"{root_dir}/instances.csv"
    if os.path.exists(instances_filename):
        shutil.move(instances_filename, instances_filename.replace(".csv", "_old.csv"))
    with open(instances_filename, "w") as instances_fw:
        for vnnlib_path in vnnlib_paths:
            vnnlib_path = f"properties/{vnnlib_path}"
            # use only relative paths from root_dir in the instances.csv file
            model_relpath = model_path[len(root_dir)+1:]  # +1 removes leading '/'
            # vnnlib_relpath = vnnlib_path[len(root_dir)+1:]  # +1 removes leading '/'
            instances_fw.write(f"{model_relpath},{vnnlib_path},{timeout}\n")


def generate_abcrown_yaml_file(
    root_dir, yaml_path, sub_model_path, vnnlib_paths, timeout, shape
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generate_instances_file(root_dir, sub_model_path, vnnlib_paths, timeout)
    with open(yaml_path, "w") as yaml_fw:
        add_general(yaml_fw, device, root_dir)
        add_model(yaml_fw, shape)
        add_attack(yaml_fw)
        add_solver(yaml_fw)
        add_bab(yaml_fw)


def extract_res_ce(output):
    if output.split("\n")[-1].startswith("safe"):
        return "safe", None
    elif output.split("\n")[-1].startswith("unsafe"):
        with open("/home/yizhak/Research/Code/early_exit/cex.txt") as fr:
            cex = [float(line.split(" ")[-1].split(")")[0]) for line in fr if "X_" in line]
        return "unsafe", torch.Tensor(cex)
    out_lines = output.split("\n")
    results = [line.split(" ") for line in out_lines if line.startswith("Result:")]
    # results = [answer, runtime for each property]
    answers = [res[1] for res in results]
    times = [float(res[-2]) for res in results]
    return answers, times

def verify(root_dir, sub_model_path, vnnlib_paths, shape, timeout):
    """
    verify with alpha beta crown:
    - generate yaml file for 
    - run in subprocess
    - extract result and counterexample
    """
    yaml_path = f"{root_dir}/abcrown.yaml"
    generate_abcrown_yaml_file(
        root_dir, yaml_path, sub_model_path, vnnlib_paths, timeout, shape
    )
    verifier = "/home/yizhak/Research/Code/alpha-beta-CROWN/complete_verifier/abcrown.py"
    python = "/home/yizhak/virtual_envs/abcrown/bin/python"
    # output_path = "abcrown_output.txt"
    # command = f"{python} {verifier} --config {yaml_path} > {output_path}"
    command = f"{python} {verifier} --config {yaml_path}"
    result = subprocess.run(command, capture_output=True, text=True, shell=True)
    output = result.stdout.strip()
    # extract only number of sat/unsat/timeout (not times and counterexamples)
    pattern = r"total verified \(safe/unsat\): (\d+) , total falsified \(unsafe/sat\): (\d+) , timeout: (\d+)"
    match = re.search(pattern, output)
    return {
        "safe": match.group(1), 
        "unsafe": match.group(2), 
        "timeout": match.group(3)
    }, None # None represents that counterexample is missing
    # res, ce = extract_res_ce(output)
    return res, ce


def convert_and_save_model_to_onnx(model, path, input_size):
    """
    Convert a PyTorch model to ONNX format and save it to the specified path.

    Parameters:
    - model (torch.nn.Module): The PyTorch model to be converted.
    - path (str): The path where the ONNX model will be saved.
    - input_size (tuple): The size of the input tensor.
    """
    model.eval()  # Set the model to evaluation mode
    dummy_input = torch.randn(*input_size)  # Create a dummy input with the specified size
    torch.onnx.export(
        model, dummy_input, path, export_params=True, opset_version=10,
        do_constant_folding=True, input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )


def load_network(network_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralNetwork_OL_v2_old()
    sd = torch.jit.load(network_path)
    model.load_state_dict(sd.state_dict())
    model.eval()
    return model


def generate_base_property(
    input_sample, epsilon, prediction, winner, root_dir, property_file
):
    flatten_sample = input_sample.flatten()
    num_features = flatten_sample.shape[0]
    input_defs = ""
    input_constraints = ""
    output_defs = ""
    output_constraints = "(assert (or\n"

    # Generate input definitions and constraints
    for i in range(num_features):
        input_defs += f"(declare-const X_{i} Real)\n"
        input_constraints += f"(assert (<= X_{i} {flatten_sample[i] + epsilon}))\n"
        input_constraints += f"(assert (>= X_{i} {flatten_sample[i] - epsilon}))\n"

    # Generate output definition and constraints
    flatten_prediction = prediction.flatten()
    for i, class_score in enumerate(flatten_prediction):
        output_defs += f"(declare-const Y_{i} Real)\n"

    # Generate output constraints for the prediction class being the winner
    for i, class_score in enumerate(flatten_prediction):
        if i != winner:
            output_constraints += f"(and (>= Y_{i} Y_{winner}))\n"
    output_constraints += "))\n"

    # output_constraints += f"(assert (>= Y_{winner} {confidence_threshold}))\n"

    # Combine constraints into VNNLIB format
    vnnlib_property = f"; VNNLIB property for epsilon ball and winner\n\n"
    
    vnnlib_property += '; Definition of input variables\n'
    vnnlib_property += input_defs + "\n"
    
    vnnlib_property += '; Definition of output variables\n'
    vnnlib_property += output_defs + "\n"
    
    vnnlib_property += '; Definition of input constraints\n'
    vnnlib_property += input_constraints + "\n"
    
    vnnlib_property += '; Definition of output constraints\n'
    vnnlib_property += output_constraints + "\n"

    # Write property to vnnlib file
    with open(property_file, "w") as f:
        f.write(vnnlib_property)
    return property_file


def generate_vnnlib_files(network, test_data, root_dir, epsilons):
    preoperties_dir = f"{root_dir}/properties"
    os.makedirs(preoperties_dir, exist_ok=True)
    for sample_index, (input_sample, label) in enumerate(test_data):
        if sample_index == 1:
            break
        for epsilon in epsilons:
            property_name = f"input_sample_{sample_index}_epsilon_{epsilon}"
            property_file = f"{preoperties_dir}/{property_name}.vnnlib"
            prediction = network(input_sample.float())
            winner = prediction.argmax()
            with open("pred_vs_gt.csv", "a") as fw:
                fw.write(f"{winner},{label[0].item()}\n")
            source_vnnlib = generate_base_property(
                input_sample, epsilon, prediction, winner, root_dir, property_file
            )
    

if __name__ == "__main__":
    
    proj_path = "/home/yizhak/Research/Code/FVOD/"
    test_path = "d_loc_test_100.csv"
    net_path = f"{proj_path}/models/d_loc_weights_class_head_scripted.pt"
    root_dir = os.path.abspath("./abcrown_dir")
    shape = "[1,1,90,90]"
    timeout = 600.0  # seconds
    timeout = 60.0  # seconds
    epsilons = [
        0.1, 0.075, 0.05, 0.025, 0.01, 0.0075, 0.005, 0.0025, 0.001, 
        0.00075, 0.0005, 0.00025, 0.0001, 7.5e-05, 5e-05, 2.5e-05, 1e-05
    ]
    generate_vnnlib_files_flag = False

    # load network
    network = load_network(net_path)
    # read input_samples from d_loc_test_100.csv
    test_df = pd.read_csv(test_path)
    size = 90  # height & width of an image in the custom dataset
    testingData = CustomMnistDataset_OL(test_df, test=True, ns=size)
    test_dataloader = DataLoader(testingData, batch_size=1, shuffle=False)
    test_iterator = iter(test_dataloader)

    if generate_vnnlib_files_flag:
        generate_vnnlib_files(network, test_iterator, root_dir, epsilons)
        sys.exit(0)

    preoperties_dir = f"{root_dir}/properties"
    vnnlib_paths = [
        fname for fname in os.listdir(preoperties_dir) if fname.endswith(".vnnlib")
    ]#[100:200]
    onnx_name = os.path.basename(net_path).replace('.pt', '.onnx')
    onnx_path = f"{root_dir}/{onnx_name}"
    convert_and_save_model_to_onnx(network, onnx_path, (1, 1, size, size))
    res, ce = verify(root_dir, onnx_path, vnnlib_paths, shape, timeout)
    print(f"res: {res}, ce={ce}")
    # if res == "safe":
    #     return "safe", None
    # elif res == "unsafe":
    #     # check that ce is not spurious: runner > winner and confidence > threshold
    #     if network(ce)[1].argmax() != winner:
    #         return "unsafe", ce
