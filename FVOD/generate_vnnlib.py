def generate_threshold_property(source_vnnlib, target_vnnlib, threshold):
    with open(source_vnnlib, 'r') as file:
        lines = file.readlines()
    
    input_constraints = []
    output_constraints = []
    in_output_constraints = False
    
    for line in lines:
        stripped_line = line.strip()
        if stripped_line.startswith('; Definition of output constraints'):
            in_output_constraints = True
        if in_output_constraints:
            if stripped_line.startswith('(assert (or') or stripped_line.startswith('(and (<='):
                output_constraints.append(stripped_line)
        else:
            input_constraints.append(stripped_line)
    
    comparison_counts = {}
    
    for constraint in output_constraints:
        if constraint.startswith('(assert (or'):
            continue
        parts = constraint.split()
        greater_class = parts[2]
        lesser_class = parts[3].rstrip('))')
        
        # we encode adversarial property, hence the lesser class is the winner
        # if one of the classes is bigger than the winner, we get SAT
        if greater_class not in comparison_counts:
            comparison_counts[greater_class] = 0
        if lesser_class not in comparison_counts:
            comparison_counts[lesser_class] = 0
        
        comparison_counts[lesser_class] += 1
    
    winner_class = max(comparison_counts, key=comparison_counts.get)
    
    print(f"The winner class is: {winner_class}")
    
    with open(target_vnnlib, 'w') as file:
        file.write("; Definition of input constraints\n")
        for ic in input_constraints:
            file.write(ic + "\n")
        file.write("\n")
        file.write("; Definition of output constraints\n")
        file.write(f"(assert (<= {winner_class} {threshold}))")
        file.write("\n")
    
    print(f"New VNNLIB property generated in '{target_vnnlib}' with confidence threshold {threshold} for class {winner_class}")


if __name__ == "__main__":
    vnncomp23_benchmarks = "/home/yizhak/Research/Code/vnncomp2023_benchmarks/benchmarks/"
    gtsrb = f"{vnncomp23_benchmarks}/traffic_signs_recognition/vnnlib/"
    source_vnnlib = f"{gtsrb}/model_30_idx_7040_eps_1.00000.vnnlib"
    threshold = 0.9
    target_vnnlib = source_vnnlib.split('.vnnlib')[0] + f'_threshold_{threshold}.vnnlib'
    generate_threshold_property(source_vnnlib, target_vnnlib, threshold)
