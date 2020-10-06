import os
import argparse

parser = argparse.ArgumentParser(description="Extract results to latex writings")

parser.add_argument('--results', type=str, help='results file')
parser.add_argument('--output', type=str, help='output results file')

args = parser.parse_args()

p_results = args.results
p_output  = args.output

output_values = []

with open(p_results, 'r') as f:

    lines = f.readlines()

    header = lines[0].replace(';\n', '').split(';')
    del lines[0]

    data_lines = [ l.replace(';\n', '').split(';') for l in lines ]

    # extract params here
    for line in data_lines:

        output_line = []
        model_name = line[0]

        sequence_size = int(model_name.split('seq')[1].split('_')[0])
        w_size, h_size, sv_begin, sv_end = list(map(int, model_name.split('seq')[0].split('_')[-5:-1]))

        bnorm = False
        if '_norm_' in model_name:
            bnorm = True

        snorm = False
        if '_seqnorm_' in model_name:
            snorm = True


        output_line.append(sequence_size)
        output_line.append((200 / w_size) * (200 / h_size))
        output_line.append((sv_begin, sv_end))
        output_line.append(model_name.split('bsize')[-1])
        output_line.append(bnorm)
        output_line.append(snorm)
        # 1, 3, 4, 6
        output_line.append(float(line[1]))
        output_line.append(float(line[3]))
        output_line.append(float(line[4]))
        output_line.append(float(line[6]))

        output_values.append(output_line)


output_f = open(p_output, 'w')

output_f.write('\\begin{tabular}{|r|r|c|c|r|r|r|r|r|r|} \n')
output_f.write('\t\hline \n')
output_f.write('\t\\textbf{$k$} & \\textbf{$m$} & \\textbf{$m_i$ SV size} & \\textbf{batch} & \\textbf{$bnorm$} & \\textbf{$snorm$} & \\textbf{Accuracy Train} & \\textbf{Accuracy Test} & \\textbf{AUC Train} & \\textbf{AUC Test} \\\\ \n')
output_f.write('\t\hline \n')

output_values = sorted(output_values, key=lambda l:l[-1], reverse=True)

# for each data
for v in output_values[:20]:
    output_f.write('\t{0} & {1} & {2} & {3} & {4} & {5} & {6:.3f} & {7:.3f} & {8:.3f} & {9:.3f} \\\\ \n' \
        .format(v[0], int(v[1]), "[{0}, {1}]".format(int(v[2][0]), int(v[2][1])), int(v[3]), int(v[4]), int(v[5]), v[6], v[7], v[8],  v[9]))

output_f.write('\t\hline \n')
output_f.write('\end{tabular} \n')
