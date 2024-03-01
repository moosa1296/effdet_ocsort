
def read_and_sort_data(file_path):
    with open(file_path, 'r') as file:
        data = [list(map(float, line.split())) for line in file]

    # Sort the data by the last element in each list (frame number)
    sorted_data = sorted(data, key=lambda x: x[-1])

    return sorted_data

def write_data_to_file(sorted_data, output_file_path):
    with open(output_file_path, 'w') as file:
        for line in sorted_data:
            line_str = ' '.join(map(str, line))
            file.write(line_str + '\n')

# File paths
input_file_path = '/home/user-1/results/tracking_results.txt'
 # Replace with your input file path
output_file_path = '/home/user-1/results/tracking_results_2.txt'  # Replace with your desired output file path

# Read, sort, and write data
sorted_data = read_and_sort_data(input_file_path)
write_data_to_file(sorted_data, output_file_path)