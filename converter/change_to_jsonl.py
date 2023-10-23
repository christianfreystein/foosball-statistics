import json

def convert_json_to_jsonl(input_json_path, output_jsonl_path):
    # Load the data from the JSON file
    with open(input_json_path, 'r') as json_file:
        data = json.load(json_file)

    # Write the data to a JSONL file
    with open(output_jsonl_path, 'w') as jsonl_file:
        for item in data:
            json.dump(item, jsonl_file)
            jsonl_file.write('\n')

# Specify the input and output file paths
input_json_path = r"C:\Users\chris\foosball-dataset\foosball_coco\test\metadata.json"
output_jsonl_path = r"C:\Users\chris\foosball-dataset\foosball_coco\test\metadata.jsonl"
input_json_path1 = r"C:\Users\chris\foosball-dataset\foosball_coco\valid\metadata.json"
output_jsonl_path1 = r"C:\Users\chris\foosball-dataset\foosball_coco\valid\metadata.jsonl"
input_json_path2 = r"C:\Users\chris\foosball-dataset\foosball_coco\train\metadata.json"
output_jsonl_path2 = r"C:\Users\chris\foosball-dataset\foosball_coco\train\metadata.jsonl"

# Call the conversion function
convert_json_to_jsonl(input_json_path, output_jsonl_path)

convert_json_to_jsonl(input_json_path1, output_jsonl_path1)

convert_json_to_jsonl(input_json_path2, output_jsonl_path2)