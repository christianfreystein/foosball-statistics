import json


def read_json(file_path):
    """Read JSON data from a file."""
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def format_time(seconds):
    """Format seconds into 'hours:minutes:seconds' without decimal places."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)  # Use int to remove decimal places
    return f"{hours:02}:{minutes:02}:{secs:02}"


def write_txt(data, file_path):
    """Write the status, time, and accumulated time to a text file."""
    with open(file_path, 'w') as file:
        for status, time_seconds, accumulated_time in data:
            # Format time and accumulated time
            formatted_time = f"{time_seconds:.2f}"  # Time in seconds with two decimal places
            formatted_accumulated_time = format_time(accumulated_time)  # Accumulated time in 'hours:minutes:seconds'
            # Write status, time, and accumulated time to the file
            file.write(f'{status}, {formatted_time}, {formatted_accumulated_time}\n')


def process_tracking_data(data, fps=60):
    """Process the tracking data to get time spent in each status and accumulated time."""
    if not data or not isinstance(data, list):
        raise ValueError("The data should be a list of dictionaries.")

    status_times = []
    frame_count = 0
    accumulated_time = 0.0
    current_status = data[0].get('ball_status', 'Unknown')  # Start with the first status

    for frame in data:
        if isinstance(frame, dict):  # Ensure each frame is a dictionary
            new_status = frame.get('ball_status', 'Unknown')
            if new_status != current_status:
                # Calculate time for the previous status
                time_seconds = frame_count / fps
                accumulated_time += time_seconds
                # Record the time and accumulated time for the previous status
                status_times.append((current_status, time_seconds, accumulated_time))
                # Update to the new status and reset frame count
                current_status = new_status
                frame_count = 1
            else:
                frame_count += 1
        else:
            raise ValueError("Each frame should be a dictionary.")

    # Append the last status
    time_seconds = frame_count / fps
    accumulated_time += time_seconds
    status_times.append((current_status, time_seconds, accumulated_time))

    return status_times


# Define file paths
input_file_path = r"C:\Users\chris\foosball-statistics\Spredeman_Hoffmann_Romero_Gabriel_with_ball_status.json"
output_file_path = r"C:\Users\chris\Foosball Detector\ball_status_times.txt"

# Load the JSON data from the file
data = read_json(input_file_path)

# Process the tracking data to get status times and accumulated times
status_times = process_tracking_data(data, fps=60)

# Write the results to a text file
write_txt(status_times, output_file_path)

print(f"Time spent in each possession and accumulated time have been saved to {output_file_path}.")
