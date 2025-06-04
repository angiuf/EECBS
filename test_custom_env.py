#!/usr/bin/env python3
"""
Test script for EECBS algorithm on warehouse environments.
Reads environment cases, creates temporary .scen files, runs the cpp EECBS code,
and computes the same metrics as other models.
"""

import os
import sys
import time
import json
import csv
import subprocess
import tempfile
import datetime
import numpy as np
from pathlib import Path
from tqdm import tqdm

def count_collisions(solution, obstacle_map):
    """Count agent-agent and obstacle collisions in the solution."""
    agent_agent_collisions = 0
    obstacle_collisions = 0
    num_agents = 0
    
    # Convert solution format to timestep-based format
    timestep_based_solution = []
    if len(solution) > 0:
        num_agents = len(solution)
        max_timesteps = max(len(agent_path) for agent_path in solution)
        
        for timestep in range(max_timesteps):
            positions_at_timestep = []
            for agent_idx in range(num_agents):
                if timestep < len(solution[agent_idx]):
                    positions_at_timestep.append(solution[agent_idx][timestep])
                else:
                    # Agent has reached goal, stays at final position
                    positions_at_timestep.append(solution[agent_idx][-1])
            timestep_based_solution.append(positions_at_timestep)
    
    # Now count collisions
    for timestep in range(len(timestep_based_solution)):
        positions = timestep_based_solution[timestep]
        
        # Check agent-agent collisions
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                if positions[i] == positions[j]:
                    agent_agent_collisions += 1
        
        # Check obstacle collisions
        for pos in positions:
            if len(pos) >= 2:  # Ensure position has at least x, y coordinates
                x, y = pos[0], pos[1]
                if (0 <= x < obstacle_map.shape[0] and 
                    0 <= y < obstacle_map.shape[1] and 
                    obstacle_map[x, y] == 1):  # Obstacle
                    obstacle_collisions += 1
    
    return agent_agent_collisions, obstacle_collisions


def get_csv_logger(model_dir, default_model_name):
    """Create CSV logger for results."""
    model_dir_path = Path(model_dir)
    csv_path = model_dir_path / f"log-{default_model_name}.csv"
    create_folders_if_necessary(csv_path)
    csv_file = open(csv_path, "a")
    return csv_file, csv.writer(csv_file)


def create_folders_if_necessary(path):
    """Create necessary folders for the given path."""
    path = Path(path)
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)


def convert_numpy_to_native(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_native(item) for item in obj]
    else:
        return obj


def load_map_from_file(map_file_path):
    """Load map from .map file and return as numpy array."""
    with open(map_file_path, 'r') as f:
        lines = f.readlines()
    
    # Parse header
    type_line = lines[0].strip()
    height = int(lines[1].split()[1])
    width = int(lines[2].split()[1])
    
    # Parse map
    map_data = []
    map_start_idx = 4  # Skip type, height, width, "map"
    for i in range(map_start_idx, map_start_idx + height):
        row = lines[i].strip()
        map_row = []
        for char in row:
            if char == '.' or char == ' ':
                map_row.append(0)  # Free space
            elif char == '@' or char == 'T':
                map_row.append(1)  # Obstacle
            else:
                map_row.append(0)  # Default to free space
        map_data.append(map_row)
    
    return np.array(map_data)


def create_map_file(obstacles, map_file_path):
    """Create a .map file from obstacle numpy array."""
    height, width = obstacles.shape
    
    with open(map_file_path, 'w') as f:
        f.write("type octile\n")
        f.write(f"height {height}\n")
        f.write(f"width {width}\n")
        f.write("map\n")
        
        for row in obstacles:
            line = ""
            for cell in row:
                if cell == 1:
                    line += "@"  # Obstacle
                else:
                    line += "."  # Free space
            f.write(line + "\n")


def create_scenario_file(start_positions, goal_positions, map_name, scenario_file_path):
    """Create a .scen file from start and goal positions."""
    num_agents = len(start_positions)
    
    with open(scenario_file_path, 'w') as f:
        f.write("version 1\n")
        
        for i in range(num_agents):
            start_x, start_y = start_positions[i]
            goal_x, goal_y = goal_positions[i]
            
            # Calculate optimal distance (Manhattan distance for grid)
            optimal_length = abs(goal_x - start_x) + abs(goal_y - start_y)
            
            # Format: bucket, map, map_width, map_height, start_x, start_y, goal_x, goal_y, optimal_length
            f.write(f"0\t{map_name}\t32\t32\t{start_y}\t{start_x}\t{goal_y}\t{goal_x}\t{optimal_length:.8f}\n")


def parse_paths_output(paths_file_path, num_agents):
    """Parse the paths output file from EECBS."""
    solution = [[] for _ in range(num_agents)]
    
    if not os.path.exists(paths_file_path):
        return solution
    
    try:
        with open(paths_file_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if line.startswith("Agent"):
                # Parse agent line: "Agent 0: (16,5)->(17,5)->(17,6)->..."
                parts = line.split(': ', 1)
                if len(parts) == 2:
                    agent_idx = int(parts[0].split()[1])
                    path_str = parts[1]
                    
                    if agent_idx < num_agents and "->" in path_str:
                        # Parse path: format like "(5,16)->(6,16)->(7,16)->"
                        positions = path_str.split("->")
                        for pos_str in positions:
                            pos_str = pos_str.strip("() ")
                            if "," in pos_str and pos_str:
                                try:
                                    x, y = map(int, pos_str.split(","))
                                    solution[agent_idx].append((x, y))
                                except ValueError:
                                    # Skip invalid position strings
                                    continue
    except Exception as e:
        print(f"Error parsing paths file: {e}")
    
    return solution


def run_eecbs_rtc(map_file, scenario_file, output_csv, paths_file, num_agents, time_limit=60):
    """Run the EECBS executable and return results."""
    cbs_executable = Path(__file__).parent / "cbs"
    
    cmd = [
        str(cbs_executable),
        "-m", str(map_file),
        "-a", str(scenario_file),
        "-o", str(output_csv),
        "--outputPaths", str(paths_file),
        "-k", str(num_agents),
        "-t", str(time_limit)
    ]
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=time_limit + 10)
        actual_time = time.time() - start_time
        
        # Check if the process was successful and if a solution was found
        process_success = result.returncode == 0
        solution_found = False
        
        if process_success:
            # Parse stdout to check if solution was found
            stdout_lines = result.stdout.strip().split('\n')
            for line in stdout_lines:
                # Look for the result line that contains "Optimal" or failure indicators
                if ':' in line and ('Optimal' in line or 'Suboptimal' in line):
                    solution_found = True
                    break
                elif 'No solution' in line or 'Failed' in line or 'Time limit' in line:
                    solution_found = False
                    break
            
            # Also check if output files exist and have content
            if solution_found:
                if not output_csv.exists() or not paths_file.exists():
                    solution_found = False
                else:
                    # Check if paths file has actual paths
                    try:
                        with open(paths_file, 'r') as f:
                            paths_content = f.read().strip()
                            if not paths_content or 'Agent' not in paths_content:
                                solution_found = False
                    except:
                        solution_found = False
        
        success = process_success and solution_found
        
        return success, actual_time, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, time_limit, "", "Timeout"
    except Exception as e:
        return False, time.time() - start_time, "", str(e)


def parse_output_csv(csv_file_path):
    """Parse the output CSV file from EECBS and extract only runtime."""
    if not os.path.exists(csv_file_path):
        return {}
    
    try:
        with open(csv_file_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Only extract runtime
                if 'runtime' in row and 'solution cost' in row:
                    try:
                        return {'runtime': float(row['runtime']), 'solution cost': float(row['solution cost'])}
                    except (ValueError, TypeError):
                        pass
                break  # Only read first row
    except Exception as e:
        print(f"Error parsing output CSV: {e}")
    
    return {}


def run_single_test(dataset_path, map_name, num_agents, test_id, time_limit=60):
    """Run a single test case."""
    dataset_path = Path(dataset_path)
    
    # Load map
    map_file_path = dataset_path / map_name / "input" / "map" / f"{map_name}.map"
    if not map_file_path.exists():
        print(f"Map file not found: {map_file_path}")
        return None, None
    
    # Load test case
    case_file_path = dataset_path / map_name / "input" / "start_and_goal" / f"{num_agents}_agents" / f"{map_name}_{num_agents}_agents_ID_{str(test_id).zfill(3)}.npy"
    if not case_file_path.exists():
        print(f"Test case file not found: {case_file_path}")
        return None, None
    
    # Load positions
    positions = np.load(case_file_path, allow_pickle=True)
    start_positions = positions[:, 0]
    goal_positions = positions[:, 1]
    
    # Load obstacle map for collision detection
    obstacle_map = load_map_from_file(map_file_path)
    
    # Create temporary files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        
        # Create scenario file
        scenario_file = temp_dir / f"test_{test_id}.scen"
        create_scenario_file(start_positions, goal_positions, f"{map_name}.map", scenario_file)
        
        # Output files
        output_csv = temp_dir / f"output_{test_id}.csv"
        paths_file = temp_dir / f"paths_{test_id}.txt"
        
        # Run EECBS
        success, actual_time, stdout, stderr = run_eecbs_rtc(
            map_file_path, scenario_file, output_csv, paths_file, num_agents, time_limit
        )
        
        # Parse results
        results = {
            'finished': False,
            'time': actual_time,
            'episode_length': 0,
            'total_steps': 0,
            'avg_steps': 0,
            'max_steps': 0,
            'min_steps': 0,
            'total_costs': 0,
            'avg_costs': 0,
            'max_costs': 0,
            'min_costs': 0,
            'agent_collisions': 0,
            'obstacle_collisions': 0,
            'crashed': False,
            'agent_coll_rate': 0,
            'obstacle_coll_rate': 0,
            'total_coll_rate': 0
        }
        
        solution = []
        
        if success:
            # Parse paths
            solution = parse_paths_output(paths_file, num_agents)
            
            # Parse CSV output for runtime
            csv_data = parse_output_csv(output_csv)
            if csv_data and 'runtime' in csv_data:
                results['time'] = csv_data['runtime']
            
            # Validate solution: check if all agents reach their goals
            solution_valid = True
            if solution and len(solution) == num_agents:
                for agent_idx in range(num_agents):
                    if len(solution[agent_idx]) == 0:
                        solution_valid = False
                        break
                    # Check if last position equals goal position
                    last_pos = solution[agent_idx][-1]
                    goal_pos = tuple(goal_positions[agent_idx])
                    if last_pos != goal_pos:
                        solution_valid = False
                        break
            else:
                solution_valid = False
            
            results['finished'] = solution_valid
            
            if solution_valid and solution:
                # Calculate metrics from parsed paths
                episode_length = max(len(path) for path in solution if len(path) > 0) - 1  # Subtract 1 for initial position
                results['episode_length'] = max(episode_length, 0)
                
                # Calculate steps and costs for each agent
                agent_steps = []
                agent_costs = []
                
                for agent_path in solution:
                    if len(agent_path) > 0:
                        # Count actual movements (steps), not wait actions
                        steps = 0
                        for i in range(1, len(agent_path)):
                            current_pos = agent_path[i]
                            previous_pos = agent_path[i-1]
                            # Only count as a step if the agent actually moved
                            if current_pos != previous_pos:
                                steps += 1

                        cost = len(agent_path) - 1  # In grid world, cost equals actual steps
                        agent_steps.append(steps)
                        agent_costs.append(cost)
                    else:
                        agent_steps.append(0)
                        agent_costs.append(0)
                
                if agent_steps:
                    results['total_steps'] = sum(agent_steps)
                    results['avg_steps'] = np.mean(agent_steps)
                    results['max_steps'] = max(agent_steps)
                    results['min_steps'] = min(agent_steps)
                    
                    results['total_costs'] = sum(agent_costs)
                    results['avg_costs'] = np.mean(agent_costs)
                    results['max_costs'] = max(agent_costs)
                    results['min_costs'] = min(agent_costs)
                
                # assert if the computed solution cost matches the CSV data
                assert results['total_costs'] == csv_data['solution cost'], "Mismatch in solution cost from CSV and computed cost."

                # Use solution cost from CSV if available
                if csv_data and 'solution cost' in csv_data:
                    results['total_costs'] = csv_data['solution cost']
                    results['avg_costs'] = csv_data['solution cost'] / num_agents if num_agents > 0 else 0
                
                # Count collisions
                agent_coll, obs_coll = count_collisions(solution, obstacle_map)
                results['agent_collisions'] = agent_coll
                results['obstacle_collisions'] = obs_coll
                results['crashed'] = (agent_coll + obs_coll) > 0
                
                if episode_length > 0 and num_agents > 0:
                    total_agent_timesteps = episode_length * num_agents
                    results['agent_coll_rate'] = agent_coll / total_agent_timesteps
                    results['obstacle_coll_rate'] = obs_coll / total_agent_timesteps
                    results['total_coll_rate'] = (agent_coll + obs_coll) / total_agent_timesteps
            
            elif csv_data:
                if 'runtime' in csv_data:
                    results['time'] = csv_data['runtime']
        
        return results, solution


def test_eecbs_rtc():
    """Main testing function for EECBS."""
    # Base paths
    base_dir = Path(__file__).parent
    project_dir = base_dir.parent.parent
    dataset_path = project_dir / 'baselines/Dataset'
    model_name = "EECBS"
    
    # Map configurations for testing
    map_configurations = [
        {
            "map_name": "15_15_simple_warehouse",
            "size": 15,
            "n_tests": 200,
            "list_num_agents": [4, 8, 12, 16, 20, 22]
        },
        {
            "map_name": "50_55_simple_warehouse",
            "size": 50,
            "n_tests": 20,
            "list_num_agents": [4, 8, 16, 32, 64, 128],
        },
        {
            "map_name": "50_55_long_shelves",
            "size": 50,
            "n_tests": 200,
            "list_num_agents": [4, 8, 16, 32, 64, 128]
        },
        {
            "map_name": "50_55_open_space_warehouse_bottom",
            "size": 50,
            "n_tests": 200,
            "list_num_agents": [4, 8, 16, 32, 64, 128]
        }
    ]
    
    # CSV header
    header = ["agents", "success_rate", 
              "time", "time_std", "time_min", "time_max",
              "episode_length", "episode_length_std", "episode_length_min", "episode_length_max",
              "total_steps", "total_steps_std", "total_steps_min", "total_steps_max",
              "avg_steps", "avg_steps_std", "avg_steps_min", "avg_steps_max",
              "max_steps", "max_steps_std", "max_steps_min", "max_steps_max",
              "min_steps", "min_steps_std", "min_steps_min", "min_steps_max",
              "total_costs", "total_costs_std", "total_costs_min", "total_costs_max",
              "avg_costs", "avg_costs_std", "avg_costs_min", "avg_costs_max",
              "max_costs", "max_costs_std", "max_costs_min", "max_costs_max",
              "min_costs", "min_costs_std", "min_costs_min", "min_costs_max",
              "agent_collision_rate", "agent_collision_rate_std", "agent_collision_rate_min", "agent_collision_rate_max",
              "obstacle_collision_rate", "obstacle_collision_rate_std", "obstacle_collision_rate_min", "obstacle_collision_rate_max",
              "total_collision_rate", "total_collision_rate_std", "total_collision_rate_min", "total_collision_rate_max"]

    # Process each map configuration
    for config in map_configurations:
        map_name = config["map_name"]
        size = config["size"]
        n_tests = config["n_tests"]
        list_num_agents = config["list_num_agents"]

        print(f"\nProcessing map: {map_name}")
        
        # Check if map exists
        map_path = dataset_path / map_name
        if not map_path.exists():
            print(f"WARNING: Map path does not exist: {map_path}")
            continue

        # Create output directory for results
        output_dir = map_path / "output" / model_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Setup CSV logger
        results_path = base_dir / "results"
        results_path.mkdir(parents=True, exist_ok=True)
        date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
        sanitized_map_name = map_name.replace("/", "_").replace("\\", "_")
        csv_filename_base = f'{model_name}_{sanitized_map_name}_{date}'
        csv_file, csv_logger = get_csv_logger(results_path, csv_filename_base)

        csv_logger.writerow(header)
        csv_file.flush()

        # Test each agent count
        for num_agents in list_num_agents:
            print(f"Starting tests for {num_agents} agents on map {map_name}")
            
            # Create output directory for this agent count
            output_agent_dir = output_dir / f"{num_agents}_agents"
            output_agent_dir.mkdir(parents=True, exist_ok=True)

            # Initialize result storage
            results = {
                'finished': [], 'time': [], 'episode_length': [],
                'total_steps': [], 'avg_steps': [], 'max_steps': [], 'min_steps': [],
                'total_costs': [], 'avg_costs': [], 'max_costs': [], 'min_costs': [],
                'crashed': [], 'agent_coll_rate': [], 'obstacle_coll_rate': [], 'total_coll_rate': []
            }

            # Run tests
            for test_id in tqdm(range(n_tests), desc=f"Map: {map_name}, Agents: {num_agents}"):
                res, solution = run_single_test(dataset_path, map_name, num_agents, test_id)
                
                if res is not None:
                    # Save results
                    results['finished'].append(res['finished'])
                    if res['finished']:
                        results['time'].append(res['time'])
                        results['episode_length'].append(res['episode_length'])
                        results['total_steps'].append(res['total_steps'])
                        results['avg_steps'].append(res['avg_steps'])
                        results['max_steps'].append(res['max_steps'])
                        results['min_steps'].append(res['min_steps'])
                        results['total_costs'].append(res['total_costs'])
                        results['avg_costs'].append(res['avg_costs'])
                        results['max_costs'].append(res['max_costs'])
                        results['min_costs'].append(res['min_costs'])
                        results['agent_coll_rate'].append(res['agent_coll_rate'])
                        results['obstacle_coll_rate'].append(res['obstacle_coll_rate'])
                        results['total_coll_rate'].append(res['total_coll_rate'])
                        results['crashed'].append(res['crashed'])

                    # Save solution to file
                    solution_filepath = output_agent_dir / f"solution_{model_name}_{map_name}_{num_agents}_agents_ID_{str(test_id).zfill(3)}.txt"
                    with open(solution_filepath, 'w') as f:
                        f.write("Metrics:\n")
                        serializable_res = convert_numpy_to_native(res)
                        json.dump(serializable_res, f, indent=4)
                        f.write("\n\nSolution:\n")
                        if solution:
                            for agent_idx, agent_path in enumerate(solution):
                                f.write(f"Agent {agent_idx}: {agent_path}\n")
                        else:
                            f.write("No solution found.\n")

            # Calculate aggregated metrics
            final_results = {}
            final_results['finished'] = sum(results['finished']) / len(results['finished']) if len(results['finished']) > 0 else 0

            # Calculate statistics for metrics when available
            metric_keys = ['time', 'episode_length', 'total_steps', 'avg_steps', 'max_steps', 'min_steps', 
                         'total_costs', 'avg_costs', 'max_costs', 'min_costs', 
                         'agent_coll_rate', 'obstacle_coll_rate', 'total_coll_rate']
            
            for key in metric_keys:
                if results[key]:
                    final_results[key] = np.mean(results[key])
                else:
                    final_results[key] = 0

            final_results['crashed'] = sum(results['crashed']) / len(results['crashed']) if len(results['crashed']) > 0 else 0

            print(f'SR: {final_results["finished"] * 100:.2f}%, Time: {final_results["time"]:.2f}s, '
                  f'Episode Length: {final_results["episode_length"]:.2f}, Total Collision Rate: {final_results["total_coll_rate"] * 100:.2f}%')

            # Write results to CSV
            data = [num_agents,
                    final_results['finished'] * 100,  # convert to percentage
                    final_results['time'],
                    np.std(results['time']) if results['time'] else 0,
                    np.min(results['time']) if results['time'] else 0,
                    np.max(results['time']) if results['time'] else 0,
                    final_results['episode_length'],
                    np.std(results['episode_length']) if results['episode_length'] else 0,
                    np.min(results['episode_length']) if results['episode_length'] else 0,
                    np.max(results['episode_length']) if results['episode_length'] else 0,
                    final_results['total_steps'],
                    np.std(results['total_steps']) if results['total_steps'] else 0,
                    np.min(results['total_steps']) if results['total_steps'] else 0,
                    np.max(results['total_steps']) if results['total_steps'] else 0,
                    final_results['avg_steps'],
                    np.std(results['avg_steps']) if results['avg_steps'] else 0,
                    np.min(results['avg_steps']) if results['avg_steps'] else 0,
                    np.max(results['avg_steps']) if results['avg_steps'] else 0,
                    final_results['max_steps'],
                    np.std(results['max_steps']) if results['max_steps'] else 0,
                    np.min(results['max_steps']) if results['max_steps'] else 0,
                    np.max(results['max_steps']) if results['max_steps'] else 0,
                    final_results['min_steps'],
                    np.std(results['min_steps']) if results['min_steps'] else 0,
                    np.min(results['min_steps']) if results['min_steps'] else 0,
                    np.max(results['min_steps']) if results['min_steps'] else 0,
                    final_results['total_costs'],
                    np.std(results['total_costs']) if results['total_costs'] else 0,
                    np.min(results['total_costs']) if results['total_costs'] else 0,
                    np.max(results['total_costs']) if results['total_costs'] else 0,
                    final_results['avg_costs'],
                    np.std(results['avg_costs']) if results['avg_costs'] else 0,
                    np.min(results['avg_costs']) if results['avg_costs'] else 0,
                    np.max(results['avg_costs']) if results['avg_costs'] else 0,
                    final_results['max_costs'],
                    np.std(results['max_costs']) if results['max_costs'] else 0,
                    np.min(results['max_costs']) if results['max_costs'] else 0,
                    np.max(results['max_costs']) if results['max_costs'] else 0,
                    final_results['min_costs'],
                    np.std(results['min_costs']) if results['min_costs'] else 0,
                    np.min(results['min_costs']) if results['min_costs'] else 0,
                    np.max(results['min_costs']) if results['min_costs'] else 0,
                    final_results['agent_coll_rate'] * 100,  # convert to percentage
                    np.std(results['agent_coll_rate']) * 100 if results['agent_coll_rate'] else 0,
                    np.min(results['agent_coll_rate']) * 100 if results['agent_coll_rate'] else 0,
                    np.max(results['agent_coll_rate']) * 100 if results['agent_coll_rate'] else 0,
                    final_results['obstacle_coll_rate'] * 100,  # convert to percentage
                    np.std(results['obstacle_coll_rate']) * 100 if results['obstacle_coll_rate'] else 0,
                    np.min(results['obstacle_coll_rate']) * 100 if results['obstacle_coll_rate'] else 0,
                    np.max(results['obstacle_coll_rate']) * 100 if results['obstacle_coll_rate'] else 0,
                    final_results['total_coll_rate'] * 100,  # convert to percentage
                    np.std(results['total_coll_rate']) * 100 if results['total_coll_rate'] else 0,
                    np.min(results['total_coll_rate']) * 100 if results['total_coll_rate'] else 0,
                    np.max(results['total_coll_rate']) * 100 if results['total_coll_rate'] else 0]
            
            csv_logger.writerow(data)
            csv_file.flush()

        csv_file.close()
        print(f"Completed testing for map: {map_name}")

    print("Finished all tests!")


if __name__ == '__main__':
    test_eecbs_rtc()