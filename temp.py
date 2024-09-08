import argparse
import subprocess
import json

import re

def get_gpu_vram(machine_name):
    # Example command that needs to be run on the remote machine
    try:
        result = subprocess.run(
            ["ssh", machine_name, "nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE, text=True
        )
        # Parse the first GPU's free memory (assuming single GPU per node or summing/averaging if multiple)
        free_memory = int(result.stdout.strip().split('\n')[0])
        return free_memory
    except:
        return -1

# Define LABCOMPS with the correct pattern for machine names
LABCOMPS = "^(ash|beech|cedar|curve|gpu|maple|oak|pixel|ray|texel|vertex|willow)[0-9]{2}\.doc"


# Arguments
parser = argparse.ArgumentParser(description="Tool to find a free lab machine.")
parser.add_argument("-l", "--list", action='store_true', help="List all matching machines instead of selecting one.")
parser.add_argument("-c", "--constraint", default='regexp( \"' + LABCOMPS + '\", Machine)', help="Constraint on which machines to select.")
parser.add_argument("--gpu", action='store_true', help="Filter to show only GPU machines and sort by available VRAM.")
parser.add_argument("--name", action='store_true', help="Filter to show only GPU machines and sort by available VRAM.")
parser.add_argument('--load', type=float, default=None, help='Optional load average threshold for filtering machines')
   
  
ARGS = parser.parse_args()

# Get condor machine status
condor_status = subprocess.run(
    [
        "/usr/local/condor/release/bin/condor_status",
        "-long",
        "-attributes",
        "Machine,LoadAvg,State",
        "-json",
        "-constraint",
        ARGS.constraint,
    ],
    stdout=subprocess.PIPE,
    check=True,  # Raises exception if subprocess fails
)

# Priority for specific machine names
def machine_priority(machine):
    name = machine["Machine"]
    if "gpu" or "ray" in name:
        return 0  # Highest priority for GPU machines
    elif "texel" in name:
        return 1  # Second highest priority for Texel machines
    return 2  # Lowest priority for all other machines


# Collect machine information
machines = json.loads(condor_status.stdout)

# Apply filters based on the machine state and load average
# filtered_machines = [m for m in machines if m["State"] == "Unclaimed" and m["LoadAvg"] < 0.1]
filtered_machines = [m for m in machines if m["LoadAvg"] < 0.65]

# Sort machines with a priority, favoring 'gpu' and 'texel'
if ARGS.gpu:
    # Filter only GPU machines
    gpu_machines = [m for m in machines if "gpu" in m["Machine"]] #  in (, "ray")]  #  and m["State"] == "Unclaimed" and m["LoadAvg"] < 0.1]
    ray_machines = [m for m in machines if "ray" in m["Machine"]] # 8gb vRAM
    beech_machines = [m for m in machines if "beech" in m["Machine"]] # 4gb vRAM
    willow_machines = [m for m in machines if "willow" in m["Machine"]] # 4gb vRAM
    gpu_machines.extend(ray_machines)
    gpu_machines.extend(beech_machines)
    if ARGS.load is not None:
        gpu_machines = [m for m in gpu_machines if m["LoadAvg"] < ARGS.load]

    # print(machines[0].keys())
    # exit()
    # print(f"gpu machines wiht less than 0.4 load: {gpu_machines}")
    # Retrieve VRAM usage for each GPU machine
    for machine in gpu_machines:
        machine['FreeVRAM'] = get_gpu_vram(machine["Machine"])

    gpu_machines = [m for m in gpu_machines if m["FreeVRAM"] > 2000]
    # Sort by available VRAM, descending
    gpu_machines.sort(key=lambda x: x['FreeVRAM'], reverse=True)

    
    # print(f"gpu machines sorted by free vram: {gpu_machines}")

    filtered_machines = gpu_machines
else:
    # Existing logic to sort based on machine priority
    filtered_machines.sort(key=machine_priority)


# Check for listing flag
if ARGS.list:
    # List all filtered machines
    for machine in filtered_machines:
        if ARGS.gpu:
            if ARGS.name:
                print(machine["Machine"])
            else:
                print(f'{machine["Machine"]} - State: {machine["State"]} - LoadAvg: {machine["LoadAvg"]} - FreeVRAM: {machine.get("FreeVRAM", "N/A")}')
        else:
            print(machine["Machine"])
else:
    # Existing logic to select one machine
    if filtered_machines:
        labm = filtered_machines[0]["Machine"]  # Just take the first instead of random choice
        print(labm)