import pycuda.driver as cuda
import pycuda.autoinit  # Automatically initializes CUDA

"""
This script checks Peer-to-Peer (P2P) access between GPUs using the CUDA API.
Functions:
check_p2p_access(): Checks and prints whether each pair of GPUs can access each other.
The function performs the following steps:
1. Retrieves the total number of GPUs available on the system.
2. Iterates through each pair of GPUs.
3. Checks if one GPU can access the memory of the other GPU using P2P access.
4. Prints the result for each pair of GPUs.
Note: This script requires the `pycuda` library to interact with the CUDA API.
"""

def check_p2p_access():

    # Get the number of GPUs
    n_gpus = cuda.Device.count()

    print(f"Total GPUs: {n_gpus}")
    # Check P2P access between each pair of GPUs
    for i in range(n_gpus):
        for j in range(n_gpus):
            if i != j:
                dev_i = cuda.Device(i)
                dev_j = cuda.Device(j)
                can_access = dev_i.can_access_peer(dev_j)
                print(f"GPU {i} can access GPU {j}: {'Yes' if can_access else 'No'}")

if __name__ == "__main__":
    check_p2p_access()
