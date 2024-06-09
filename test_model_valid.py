import torch

def check_pkl_validity(file_path):
    try:
        # Attempt to load the file
        data = torch.load(file_path)
        print("The file is valid.")
        return True
    except Exception as e:
        # Handle the exception if the file is not valid
        print(f"An error occurred: {e}")
        return False

# Example usage
file_path = 'resnet_torch_model.pkl'
is_valid = check_pkl_validity(file_path)