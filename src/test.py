import matplotlib.pyplot as plt
import pandas as pd
import torch

def main():
    test = torch.load("test_2023_KAK.pt")
    normal = torch.load("normal_2023_KAK.pt")
    print('test')
    print(test)
    print("\nnormal")
    print(normal)

if __name__ == "__main__":
    main()
    