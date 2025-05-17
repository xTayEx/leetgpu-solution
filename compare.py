import math
import torch
import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("-m", type=int, required=True)
parser.add_argument("-n", type=int, required=True)
parser.add_argument("-k", type=int, required=True)
parser.add_argument("--new", action="store_true")

args = parser.parse_args()

M = args.m
N = args.n
K = args.k

if args.new:
    A = torch.randn((M, N)).cuda()
    B = torch.randn((N, K)).cuda()
    with open("/tmp/A.txt", "w+") as f:
        for a_ele in A.flatten().tolist():
            f.write(f"{a_ele} ")

    with open("/tmp/B.txt", "w+") as f:
        for b_ele in B.flatten().tolist():
            f.write(f"{b_ele} ")
else:
    with open("/tmp/A.txt", "r") as f:
        A = torch.tensor([float(x) for x in f.read().split()]).reshape(M, N).cuda()
    with open("/tmp/B.txt", "r") as f:
        B = torch.tensor([float(x) for x in f.read().split()]).reshape(N, K).cuda()

try:
    cuda_matmul_reuslt = subprocess.check_output(["build/matmul", "--m", "32", "--n", "32", "--k", "32"]).split()
except subprocess.CalledProcessError as e:
    print("Error running the external program:", e)
    exit(1)

matmul_result = torch.matmul(A, B).flatten().tolist()
for cuda_ele, torch_ele in zip(cuda_matmul_reuslt, matmul_result):
    if not math.isclose(float(cuda_ele), float(torch_ele), rel_tol=1e-3):
        print(float(cuda_ele), float(torch_ele))

with open("/tmp/C.txt", "w+") as f:
    for c_ele in matmul_result:
        f.write(f"{round(c_ele, 6)} ")
