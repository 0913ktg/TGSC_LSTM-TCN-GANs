import argparse
import torch
from utils import DatasetGenerator
from make_dataset import MakeDataset
from matplotlib import pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', required=True, help='Path of the generator checkpoint')
parser.add_argument('--output_path', required=True, help='Path of the output .npy file')
parser.add_argument('--delta_path', default='', help='Path of the file containing the list of deltas for conditional generation')
parser.add_argument('--dataset_path', required=True, help="Path of the dataset for normalization")
parser.add_argument('--nUser', default=1, help='Number of the users')
parser.add_argument('--nTime', default=1, help='Length of the times (hour)')
parser.add_argument('--nSim', default=1, help='Number of the simulations')
parser.add_argument('--mType', default='lstm', help='Generation model type')

opt = parser.parse_args()

#If an unknown option is provided for the dataset, then don't use any normalization
dataset = MakeDataset(opt.dataset_path) 
model = torch.load(opt.checkpoint_path)
generator = DatasetGenerator(generator=model, dataset=dataset, mType = opt.mType) #Using default params

if opt.delta_path != '':
    delta_list = [float(line) for line in open(opt.delta_path)]
else:
    delta_list = None

user_size = int(int(opt.nUser) * int(opt.nTime) * int(opt.nSim) * 180 )

# generating data
result = generator.generate_dataset(outfile=opt.output_path, delta_list=delta_list, size=user_size)
result = torch.flatten(result)

# saving numpy array
result = result.detach().numpy() 

# saving plt plot
plt.plot(result)
plt.savefig('result.png')