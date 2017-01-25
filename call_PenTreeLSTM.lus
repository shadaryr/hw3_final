require 'torch'
require 'nn'
require 'optim'
require 'eladtools'
require 'recurrent'
require './textDataProvider'
-------------------------------------------------------

-- ****Generating sentences!****
local best_model = "/home/shadaryr@st.technion.ac.il/hw3_elad/Results/WedJan2518:14:432017/Net_27.t7"
modelConfig = torch.load(best_model)
print('==>Loaded Net: ' .. best_model)
modelConfig.classifier:share(modelConfig.embedder, 'weight', 'gradWeight')
local trainingConfig = require './trainRecurrent'
local train = trainingConfig.train
local evaluate = trainingConfig.evaluate
local sample = trainingConfig.sample
local optimState = trainingConfig.optimState
local saveModel = trainingConfig.saveModel
for e = 1, 5 do
	print('\nSampled Text:\n' .. sample('Buy low, sell high is the', 5, true))
end
