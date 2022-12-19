
% open_system('rlCartPoleSimscapeModel')
% open_system('rlCartPoleSimscapeModel_modified')

%%
% env = rlPredefinedEnv('CartPoleSimscapeModel-Continuous');
numObs = 6;
obsInfo = rlNumericSpec([numObs 1]);
obsInfo.Name = "observations";
numAct = 1;
actInfo = rlNumericSpec([numAct 1],"LowerLimit",-15,"UpperLimit", 15);
actInfo.Name = "torque";


env = rlSimulinkEnv('rlCartPoleSimscapeModel_modified_2','rlCartPoleSimscapeModel_modified_2/RL Agent', obsInfo, actInfo);

obsInfo = getObservationInfo(env);
numObservations = obsInfo.Dimension(1);
actInfo = getActionInfo(env);

%% Time parameters 
Ts = 0.02;
Tf = 25;

rng(0)

statePath = [
    featureInputLayer(numObservations,'Normalization','none','Name','observation')
    fullyConnectedLayer(128,'Name','CriticStateFC1')
    reluLayer('Name','CriticRelu1')
    fullyConnectedLayer(200,'Name','CriticStateFC2')];

actionPath = [
    featureInputLayer(1,'Normalization','none','Name','action')
    fullyConnectedLayer(200,'Name','CriticActionFC1','BiasLearnRateFactor',0)];

commonPath = [
    additionLayer(2,'Name','add')
    reluLayer('Name','CriticCommonRelu')
    fullyConnectedLayer(1,'Name','CriticOutput')];

criticNetwork = layerGraph(statePath);
criticNetwork = addLayers(criticNetwork,actionPath);
criticNetwork = addLayers(criticNetwork,commonPath);
    
criticNetwork = connectLayers(criticNetwork,'CriticStateFC2','add/in1');
criticNetwork = connectLayers(criticNetwork,'CriticActionFC1','add/in2');
criticNetwork = dlnetwork(criticNetwork);

% figure
% plot(layerGraph(criticNetwork))

criticOptions = rlOptimizerOptions('LearnRate',1e-03,'GradientThreshold',1);

critic = rlQValueFunction(criticNetwork,obsInfo,actInfo,...
    'ObservationInputNames','observation','ActionInputNames','action');
actorNetwork = [
    featureInputLayer(numObservations,'Normalization','none','Name','observation')
    fullyConnectedLayer(128,'Name','ActorFC1')
    reluLayer('Name','ActorRelu1')
    fullyConnectedLayer(200,'Name','ActorFC2')
    reluLayer('Name','ActorRelu2')
    fullyConnectedLayer(1,'Name','ActorFC3')
    tanhLayer('Name','ActorTanh1')
    scalingLayer('Name','ActorScaling','Scale',max(actInfo.UpperLimit))];
actorNetwork = dlnetwork(actorNetwork);

actorOptions = rlOptimizerOptions('LearnRate',5e-04,'GradientThreshold',1);

actor = rlContinuousDeterministicActor(actorNetwork,obsInfo,actInfo);

agentOptions = rlDDPGAgentOptions(...
    'SampleTime',Ts,...
    'ActorOptimizerOptions',actorOptions,...
    'CriticOptimizerOptions',criticOptions,...
    'ExperienceBufferLength',1e6,...
    'MiniBatchSize',128);
agentOptions.NoiseOptions.Variance = 0.4;
agentOptions.NoiseOptions.VarianceDecayRate = 1e-5;

agent = rlDDPGAgent(actor,critic,agentOptions);

maxepisodes = 2000;
maxsteps = ceil(Tf/Ts);
trainingOptions = rlTrainingOptions(...
    'MaxEpisodes',maxepisodes,...
    'MaxStepsPerEpisode',maxsteps,...
    'ScoreAveragingWindowLength',6,...
    'Verbose',false,...
    'Plots','training-progress',...
    'StopTrainingCriteria','AverageReward',...
    'StopTrainingValue',-400,...
    'SaveAgentCriteria','EpisodeReward',...
    'SaveAgentValue',-400);
doTraining = true;

if doTraining    
    % Train the agent.
    trainingStats = train(agent,env,trainingOptions);
else
    % Load the pretrained agent for the example.
    load('SimscapeCartPoleDDPG.mat','agent')
end

simOptions = rlSimulationOptions('MaxSteps',500);
experience = sim(env,agent,simOptions);