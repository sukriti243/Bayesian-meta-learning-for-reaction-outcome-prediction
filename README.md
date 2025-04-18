# Bayesian Meta-Learning for Few-Shot Reaction Outcome Prediction of Asymmetric Hydrogenation of Olefins

## Data
The dataset for asymmetric hydrogenation of olefins is available through http://asymcatml.net. The access to the database is provided by registration on the website followed by signing a user license agreement. The dataset used in time-based test splits is provided in the “Data” folder.

## Model training and evaluation for meta-learning methods
Four different meta-learning models are considered: Prototypical networks, DKT, ADKF, and ADKF-prior. The code for implementing these meta-learning methods is provided in their respective folders. Each folder contains separate files for obtaining tasks for meta-learning, model architecture, meta-training, and meta-testing. We describe these files for prototypical networks, but it remains the same for other meta-learning methods as well.
Prototypical network: this folder contains five files. The ‘get_tasks.py’ takes the training and test data to return training and test tasks respectively for meta-learning. The ‘proto_meta.py’ file contains the model architecture details to implement prototypical network. The ‘proto_meta_utils.py’ file has class ‘PrototypicalNetworkTrainer’ for meta-training and the function ‘evaluate_protonet_model’ for model evaluation. The function ‘test_protonet_model’ is used to obtain meta-model performance on the test set. 
The ‘proto_train_meta.py’ file can be run as python proto_train_meta.py in the command line to train the meta-model. There are several hyperparameters that can be tuned on the validation tasks such as number of examples in the support set (num_support), epochs (num_tain_steps), learning rate, distance metric, and so on. The results on the test tasks can be obtained by running proto_test_meta.py using the saved checkpoint file of the meta-trained model.

## Model training and evaluation for single-task methods
The code for the implementation of single-task methods is provided in the folder ‘Single task methods’. The tasks for meta-training and meta-testing can be obtained from get_tasks.py file. The single-task methods: random forest, support vector machines, gradient boosting, extra trees, decision tree, and adaptive boosting (AdaBoost) are implemented using scikit-learn. These can be run using, for example, ‘python benchmark_RF.py’ in the command line. The other single-task method is deep kernel learning (DKL). The Gaussian process model used in training DKL can be found in model.py file. The DKL model is implemented using PyTorch and GPyTorch. Some important hyperparameters are number of epochs, learning rate, etc. The next single-task method used as baseline is graph neural networks (GNN). The training and test tasks are constructed using the get_tasks_graph.py file. The architecture details of the message passing neural network is provided in model_graph.py file.   

## References
1. https://github.com/Wenlin-Chen/ADKF-IFT
2. https://github.com/seokhokang/reaction_yield_nn
3. https://onlinelibrary.wiley.com/doi/abs/10.1002/anie.202106880

