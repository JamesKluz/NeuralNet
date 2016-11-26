//Neural Net class
//Author: James Kluz

#ifndef NEURAL_NET_H_
#define NEURAL_NET_H_

#include <vector>
#include </usr/local/include/Eigen/Dense>
using namespace Eigen;

class NeuralNet{
 public:
  //constructor for a new net
  //s_coef is a coefficient in front of the independent variable x in the
  //sigmoid function. The larger x, the more closely the sigmoid represents 
  //a step function
  NeuralNet(int inputs, int hidden_nodes, int outputs, double s_coef = 1.0);

  //Constructor for loading a saved net
  NeuralNet(std::string loader);

  //Takes in a vector of input for the net
  //vector.size() must equal the declared number of inputs for the net
  //Returns the output vector 
  std::vector<double> QueryNet(std::vector<double> &query_input) const;

  //the two vector inputs are an input vector and the desired target output vector
  //The learn rate is the epsilon value for the descent algorithm that
  //makes rate corrections
  void TrainNet(std::vector<double> &train_input, std::vector<double> &target_ouput, double learn_rate);
  
  //saves the net to a .txt file
  void Save(std::string save_file) const;

  int GetNumInputs() const{
    return num_inputs_;
  }

  int GetNumHidden() const{
    return num_hidden_;
  }
  
  int GetNumOutputs() const{
    return num_outputs_;
  }

 private: 
  double Sigmoid(double x) const;

  //loads a previously saved net
  void Load(std::string load_file);

  int num_inputs_;
  int num_outputs_;
  int num_hidden_;
  double sigmoid_coefficient_;

  MatrixXd weights_input_hidden_;
  MatrixXd weights_hidden_output_;
};

#endif //NEURAL_NET_H_

