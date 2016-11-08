//A driver for testing the NueralNet class on the MNIST data
//Usage:
//To Test:
//mnist_net test test_data.csv saved_net.txt
//To Train:
//Option: 1 (for a static learning rate)
//mnist_net train train_data.csv file_to_save.txt learning_rate #epochs sigmoid_coefficient
//Option: 2 (for a static learning rate)
//mnist_net train train_data.csv file_to_save.txt learning_rate #epochs sigmoid_coefficient #of_steps_till_epsilon_decrease
//note: at the nth epsilon decrease epsilon = epsilon/n

#include <iostream>
#include "neural_net.h"
#include <fstream>

double TestNet(NeuralNet &net, const std::string &data_file);
void TrainNet(NeuralNet &net, const std::string &data_file, double epsilon, int epochs, int num_steps_change = -1);


int main(int argc, char **argv){  
  if (argc!=4 && argc!=7 && argc!=8) {
    std::cout << "Incorrect usage, correct usage:\n";
    std::cout << "mnist_net {\"test\"} {input file} {load file}\n";
    std::cout << "-OR-\nmnist_net {\"train\"} {input file} {save file} {learning rate} {epochs} {Sigmoid Coefficient}\n";
    std::cout << "-OR-\nmnist_net {\"train\"} {input file} {save file} {Initial learn rate} {epochs} {Sigmoid Coefficient} {Steps until learning rate change}\n";
    return 0;
  } 
  const std::string test_or_train(argv[1]); 
  if(test_or_train == "test"){
    const std::string loader(argv[3]);
    NeuralNet mnist_net(loader);
    const std::string data_file(argv[2]); 
    double result = TestNet(mnist_net, data_file);
    std::cout << "Net Accuracy: " << result << std::endl;

  } else if (test_or_train == "train") {
    const std::string data_file(argv[2]);
    const std::string save_file(argv[3]);
    double epsilon = std::stod(argv[4]);
    int epochs = std::stoi(argv[5]);
    double sig_coef = std::stod(argv[6]);
    NeuralNet mnist_net{784, 397, 10, sig_coef, false};
    if(argc == 7){
      TrainNet(mnist_net, data_file, epsilon, epochs);
      mnist_net.Save(save_file);
    }
    else if (argc == 8){
      int num_steps_change = std::stoi(argv[7]);
      TrainNet(mnist_net, data_file, epsilon, epochs, num_steps_change);
      mnist_net.Save(save_file);
    }
  }
  return 0;
}

void TrainNet(NeuralNet &net, const std::string &data_file, double epsilon, int epochs, int num_steps_change){
  bool decreasing_learn = true;
  if(num_steps_change < 1)
    decreasing_learn = false;
  double learn_rate = epsilon;
  int counter = 0;
  int epsilon_divisor = 1;
  for(int x = 0; x < epochs; ++x){
    std::ifstream in_stream(data_file);
    std::cout << "******************** Epoch_" << x+1 << " **********************\n";
    if(!in_stream.is_open()){
      std::cout << "Failed to open training file, exiting\n";
      exit(1);
    }
    std::string token;
    while(std::getline(in_stream, token)){
      if(counter % 1000 == 0){
        std::cout << counter << " training instances\n";
      }
      ++counter;
      std::stringstream ss(token);
      std::vector<double> trainer;
      double k;
      int target;
      ss >> target;
      ss.ignore();
      //get and input, put in a vector
      while (ss >> k){
        trainer.push_back(k);
        //skip the comma and space
        ss.ignore();
      }
      //check to make sure input is okay
      if(trainer.size() != net.GetNumInputs() || target >= net.GetNumOutputs()){
        std::cout << "Input data formatted incorrectly, exiting.\n";
        exit(1);
      }
      for(int i = 0; i < trainer.size(); ++i)
        trainer[i] = (trainer[i] / 255)*.99 + .01;
      //set up target output vector
      //outputs are .01 and .99 as 0 and 1 are impossible outputs of sigmoid    
      std::vector<double> target_output(10, 0.01);
      target_output[target] = .99;
      net.TrainNet(trainer, target_output, learn_rate); 
      if(decreasing_learn && counter % num_steps_change == 0){
        ++epsilon_divisor;
        learn_rate = epsilon / epsilon_divisor;
      }
    }
    in_stream.close();
  }
}

double TestNet(NeuralNet &net, const std::string &data_file){
  std::ifstream in_stream(data_file);
  int counter = 0;
  if(!in_stream.is_open()){
    std::cout << "Failed to open training file, exiting\n";
    exit(1);
  }
  std::string token;
  int num_correct = 0;
  int total = 0;
  //get input vectors and update weights
  while(std::getline(in_stream, token)){
    if(counter % 1000 == 0){
      std::cout << counter << " instances tested.\n";
    }
    ++counter;
    std::stringstream ss(token);
    std::vector<double> tester;
    double k;
    int target;
    ss >> target;
    ss.ignore();
    //get and input, put in a vector
    while (ss >> k){
      tester.push_back(k);
      //skip the comma and space
      ss.ignore();
    }
    //check to make sure input is okay
    if(tester.size() != net.GetNumInputs() || target >= net.GetNumOutputs()){
      std::cout << "Input data formatted incorrectly, exiting.\n";
      exit(1);
    }
    std::vector<double> output = net.QueryNet(tester);
    int max_position = 0;
    double max_value = output[0];
    for(int i = 1; i < 10; ++i){
      if(max_value < output[i]){
        max_position = i;
        max_value = output[i];
      }
    }
    if(max_position == target)
      ++num_correct;
    ++total;    
  }
  in_stream.close();
  double percent_correct = ((double) num_correct) / total;
  return percent_correct;
}